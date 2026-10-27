/// <reference lib="webworker" />
/**
 * Analysis worker: keyframe decode (WebCodecs) -> streaming pass 1 -> thumbnails -> detector on a sparse
 * stride (perspective) -> pass 2 -> light ensemble -> detector on unsure samples (cascade).
 * The luma samples and the detector boxes of the last run are kept (see Cache), so that a new zone for the
 * same files is evaluated without decoding the video again.
 */
import { ALL_FORMATS, BlobSource, EncodedPacketSink, Input, type InputVideoTrack } from 'mediabunny';
import { pass2, perspectiveFromBlobs, perspectiveFromBoxes, type Pass1, type Perspective } from '../core/features';
import { cascadeProb, detectorCount, lightProb, unsureSamples, type ModelFile } from '../core/model';
import { polygonMask } from '../core/roi';
import type { AnalysisResult, AnalyzeRequest, Diagnostics, SessionFile, WorkerMessage } from '../core/session';
import { StreamingPass1 } from '../core/streaming';
import { createDetector, type Detector } from './detector';
import { drawRotated, frameToLuma } from './luma';

const MIN_SPACING = 0.9; // seconds between taken keyframes (same rule as the Python sampler)
const THUMB_W = 192;
const THUMB_H = 108;
const THUMB_BATCH = 40;
const UNSURE_SHARE = 0.3;          // about this share of the samples is uncertain (29% on the evaluated recording): guess for the time estimate
const KEYFRAME_REDECODE_S = 0.3;   // decoding one key frame again for the cascade
const STORE_PREFIX = 'futsalcut-samples-';
const MEM_BUDGET = 512 * 2 ** 20; // bytes of luma kept in memory when there is no private file system

const post = (m: WorkerMessage, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);
let cancelled = false;

self.onmessage = (ev: MessageEvent) => {
  const msg = ev.data;
  if (msg?.type === 'cancel') cancelled = true;
  if (msg?.type === 'drop') dropCache().catch(() => undefined);
  if (msg?.type === 'preview') {
    preview(msg.file as File, msg.atRatio ?? 0.1).catch((e) => post({ type: 'error', message: String(e?.message ?? e) }));
  }
  if (msg?.type === 'analyze') {
    cancelled = false;
    analyze(msg as AnalyzeRequest).catch(async (e) => {
      const byUser = cancelled || e?.message === 'cancelled';
      // after a cancel the stored samples stay usable once they are complete; after a failure nothing is trusted
      if (!byUser || (cache && !cache.complete)) await dropCache();
      if (byUser) post({ type: 'cancelled' });
      else post({ type: 'error', message: String(e?.message ?? e) });
    });
  }
};
removeStaleStores().catch(() => undefined);

interface Opened { input: Input; track: InputVideoTrack; config: VideoDecoderConfig; info: SessionFile }

async function openFiles(files: File[]): Promise<Opened[]> {
  const out: Opened[] = [];
  let offset = 0;
  for (const file of files) {
    const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error(`${file.name}: no video track`);
    const config = await track.getDecoderConfig();
    if (!config) throw new Error(`${file.name}: unsupported codec`);
    const support = await VideoDecoder.isConfigSupported(config).catch(() => ({ supported: false }));
    const first = await track.getFirstTimestamp();
    const duration = (await track.computeDuration()) - first;
    // keyframe interval from the first few key packets (metadata only)
    const sink = new EncodedPacketSink(track);
    const gaps: number[] = [];
    let k = await sink.getFirstKeyPacket({ metadataOnly: true });
    for (let i = 0; k && i < 12; i++) {
      const n = await sink.getNextKeyPacket(k, { metadataOnly: true });
      if (!n) break;
      gaps.push(n.timestamp - k.timestamp);
      k = n;
    }
    gaps.sort((a, b) => a - b);
    const info: SessionFile = {
      name: file.name, size: file.size, codec: track.codec ?? '?', codecString: config.codec,
      width: track.displayWidth, height: track.displayHeight, rotation: track.rotation, durationS: duration,
      firstTimestamp: first, offsetS: offset, supported: !!support.supported,
      medianGopS: gaps.length ? gaps[gaps.length >> 1] : 0,
    };
    out.push({ input, track, config, info });
    offset += duration;
  }
  return out;
}

/** Decode one key packet on its own (the zone editor's preview). */
async function decodeKeyAt(o: Opened, localT: number): Promise<VideoFrame | null> {
  const sink = new EncodedPacketSink(o.track);
  const pkt = await sink.getKeyPacket(o.info.firstTimestamp + localT + 0.05);
  if (!pkt) return null;
  return new Promise<VideoFrame | null>((resolve, reject) => {
    let got: VideoFrame | null = null;
    const dec = new VideoDecoder({ output: (f) => { if (got) f.close(); else got = f; }, error: reject });
    dec.configure({ ...o.config, optimizeForLatency: true });
    dec.decode(pkt.toEncodedVideoChunk());
    dec.flush().then(() => { dec.close(); resolve(got); }, reject);
  });
}

/** Random access to the key frames of one file for the cascade; the decoder stays configured between reads
 * (setting up a 4K hardware decoder for every frame costs more than decoding the frame). */
class KeyReader {
  private dec: VideoDecoder | null = null;
  private sink: EncodedPacketSink;
  private onFrame: ((f: VideoFrame) => void) | null = null;

  constructor(private o: Opened) {
    this.sink = new EncodedPacketSink(o.track);
  }

  async read(localT: number): Promise<VideoFrame | null> {
    try {
      const pkt = await this.sink.getKeyPacket(this.o.info.firstTimestamp + localT + 0.05);
      if (!pkt) return null;
      if (!this.dec || this.dec.state === 'closed') {
        this.dec = new VideoDecoder({ output: (f) => (this.onFrame ? this.onFrame(f) : f.close()), error: () => undefined });
        this.dec.configure({ ...this.o.config, optimizeForLatency: true });
      }
      let got: VideoFrame | null = null;
      this.onFrame = (f) => { if (got) f.close(); else got = f; };
      this.dec.decode(pkt.toEncodedVideoChunk());
      await this.dec.flush();
      this.onFrame = null;
      return got;
    } catch (e) {
      console.warn('key frame decode failed', e); // skip this sample; the next read sets up a fresh decoder
      this.onFrame = null;
      this.close();
      return null;
    }
  }

  close(): void {
    if (this.dec && this.dec.state !== 'closed') this.dec.close();
    this.dec = null;
  }
}

/** A display-oriented frame for the zone editor. */
async function preview(file: File, atRatio: number): Promise<void> {
  const [o] = await openFiles([file]);
  if (!o.info.supported) throw new Error(`unsupported codec ${o.info.codecString}`);
  const frame = await decodeKeyAt(o, o.info.durationS * atRatio);
  if (!frame) throw new Error('no keyframe');
  const w = 1280;
  const h = Math.round((w * o.info.height) / o.info.width);
  const canvas = new OffscreenCanvas(w, h);
  drawRotated(canvas.getContext('2d')!, frame, o.info.rotation, w, h);
  frame.close();
  const bitmap = canvas.transferToImageBitmap();
  (self as unknown as Worker).postMessage({ type: 'preview', bitmap, width: w, height: h, file: o.info }, [bitmap]);
  o.input.dispose?.();
}

/** The luma samples of one analysis, appended while decoding and read back in order for a new zone.
 * Lives in the origin private file system (a 2-hour game is ~1.7 GB), else in a bounded memory copy. */
class SampleStore {
  count = 0;
  broken = false;
  private size = 0;
  private handle: FileSystemSyncAccessHandle | null = null;
  private name = '';
  private mem: Uint8Array[] = [];

  get kind(): string { return this.broken ? 'none' : this.handle ? 'file' : 'memory'; }

  /** Called with the first sample: its size is the same for the whole session. */
  async open(size: number): Promise<void> {
    this.size = size;
    try {
      const root = await navigator.storage.getDirectory();
      this.name = `${STORE_PREFIX}${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}.bin`;
      this.handle = await (await root.getFileHandle(this.name, { create: true })).createSyncAccessHandle();
      this.handle.truncate(0);
    } catch {
      this.handle = null;
    }
  }

  append(luma: Uint8Array): void {
    if (this.broken) return;
    try {
      if (this.handle) this.handle.write(luma, { at: this.count * this.size });
      else if ((this.count + 1) * this.size <= MEM_BUDGET) this.mem.push(luma);
      else { this.broken = true; return; }
      this.count++;
    } catch {
      this.broken = true; // e.g. disk quota: this analysis just cannot be reused
    }
  }

  read(i: number): Uint8Array {
    if (!this.handle) return this.mem[i];
    const out = new Uint8Array(this.size);
    this.handle.read(out, { at: i * this.size });
    return out;
  }

  async dispose(): Promise<void> {
    this.mem = [];
    if (!this.handle) return;
    try {
      this.handle.close();
      await (await navigator.storage.getDirectory()).removeEntry(this.name);
    } catch { /* already gone */ }
    this.handle = null;
  }
}

/** Sample files left behind by an earlier page (an open one in another tab is locked and stays). */
async function removeStaleStores(): Promise<void> {
  const root = await navigator.storage.getDirectory();
  for await (const [name] of (root as unknown as { entries(): AsyncIterable<[string, unknown]> }).entries()) {
    if (name.startsWith(STORE_PREFIX)) await root.removeEntry(name).catch(() => undefined);
  }
}

/** Everything about one analysis that does not depend on the zone. */
interface Cache {
  key: string;
  opened: Opened[];
  readers: KeyReader[];
  files: SessionFile[];
  store: SampleStore;
  W: number;
  H: number;
  t: number[];
  fileIndex: number[];
  localT: number[];
  boxes: Map<number, number[][]>; // person boxes per sample index (whole frame)
  strideIdx: number[];            // samples whose boxes feed the perspective fit
  formats: Set<string>;
  exact: number;
  fallback: number;
  decodeS: number;
  detector: Detector | null;
  detectorAsked: string;
  detMs: number;                  // mean detector call, for the first time estimate of a later run
  complete: boolean;              // all samples decoded and stored
}
let cache: Cache | null = null;

async function dropCache(): Promise<void> {
  const c = cache;
  cache = null;
  if (!c) return;
  for (const r of c.readers) r.close();
  await c.detector?.release().catch(() => undefined);
  await c.store.dispose();
  for (const o of c.opened) o.input.dispose?.();
}

interface Run {
  req: AnalyzeRequest;
  model: ModelFile;
  c: Cache;
  t0: number;
  /** seconds since the analysis began */
  wall(): number;
  sessionS: number;
  detTimes: number[];
  timing: { luma: number; pass1: number; thumb: number; strideDet: number; redecode: number; replay: number };
  getDetector(): Promise<Detector | null>;
}

/** Frame -> 640x640 RGBA -> person boxes (normalized x1, y1, x2, y2, score). */
class BoxFinder {
  private ctx = new OffscreenCanvas(640, 640).getContext('2d', { willReadFrequently: true })!;
  constructor(private detector: Detector, private times: number[]) {}

  async find(frame: VideoFrame, rotation: number): Promise<number[][]> {
    drawRotated(this.ctx, frame, rotation, 640, 640);
    const px = this.ctx.getImageData(0, 0, 640, 640).data;
    const s = performance.now();
    const boxes = await this.detector.detect(px);
    this.times.push(performance.now() - s);
    return boxes;
  }
}

/** Decode every keyframe once: luma samples -> pass 1 (zone dependent) and the sample store, thumbnails, sparse detector boxes. */
async function decodeAll(run: Run): Promise<Pass1[]> {
  const { req, c, timing } = run;
  const detector = await run.getDetector();
  const finder = detector ? new BoxFinder(detector, run.detTimes) : null;
  const stride = run.model.detector.perspective_stride;
  const thumbCanvas = new OffscreenCanvas(THUMB_W, THUMB_H);
  const thumbCtx = thumbCanvas.getContext('2d')!;
  const st = { sp: null as StreamingPass1 | null }; // assigned inside the async frame handler (closure), hence a holder object
  let thumbs: Blob[] = [];
  let thumbStart = 0;
  const lap = (since: number) => (performance.now() - since) / 1000;

  for (let fi = 0; fi < c.opened.length; fi++) {
    const o = c.opened[fi];
    const sink = new EncodedPacketSink(o.track);
    const frames: VideoFrame[] = [];
    let failure: unknown = null;
    let wake: (() => void) | null = null;
    const dec = new VideoDecoder({ output: (f) => { frames.push(f); wake?.(); }, error: (e) => { failure = e; wake?.(); } });
    dec.configure({ ...o.config, optimizeForLatency: true });
    let lastTaken = -Infinity;

    const handle = async (f: VideoFrame) => {
      try {
        const lt = f.timestamp / 1e6 - o.info.firstTimestamp;
        let s = performance.now();
        const luma = await frameToLuma(f, o.info.rotation);
        timing.luma += lap(s);
        c.formats.add(luma.format);
        if (luma.path === 'exact') c.exact++; else c.fallback++;
        if (!st.sp) {
          c.W = luma.w;
          c.H = luma.h;
          st.sp = new StreamingPass1(c.W, c.H, polygonMask(req.zone, c.W, c.H));
          await c.store.open(c.W * c.H);
        }
        const idx = st.sp.count;
        c.store.append(luma.data);
        s = performance.now();
        st.sp.push(luma.data);
        timing.pass1 += lap(s);
        c.t.push(o.info.offsetS + lt);
        c.fileIndex.push(fi);
        c.localT.push(lt);
        s = performance.now();
        drawRotated(thumbCtx, f, o.info.rotation, THUMB_W, THUMB_H);
        thumbs.push(await thumbCanvas.convertToBlob({ type: 'image/jpeg', quality: 0.7 }));
        if (thumbs.length >= THUMB_BATCH) {
          post({ type: 'thumbs', start: thumbStart, blobs: thumbs });
          thumbStart += thumbs.length;
          thumbs = [];
        }
        timing.thumb += lap(s);
        if (finder && idx % stride === 0) {
          s = performance.now();
          c.boxes.set(idx, await finder.find(f, o.info.rotation));
          c.strideIdx.push(idx);
          timing.strideDet += lap(s);
        }
      } finally {
        f.close();
      }
    };

    const d0 = performance.now();
    try {
      for (let pkt = await sink.getFirstKeyPacket(); pkt; pkt = await sink.getNextKeyPacket(pkt)) {
        if (cancelled) throw new Error('cancelled');
        if (failure) throw failure;
        if (pkt.timestamp - lastTaken + 1e-3 < MIN_SPACING) continue;
        lastTaken = pkt.timestamp;
        dec.decode(pkt.toEncodedVideoChunk());
        while (dec.decodeQueueSize > 4 || frames.length) {
          if (frames.length) await handle(frames.shift()!);
          else await new Promise<void>((r) => { wake = r; dec.addEventListener('dequeue', () => r(), { once: true }); });
          if (failure) throw failure;
        }
        const detMs = run.detTimes.length >= 2 ? run.detTimes.reduce((a, b) => a + b, 0) / run.detTimes.length : 0;
        post({ type: 'progress', stage: 'decode', done: c.t.length, total: 0, sessionDoneS: c.t.length ? c.t[c.t.length - 1] : 0,
          wallS: run.wall(), detGuessS: detMs ? UNSURE_SHARE * run.sessionS * (detMs / 1000 + KEYFRAME_REDECODE_S) : undefined });
      }
      await dec.flush();
      while (frames.length) await handle(frames.shift()!);
    } finally {
      for (const f of frames) f.close();
      if (dec.state !== 'closed') dec.close();
    }
    c.decodeS += lap(d0);
  }
  if (thumbs.length) post({ type: 'thumbs', start: thumbStart, blobs: thumbs });
  if (!st.sp) throw new Error('no samples decoded');
  c.complete = true;
  const s = performance.now();
  const p1 = st.sp.end();
  timing.pass1 += lap(s);
  return p1;
}

/** Pass 1 again for a new zone, from the stored samples. */
async function replayAll(run: Run): Promise<Pass1[]> {
  const { c } = run;
  const sp = new StreamingPass1(c.W, c.H, polygonMask(run.req.zone, c.W, c.H));
  const n = c.store.count;
  const r0 = performance.now();
  for (let i = 0; i < n; i++) {
    if (cancelled) throw new Error('cancelled');
    sp.push(c.store.read(i));
    if (i % 20 === 19 || i === n - 1) {
      const stageWallS = (performance.now() - r0) / 1000;
      post({ type: 'progress', stage: 'replay', done: i + 1, total: n, sessionDoneS: c.t[i], wallS: run.wall(), stageWallS });
      await new Promise((r) => setTimeout(r, 0)); // let a cancel message through
    }
  }
  const p1 = sp.end();
  run.timing.replay = (performance.now() - r0) / 1000;
  return p1;
}

async function analyze(req: AnalyzeRequest): Promise<void> {
  const t0 = performance.now();
  const model = req.modelJson as ModelFile;
  const key = req.files.map((f) => `${f.name}:${f.size}:${f.lastModified}`).join('|');
  const reuse = req.reuse !== false && !!cache?.complete && !cache.store.broken && cache.key === key;
  if (!reuse) await dropCache();
  if (!cache) {
    const opened = await openFiles(req.files);
    cache = {
      key, opened, readers: opened.map((o) => new KeyReader(o)), files: opened.map((o) => o.info), store: new SampleStore(),
      W: 0, H: 0, t: [], fileIndex: [], localT: [], boxes: new Map(), strideIdx: [], formats: new Set(), exact: 0, fallback: 0,
      decodeS: 0, detector: null, detectorAsked: '', detMs: 0, complete: false,
    };
  }
  const c = cache;
  const files = c.files;
  post({ type: 'files', files });
  const bad = files.filter((f) => !f.supported);
  if (bad.length) throw new Error(`이 브라우저에서 디코딩할 수 없는 코덱: ${bad.map((f) => `${f.name} (${f.codecString})`).join(', ')}`);

  const run: Run = {
    req, model, c, t0, wall: () => (performance.now() - t0) / 1000, sessionS: files.reduce((s, f) => s + f.durationS, 0), detTimes: [],
    timing: { luma: 0, pass1: 0, thumb: 0, strideDet: 0, redecode: 0, replay: 0 },
    async getDetector() {
      if (req.detector === 'off') return null;
      const asked = `${req.detectorBackend ?? 'wasm'}/${req.ortThreads ?? 0}`;
      if (c.detector && c.detectorAsked !== asked) {
        await c.detector.release().catch(() => undefined);
        c.detector = null;
      }
      if (!c.detector) {
        try {
          c.detector = await createDetector(req.modelUrl, req.detectorBackend ?? 'wasm', req.ortThreads,
            { webgpuWaitMs: req.webgpuWaitS ? req.webgpuWaitS * 1000 : undefined });
          c.detectorAsked = asked;
        } catch (e) {
          console.warn('detector unavailable', e);
        }
      }
      return c.detector;
    },
  };

  const p1 = reuse ? await replayAll(run) : await decodeAll(run);

  // pass 2
  const f0 = performance.now();
  const zoneTop = Math.min(...req.zone.map((p) => p[1])) * c.H;
  const strideBoxes = c.strideIdx.slice().sort((a, b) => a - b).map((i) => c.boxes.get(i)!);
  const persp: Perspective = perspectiveFromBoxes(strideBoxes, c.H, zoneTop)
    ?? perspectiveFromBlobs(p1, zoneTop) ?? { a: 0.5, c: 0, source: 'default' };
  const scores = pass2(p1, persp, c.H);
  const light = lightProb(model, scores);
  const featuresS = (performance.now() - f0) / 1000;

  // cascade: detector only where the light ensemble is unsure
  const detScores = new Map<number, number>();
  const count = (boxes: number[][]) => detectorCount(boxes, req.zone, model.detector.min_score);
  let prob = light;
  const ds0 = performance.now();
  if (req.detector !== 'off') {
    const unsure = unsureSamples(model, light);
    for (const i of [...c.strideIdx, ...unsure]) {
      const b = c.boxes.get(i);
      if (b) detScores.set(i, count(b));
    }
    const todo = unsure.filter((i) => !c.boxes.has(i));
    const detector = todo.length || !c.boxes.size ? await run.getDetector() : c.detector;
    if (detector && todo.length) {
      const finder = new BoxFinder(detector, run.detTimes);
      const hint = (run.detTimes.length ? run.detTimes.reduce((a, b) => a + b, 0) / run.detTimes.length : c.detMs || 1500) / 1000 + KEYFRAME_REDECODE_S;
      const progress = (done: number) => post({ type: 'progress', stage: 'detector', done, total: todo.length, sessionDoneS: run.sessionS,
        wallS: run.wall(), stageWallS: (performance.now() - ds0) / 1000, itemHintS: hint });
      progress(0);
      const start = (i: number) => c.readers[c.fileIndex[i]].read(c.localT[i]);
      let next: Promise<VideoFrame | null> | null = start(todo[0]);
      try {
        for (let k = 0; k < todo.length; k++) {
          if (cancelled) throw new Error('cancelled');
          const i = todo[k];
          const s = performance.now();
          const frame = await next;
          next = k + 1 < todo.length ? start(todo[k + 1]) : null; // decodes while the detector works on this one
          run.timing.redecode += (performance.now() - s) / 1000;
          if (frame) {
            try {
              const boxes = await finder.find(frame, files[c.fileIndex[i]].rotation);
              c.boxes.set(i, boxes);
              detScores.set(i, count(boxes));
            } finally {
              frame.close();
            }
          }
          // samples are in file order: a 4K hardware decoder that is no longer needed should not stay allocated
          if (k + 1 >= todo.length || c.fileIndex[todo[k + 1]] !== c.fileIndex[i]) c.readers[c.fileIndex[i]].close();
          progress(k + 1);
        }
      } finally {
        (await next?.catch(() => null))?.close(); // a prefetched frame nobody used
        for (const r of c.readers) r.close();
      }
    }
    if (detector || c.boxes.size) prob = cascadeProb(model, scores, light, detScores);
    if (run.detTimes.length) c.detMs = run.detTimes.reduce((a, b) => a + b, 0) / run.detTimes.length;
  }
  const wallS = run.wall();
  const detector = c.detector;
  const diagnostics: Diagnostics = {
    wallS, decodeS: reuse ? 0 : c.decodeS, featuresS, detectorS: (performance.now() - ds0) / 1000, samples: c.t.length, sessionS: run.sessionS,
    xRealtime: run.sessionS / wallS, frameFormats: [...c.formats], exactPath: c.exact, fallbackPath: c.fallback,
    detectorBackend: detector?.backend ?? (c.boxes.size ? 'cached' : 'off'), detectorFallbackReason: detector?.fallbackReason ?? null, detectorGpuCheck: detector?.gpuCheck ?? null,
    detectorCalls: run.detTimes.length, detectorThreads: detector?.threads, crossOriginIsolated: self.crossOriginIsolated,
    detectorMsPerCall: run.detTimes.length ? run.detTimes.reduce((a, b) => a + b, 0) / run.detTimes.length : 0,
    reused: reuse, replayS: run.timing.replay, lumaS: run.timing.luma, pass1S: run.timing.pass1, thumbS: run.timing.thumb,
    strideDetectorS: run.timing.strideDet, redecodeS: run.timing.redecode, sampleStore: c.store.kind,
    perspective: persp, userAgent: navigator.userAgent,
  };
  const result: AnalysisResult = {
    files, t: Float64Array.from(c.t), fileIndex: Int32Array.from(c.fileIndex), localT: Float64Array.from(c.localT), scores, light,
    prob, detector: [...detScores.entries()].sort((a, b) => a[0] - b[0]), perspective: persp, diagnostics,
  };
  post({ type: 'result', result });
}
