/// <reference lib="webworker" />
/**
 * Analysis worker: keyframe decode (WebCodecs) and copy of each frame's planes -> luma samples, shrunk (with the thumbnail) in a
 * luma helper worker, streaming pass 1 in another helper, the detector (a worker of its own) on a sparse stride (perspective)
 * -> pass 2 -> light ensemble -> detector on unsure samples (cascade): the frame for the next detector call is prepared here
 * while the detector works on the last one, and a graphics-chip detector gets a CPU detector beside it where there are cores.
 * The luma samples and the detector boxes of the last run are kept (see Cache), so that a new zone for the
 * same files is evaluated without decoding the video again.
 */
import { ALL_FORMATS, BlobSource, EncodedPacketSink, Input, type InputVideoTrack } from 'mediabunny';
import { pass2, perspectiveFromBlobs, perspectiveFromBoxes, type Pass1, type Perspective } from '../core/features';
import { Limiter } from '../core/limiter';
import { cascadeProb, detectorCount, lightProb, unsureSamples, type ModelFile } from '../core/model';
import { polygonMask } from '../core/roi';
import type { AnalysisResult, AnalyzeRequest, Diagnostics, SessionFile, WorkerMessage } from '../core/session';
import { StreamingPass1 } from '../core/streaming';
import { createPipelinedDetector, type PipelinedDetector } from './detector-client';
import { canThumbFromPlanes, copyPlanes, drawRotated, frameToLuma, lumaClock, type LumaSample } from './luma';
import { openLumaWorker, type LumaDone, type LumaWorker } from './luma-client';
import { openSide, type SideWork } from './side-client';
import { FrameThumbs, ThumbBatch } from './thumbs';

const MIN_SPACING = 0.9; // seconds between taken keyframes (same rule as the Python sampler)
const IN_FLIGHT = 3;               // cascade: detector calls handed over before the first answer is back (detector in its own worker)
const SECOND_IN_FLIGHT = 2;        // the same for the CPU detector that helps a graphics-chip detector in the cascade
const SECOND_MIN_CORES = 8;        // logical cores needed before a CPU detector joins a graphics-chip detector
const LUMA_AHEAD = 3;              // copied frames the luma helper may still be shrinking while the next one is copied (about 12 MB each at 4K)
const STRIDE_QUEUE = 24;           // sparse calls while decoding: frames (1.6 MB each) that may wait for the detector, which is still being set up at first
const UNSURE_SHARE = 0.3;          // about this share of the samples is uncertain (29% on the evaluated recording): guess for the time estimate
const KEYFRAME_REDECODE_S = 0.3;   // decoding one key frame again for the cascade (the detector runs in this thread then: no overlap)
const OVERLAPPED_PREP_S = 0.05;    // what a cascade item costs on top of the detector call when the detector has a worker of its own
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
  detector: PipelinedDetector | null;
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
  /** milliseconds per detector call (inference only), for the pace estimate and the diagnostics; detTimes2: the helping CPU detector */
  detTimes: number[];
  detTimes2: number[];
  timing: { luma: number; pass1: number; thumb: number; strideDet: number; redecode: number; replay: number; prep: number; setup: number;
    side: number; tail: number; lumaWait: number; shrinkHelper: number; thumbHelper: number; secondSetup: number };
  /** whether pass 1 ran in a helper worker, and whether the luma helper shrank the samples (null: this run did not decode) */
  sideRemote: boolean | null;
  lumaRemote: boolean | null;
  /** cascade calls answered by the main detector and by the helping CPU detector, and the latter's backend if it ran */
  split: { primary: number; secondary: number };
  secondBackend: string | null;
  getDetector(): Promise<PipelinedDetector | null>;
}

/** Frame -> the 640x640 RGBA pixels the detector takes. Drawing a 4K frame small and reading it back costs about 0.1 s. */
class FramePrep {
  private ctx = new OffscreenCanvas(640, 640).getContext('2d', { willReadFrequently: true })!;
  constructor(private timing: Run['timing']) {}

  pixels(frame: VideoFrame, rotation: number): Uint8ClampedArray {
    const p0 = performance.now();
    drawRotated(this.ctx, frame, rotation, 640, 640);
    const px = this.ctx.getImageData(0, 0, 640, 640).data;
    this.timing.prep += (performance.now() - p0) / 1000;
    return px;
  }
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

/** Counts of probabilities in ten bins of width 0.1 (diagnostics: how sure the light ensemble is). */
function histogram(p: ArrayLike<number>): number[] {
  const bins = new Array<number>(10).fill(0);
  for (let i = 0; i < p.length; i++) bins[Math.min(9, Math.max(0, Math.floor(p[i] * 10)))]++;
  return bins;
}

/** Seconds one cascade item takes: the detector call, plus the frame work that does not overlap with it (all of it if the detector runs in this thread). */
const cascadeItemS = (detMs: number, overlapped: boolean) => detMs / 1000 + (overlapped ? OVERLAPPED_PREP_S : KEYFRAME_REDECODE_S);

/** Decode every keyframe once: luma samples -> the sample store; shrinking and thumbnails in the luma helper, pass 1 (zone
 * dependent) in the side helper; sparse detector boxes. Without helpers the same work happens here. */
async function decodeAll(run: Run): Promise<Pass1[]> {
  const parallel = run.req.parallel !== false;
  const [side, lumaW] = await Promise.all([
    openSide(parallel, (start, blobs) => post({ type: 'thumbs', start, blobs })),
    parallel ? openLumaWorker() : Promise.resolve(null),
  ]);
  run.sideRemote = side.remote;
  run.lumaRemote = !!lumaW;
  try {
    return await decodeWith(run, side, lumaW);
  } finally {
    side.close();
    lumaW?.close();
  }
}

/** A sample on its way through the luma helper (or already shrunk here), with the thumbnail made here if the helper cannot. */
interface PendingSample { idx: number; job: Promise<LumaDone> | null; inline: LumaSample | null; thumb: Promise<Blob> | null }

async function decodeWith(run: Run, side: SideWork, lumaW: LumaWorker | null): Promise<Pass1[]> {
  const { req, c, timing } = run;
  const detectorP = run.getDetector(); // set up while the decoding goes on: the frames meant for it wait until it is ready
  const prep = req.detector === 'off' ? null : new FramePrep(timing);
  const stride = run.model.detector.perspective_stride;
  const calls = new Limiter(STRIDE_QUEUE); // the same bound before and after the detector is ready, or the decoding would stop to let the backlog drain
  let started = false;
  const lap = (since: number) => (performance.now() - since) / 1000;
  const pending: PendingSample[] = []; // in sample order
  const spare: ArrayBuffer[] = [];     // copy buffers the luma helper gave back
  const frameThumbs = new FrameThumbs();
  const thumbBatch = new ThumbBatch((start, blobs) => post({ type: 'thumbs', start, blobs }));

  /** A finished luma sample, in sample order: the first one fixes the size; it goes into the store. */
  const accept = async (luma: LumaSample) => {
    c.formats.add(luma.format);
    if (luma.path === 'exact') c.exact++; else c.fallback++;
    if (!started) {
      started = true;
      c.W = luma.w;
      c.H = luma.h;
      side.start(c.W, c.H, req.zone);
      await c.store.open(c.W * c.H);
    }
    c.store.append(luma.data);
  };

  /** The oldest sample on its way through the luma helper: wait for it, store it, hand it to pass 1, deliver its thumbnail. */
  const flushOne = async () => {
    const p = pending.shift()!;
    let s = performance.now();
    let luma = p.inline;
    let thumb: Blob | null = null;
    if (p.job) {
      const d = await p.job;
      spare.push(d.buf);
      timing.shrinkHelper += d.shrinkS;
      timing.thumbHelper += d.thumbS;
      luma = { data: d.luma, w: d.w, h: d.h, path: 'exact', format: d.format };
      thumb = d.thumb;
    }
    if (p.thumb) thumb = await p.thumb;
    timing.lumaWait += lap(s);
    await accept(luma!);
    s = performance.now();
    await side.push(luma!.data, null, 0); // pass 1 only: the thumbnail came with the luma
    timing.side += lap(s);
    if (thumb) thumbBatch.add(p.idx, thumb);
  };

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
        const idx = c.t.length;
        let s = performance.now();
        if (lumaW) {
          // copy here (that needs the frame); the helper shrinks it while the next frame is decoded and copied
          const raw = await copyPlanes(f, spare.pop());
          timing.luma += lap(s);
          if (raw) {
            const inHelper = canThumbFromPlanes(raw);
            pending.push({ idx, job: lumaW.shrink(idx, raw, o.info.rotation, inHelper), inline: null,
              thumb: inHelper ? null : frameThumbs.make(f, o.info.rotation) });
          } else { // RGB or GPU-only frame: the canvas path, here
            s = performance.now();
            const inline = await frameToLuma(f, o.info.rotation);
            timing.luma += lap(s);
            pending.push({ idx, job: null, inline, thumb: frameThumbs.make(f, o.info.rotation) });
          }
        } else {
          const luma = await frameToLuma(f, o.info.rotation);
          timing.luma += lap(s);
          await accept(luma);
          s = performance.now();
          await side.push(luma.data, f, o.info.rotation); // pass 1 and the thumbnail; with a helper worker they run beside the decoding
          timing.side += lap(s);
        }
        c.t.push(o.info.offsetS + lt);
        c.fileIndex.push(fi);
        c.localT.push(lt);
        if (prep && idx % stride === 0) {
          s = performance.now();
          const px = prep.pixels(f, o.info.rotation);
          c.strideIdx.push(idx);
          await calls.run(async () => { // the answer comes in while decoding goes on; a failure surfaces in the next run() or in drain()
            const d = await detectorP;
            if (!d) return;
            const det = await d.submit(px);
            run.detTimes.push(det.ms);
            c.boxes.set(idx, det.boxes);
          });
          timing.strideDet += lap(s);
        }
      } finally {
        f.close();
      }
      while (pending.length > LUMA_AHEAD) await flushOne(); // after the frame is closed: the decoder can reuse it meanwhile
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
          wallS: run.wall(), detGuessS: detMs ? UNSURE_SHARE * run.sessionS * cascadeItemS(detMs, !!c.detector?.remote) : undefined });
      }
      await dec.flush();
      while (frames.length) await handle(frames.shift()!);
    } finally {
      for (const f of frames) f.close();
      if (dec.state !== 'closed') dec.close();
    }
    c.decodeS += lap(d0);
  }
  const w0 = performance.now();
  while (pending.length) await flushOne();
  thumbBatch.flush();
  if (!started) throw new Error('no samples decoded');
  c.complete = true;
  await calls.drain(); // sparse detector calls that are still running
  const done = await side.end();
  timing.tail = lap(w0);
  timing.pass1 = done.pass1S;
  timing.thumb = done.thumbS;
  c.strideIdx = c.strideIdx.filter((i) => c.boxes.has(i)); // without a detector there are no boxes
  return done.pass1;
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
  lumaClock.copy = 0;
  lumaClock.shrink = 0;
  const shipped = req.modelJson as ModelFile;
  const model: ModelFile = req.cascadeBand ? { ...shipped, detector: { ...shipped.detector, cascade: req.cascadeBand } } : shipped;
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

  let creating: Promise<PipelinedDetector | null> | null = null; // one setup per run, however often the detector is asked for
  const run: Run = {
    req, model, c, t0, wall: () => (performance.now() - t0) / 1000, sessionS: files.reduce((s, f) => s + f.durationS, 0), detTimes: [], detTimes2: [],
    timing: { luma: 0, pass1: 0, thumb: 0, strideDet: 0, redecode: 0, replay: 0, prep: 0, setup: 0, side: 0, tail: 0,
      lumaWait: 0, shrinkHelper: 0, thumbHelper: 0, secondSetup: 0 },
    sideRemote: null,
    lumaRemote: null,
    split: { primary: 0, secondary: 0 },
    secondBackend: null,
    getDetector() {
      creating ??= (async () => {
        if (req.detector === 'off') return null;
        const asked = `${req.detectorBackend ?? 'auto'}/${req.ortThreads ?? 0}`;
        if (c.detector && c.detectorAsked !== asked) {
          await c.detector.release().catch(() => undefined);
          c.detector = null;
        }
        if (!c.detector) {
          const s0 = performance.now();
          try {
            const made = await createPipelinedDetector(req.modelUrl, req.detectorBackend ?? 'auto', req.ortThreads,
              { webgpuWaitMs: req.webgpuWaitS ? req.webgpuWaitS * 1000 : undefined, anyGpu: req.anyGpu }, req.parallel !== false);
            if (cache !== c) { // cancelled or dropped while it was being set up: nobody would release it
              await made.release().catch(() => undefined);
              return null;
            }
            c.detector = made;
            c.detectorAsked = asked;
          } catch (e) {
            console.warn('detector unavailable', e);
          }
          run.timing.setup += (performance.now() - s0) / 1000;
        }
        return c.detector;
      })();
      return creating;
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
  let unsureShare: number | null = null;
  const ds0 = performance.now();
  if (req.detector !== 'off') {
    const unsure = unsureSamples(model, light);
    unsureShare = light.length ? unsure.length / light.length : 0;
    for (const i of [...c.strideIdx, ...unsure]) {
      const b = c.boxes.get(i);
      if (b) detScores.set(i, count(b));
    }
    const todo = unsure.filter((i) => !c.boxes.has(i));
    const detector = todo.length || !c.boxes.size ? await run.getDetector() : c.detector;
    if (detector && todo.length) {
      const prep = new FramePrep(run.timing);
      const calls = new Limiter(detector.remote ? IN_FLIGHT : 1);
      const meanMs = run.detTimes.length ? run.detTimes.reduce((a, b) => a + b, 0) / run.detTimes.length : c.detMs || 1500;
      const hint = cascadeItemS(meanMs, detector.remote);
      let finished = 0; // items done: detected, or skipped because their frame could not be decoded
      let over = false; // the loop has ended (done, cancelled or failed): late answers must not post progress any more
      const progress = (done: number) => post({ type: 'progress', stage: 'detector', done, total: todo.length, sessionDoneS: run.sessionS,
        wallS: run.wall(), stageWallS: (performance.now() - ds0) / 1000, itemHintS: hint });
      const finish = () => { finished++; if (!over) progress(finished); };
      progress(0);
      // A CPU detector beside a graphics-chip detector: the processor has little else to do in this stage. It joins once it is
      // set up and takes a frame only when the graphics chip has no free slot. (Test override: two CPU detectors.)
      const wantSecond = detector.remote && (!!req.dualDetector
        || (detector.backend === 'webgpu' && (navigator.hardwareConcurrency || 0) >= SECOND_MIN_CORES));
      const calls2 = new Limiter(SECOND_IN_FLIGHT);
      const helper = { d: null as PipelinedDetector | null };
      const s2 = performance.now();
      const secondP = wantSecond
        ? createPipelinedDetector(req.modelUrl, 'wasm', req.ortThreads, {}, true).then((d) => {
          run.timing.secondSetup = (performance.now() - s2) / 1000;
          if (over) { void d.release(); return null; } // the stage ended before it was ready
          helper.d = d;
          run.secondBackend = d.backend;
          return d;
        }, (e) => { console.warn('second detector unavailable', e); return null; })
        : null;
      const start = (i: number) => c.readers[c.fileIndex[i]].read(c.localT[i]);
      let next: Promise<VideoFrame | null> | null = start(todo[0]);
      try {
        for (let k = 0; k < todo.length; k++) {
          if (cancelled) throw new Error('cancelled');
          const i = todo[k];
          const s = performance.now();
          const frame = await next;
          next = k + 1 < todo.length ? start(todo[k + 1]) : null; // decodes while this frame is prepared and the detector works
          run.timing.redecode += (performance.now() - s) / 1000;
          if (frame) {
            let px: Uint8ClampedArray;
            try {
              px = prep.pixels(frame, files[c.fileIndex[i]].rotation);
            } finally {
              frame.close();
            }
            // a detector works on this one while the loop goes on to prepare the next frame
            const detect = (d: PipelinedDetector, which: 'primary' | 'secondary') => async () => {
              const det = await d.submit(px);
              (which === 'primary' ? run.detTimes : run.detTimes2).push(det.ms);
              run.split[which]++;
              c.boxes.set(i, det.boxes);
              detScores.set(i, count(det.boxes));
              finish();
            };
            for (;;) {
              const second = helper.d;
              if (!second || calls.hasRoom || calls.failed) { await calls.run(detect(detector, 'primary')); break; } // run() throws a failure
              if (calls2.failed) await calls2.drain(); // throws its failure
              if (calls2.hasRoom) { await calls2.run(detect(second, 'secondary')); break; }
              await Promise.race([calls.untilRoom(), calls2.untilRoom()]);
            }
          } else {
            finish();
          }
          // samples are in file order: a 4K hardware decoder that is no longer needed should not stay allocated
          if (k + 1 >= todo.length || c.fileIndex[todo[k + 1]] !== c.fileIndex[i]) c.readers[c.fileIndex[i]].close();
        }
        await calls.drain();
        await calls2.drain();
      } finally {
        over = true;
        (await next?.catch(() => null))?.close(); // a prefetched frame nobody used
        for (const r of c.readers) r.close();
        void secondP?.then((d) => d?.release()); // its threads and model are not kept for a later run
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
    detectorBackend: detector?.backend ?? (c.boxes.size ? 'cached' : 'off'), detectorAsked: req.detectorBackend ?? 'auto', detectorFallbackReason: detector?.fallbackReason ?? null, detectorGpuCheck: detector?.gpuCheck ?? null,
    detectorCalls: run.detTimes.length + run.detTimes2.length, detectorThreads: detector?.threads, crossOriginIsolated: self.crossOriginIsolated,
    detectorMsPerCall: mean(run.detTimes),
    reused: reuse, replayS: run.timing.replay, lumaS: run.timing.luma, pass1S: run.timing.pass1, thumbS: run.timing.thumb + run.timing.thumbHelper,
    strideDetectorS: run.timing.strideDet, redecodeS: run.timing.redecode, sampleStore: c.store.kind,
    detectorSetupS: run.timing.setup, lumaCopyS: reuse ? 0 : lumaClock.copy, lumaShrinkS: reuse ? 0 : lumaClock.shrink + run.timing.shrinkHelper,
    frameToBoxesS: run.timing.prep,
    detectorWorker: detector?.remote, sideWorker: run.sideRemote ?? undefined, detectorIsolated: detector?.isolated,
    lumaWorker: run.lumaRemote ?? undefined, lumaWaitS: run.timing.lumaWait,
    sideWaitS: run.timing.side, tailWaitS: run.timing.tail,
    detectorSplit: run.split, secondDetector: run.secondBackend, secondDetectorMsPerCall: mean(run.detTimes2), secondDetectorSetupS: run.timing.secondSetup,
    zone: req.zone.map(([x, y]) => [Math.round(x * 1000) / 1000, Math.round(y * 1000) / 1000]),
    lightHistogram: histogram(light), unsureShare, cascadeBand: model.detector.cascade,
    perspective: persp, userAgent: navigator.userAgent,
  };
  const result: AnalysisResult = {
    files, t: Float64Array.from(c.t), fileIndex: Int32Array.from(c.fileIndex), localT: Float64Array.from(c.localT), scores, light,
    prob, detector: [...detScores.entries()].sort((a, b) => a[0] - b[0]), perspective: persp, diagnostics,
  };
  post({ type: 'result', result });
}
