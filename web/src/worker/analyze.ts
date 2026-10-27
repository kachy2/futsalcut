/// <reference lib="webworker" />
/**
 * Analysis worker: keyframe decode (WebCodecs) -> streaming pass 1 -> thumbnails -> detector on a sparse
 * stride (perspective) -> pass 2 -> light ensemble -> detector on unsure samples (cascade).
 */
import { ALL_FORMATS, BlobSource, EncodedPacketSink, Input, type InputVideoTrack } from 'mediabunny';
import { pass2, perspectiveFromBlobs, perspectiveFromBoxes, type Perspective } from '../core/features';
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

const post = (m: WorkerMessage, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);
let cancelled = false;

self.onmessage = (ev: MessageEvent) => {
  const msg = ev.data;
  if (msg?.type === 'cancel') cancelled = true;
  if (msg?.type === 'preview') {
    preview(msg.file as File, msg.atRatio ?? 0.1).catch((e) => post({ type: 'error', message: String(e?.message ?? e) }));
  }
  if (msg?.type === 'analyze') {
    cancelled = false;
    analyze(msg as AnalyzeRequest).catch((e) => post({ type: 'error', message: String(e?.message ?? e) }));
  }
};

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

/** Decode one key packet on its own (random access for the cascade). */
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

async function analyze(req: AnalyzeRequest): Promise<void> {
  const t0 = performance.now();
  const model = req.modelJson as ModelFile;
  const opened = await openFiles(req.files);
  const files = opened.map((o) => o.info);
  post({ type: 'files', files });
  const bad = files.filter((f) => !f.supported);
  if (bad.length) throw new Error(`이 브라우저에서 디코딩할 수 없는 코덱: ${bad.map((f) => `${f.name} (${f.codecString})`).join(', ')}`);
  const sessionS = files.reduce((s, f) => s + f.durationS, 0);

  let detector: Detector | null = null;
  const detTimes: number[] = [];
  if (req.detector !== 'off') {
    try {
      detector = await createDetector(req.modelUrl, req.detectorBackend ?? 'wasm');
    } catch (e) {
      console.warn('detector unavailable', e);
    }
  }
  const stride = model.detector.perspective_stride;
  const detCanvas = new OffscreenCanvas(640, 640);
  const detCtx = detCanvas.getContext('2d', { willReadFrequently: true })!;
  const thumbCanvas = new OffscreenCanvas(THUMB_W, THUMB_H);
  const thumbCtx = thumbCanvas.getContext('2d')!;
  const runDetector = async (frame: VideoFrame, rotation: number): Promise<number[][]> => {
    drawRotated(detCtx, frame, rotation, 640, 640);
    const px = detCtx.getImageData(0, 0, 640, 640).data;
    const s = performance.now();
    const boxes = await detector!.detect(px);
    detTimes.push(performance.now() - s);
    return boxes;
  };

  // assigned inside the async frame handler (closure), hence a holder object
  const st = { W: 0, H: 0, sp: null as StreamingPass1 | null };
  const t: number[] = [];
  const fileIndex: number[] = [];
  const localT: number[] = [];
  const boxes = new Map<number, number[][]>();
  const formats = new Set<string>();
  let exact = 0;
  let fallback = 0;
  let thumbs: Blob[] = [];
  let thumbStart = 0;
  let decodeS = 0;

  for (let fi = 0; fi < opened.length; fi++) {
    const o = opened[fi];
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
        const luma = await frameToLuma(f, o.info.rotation);
        formats.add(luma.format);
        if (luma.path === 'exact') exact++; else fallback++;
        if (!st.sp) {
          st.W = luma.w;
          st.H = luma.h;
          st.sp = new StreamingPass1(st.W, st.H, polygonMask(req.zone, st.W, st.H));
        }
        const idx = st.sp.count;
        st.sp.push(luma.data);
        t.push(o.info.offsetS + lt);
        fileIndex.push(fi);
        localT.push(lt);
        drawRotated(thumbCtx, f, o.info.rotation, THUMB_W, THUMB_H);
        thumbs.push(await thumbCanvas.convertToBlob({ type: 'image/jpeg', quality: 0.7 }));
        if (thumbs.length >= THUMB_BATCH) {
          post({ type: 'thumbs', start: thumbStart, blobs: thumbs });
          thumbStart += thumbs.length;
          thumbs = [];
        }
        if (detector && idx % stride === 0) boxes.set(idx, await runDetector(f, o.info.rotation));
      } finally {
        f.close();
      }
    };

    const d0 = performance.now();
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
      post({ type: 'progress', stage: 'decode', done: t.length, total: 0, sessionDoneS: t.length ? t[t.length - 1] : 0,
        wallS: (performance.now() - t0) / 1000 });
    }
    await dec.flush();
    while (frames.length) await handle(frames.shift()!);
    dec.close();
    decodeS += (performance.now() - d0) / 1000;
  }
  if (thumbs.length) post({ type: 'thumbs', start: thumbStart, blobs: thumbs });
  const sp = st.sp;
  const H = st.H;
  if (!sp) throw new Error('no samples decoded');

  // pass 2
  const f0 = performance.now();
  const p1 = sp.end();
  const zoneTop = Math.min(...req.zone.map((p) => p[1])) * H;
  const strideBoxes = [...boxes.entries()].sort((a, b) => a[0] - b[0]).map(([, b]) => b);
  const persp: Perspective = perspectiveFromBoxes(strideBoxes, H, zoneTop)
    ?? perspectiveFromBlobs(p1, zoneTop) ?? { a: 0.5, c: 0, source: 'default' };
  const scores = pass2(p1, persp, H);
  const light = lightProb(model, scores);
  const featuresS = (performance.now() - f0) / 1000;

  // cascade: detector only where the light ensemble is unsure
  const detScores = new Map<number, number>();
  for (const [i, b] of boxes) detScores.set(i, detectorCount(b, req.zone, model.detector.min_score));
  let prob = light;
  const ds0 = performance.now();
  if (detector) {
    const todo = unsureSamples(model, light).filter((i) => !detScores.has(i));
    for (let k = 0; k < todo.length; k++) {
      if (cancelled) throw new Error('cancelled');
      const i = todo[k];
      const frame = await decodeKeyAt(opened[fileIndex[i]], localT[i]);
      if (frame) {
        try {
          detScores.set(i, detectorCount(await runDetector(frame, opened[fileIndex[i]].info.rotation), req.zone, model.detector.min_score));
        } finally {
          frame.close();
        }
      }
      post({ type: 'progress', stage: 'detector', done: k + 1, total: todo.length, sessionDoneS: sessionS, wallS: (performance.now() - t0) / 1000 });
    }
    prob = cascadeProb(model, scores, light, detScores);
  }
  await detector?.release();
  const wallS = (performance.now() - t0) / 1000;
  const diagnostics: Diagnostics = {
    wallS, decodeS, featuresS, detectorS: (performance.now() - ds0) / 1000, samples: t.length, sessionS,
    xRealtime: sessionS / wallS, frameFormats: [...formats], exactPath: exact, fallbackPath: fallback,
    detectorBackend: detector?.backend ?? 'off', detectorFallbackReason: detector?.fallbackReason ?? null, detectorCalls: detTimes.length,
    detectorMsPerCall: detTimes.length ? detTimes.reduce((a, b) => a + b, 0) / detTimes.length : 0,
    perspective: persp, userAgent: navigator.userAgent,
  };
  const result: AnalysisResult = {
    files, t: Float64Array.from(t), fileIndex: Int32Array.from(fileIndex), localT: Float64Array.from(localT), scores, light,
    prob, detector: [...detScores.entries()], perspective: persp, diagnostics,
  };
  post({ type: 'result', result });
  for (const o of opened) o.input.dispose?.();
}
