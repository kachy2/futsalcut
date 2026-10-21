// WebCodecs benchmark for the user's real device (plan P0.5(e)).
// Measures keyframe-only decoding (exact Y-plane path and fast canvas path) and full decoding speed.
import { ALL_FORMATS, BlobSource, EncodedPacketSink, Input, type InputVideoTrack } from 'mediabunny';
import { boxDownscale, targetSize } from '../src/core/downscale';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const fileEl = $<HTMLInputElement>('file');
const runEl = $<HTMLButtonElement>('run');
const copyEl = $<HTMLButtonElement>('copy');
const secondsEl = $<HTMLInputElement>('seconds');
const logEl = $<HTMLPreElement>('log');

const result: Record<string, unknown> = {};
const log = (s: string) => { logEl.textContent += s + '\n'; };

fileEl.addEventListener('change', () => { runEl.disabled = !fileEl.files?.length; });
copyEl.addEventListener('click', async () => {
  await navigator.clipboard.writeText(JSON.stringify(result, null, 1));
  copyEl.textContent = '복사됨';
});
runEl.addEventListener('click', () => {
  run().catch((e) => {
    log('오류: ' + (e?.stack ?? e));
    result.error = String(e);
    copyEl.disabled = false;
    runEl.disabled = false;
  });
});

async function run() {
  runEl.disabled = true;
  copyEl.disabled = true;
  copyEl.textContent = '결과 복사';
  logEl.textContent = '';
  const file = fileEl.files![0];
  result.env = {
    ua: navigator.userAgent,
    secureContext: isSecureContext,
    cores: navigator.hardwareConcurrency,
    webcodecs: 'VideoDecoder' in self,
    webgpu: 'gpu' in navigator,
    fileSystemAccess: 'showSaveFilePicker' in self,
  };
  result.file = { name: file.name, sizeGiB: +(file.size / 2 ** 30).toFixed(3) };
  log(`파일: ${file.name} (${(file.size / 2 ** 30).toFixed(2)} GiB)`);
  if (!('VideoDecoder' in self)) throw new Error('이 브라우저는 WebCodecs를 지원하지 않습니다');

  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  const track = await input.getPrimaryVideoTrack();
  if (!track) throw new Error('영상 트랙이 없습니다');
  const config = await track.getDecoderConfig();
  if (!config) throw new Error('디코더 설정을 만들 수 없습니다(지원하지 않는 코덱)');
  const duration = await track.computeDuration();
  result.video = {
    codec: track.codec, codecString: config.codec, coded: [track.codedWidth, track.codedHeight],
    rotation: track.rotation, durationS: +duration.toFixed(2),
  };
  log(`코덱 ${config.codec}, ${track.codedWidth}x${track.codedHeight}, ${duration.toFixed(1)}초`);

  const support: Record<string, boolean> = {};
  for (const hw of ['prefer-hardware', 'prefer-software', 'no-preference'] as const) {
    try {
      support[hw] = !!(await VideoDecoder.isConfigSupported({ ...config, hardwareAcceleration: hw })).supported;
    } catch {
      support[hw] = false;
    }
  }
  result.support = support;
  log('디코더 지원: ' + JSON.stringify(support));

  const limit = Number(secondsEl.value) > 0 ? Number(secondsEl.value) : Infinity;
  log('\n[1] 키프레임만 디코딩 — 정확 경로(Y 평면 복사 + 박스 축소)');
  result.keyframesExact = await keyframeBench(track, config, limit, 'exact');
  log('\n[2] 키프레임만 디코딩 — 빠른 경로(캔버스 축소)');
  result.keyframesFast = await keyframeBench(track, config, Math.min(limit, 120), 'fast');
  log('\n[3] 전체 프레임 디코딩 20초');
  result.fullDecode = await fullDecodeBench(track, config, 20);
  log('\n완료. "결과 복사"를 눌러 결과를 전달해 주세요.');
  copyEl.disabled = false;
  runEl.disabled = false;
}

type Path = 'exact' | 'fast';

async function keyframeBench(track: InputVideoTrack, config: VideoDecoderConfig, limitS: number, path: Path) {
  const sink = new EncodedPacketSink(track);
  const [w, h] = targetSize(track.codedWidth, track.codedHeight);
  const small = new Uint8Array(w * h);
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  let buf = new Uint8Array(0);
  const frames: VideoFrame[] = [];
  let failure: unknown = null;
  let wake: (() => void) | null = null;
  const decoder = new VideoDecoder({
    output: (f) => { frames.push(f); wake?.(); },
    error: (e) => { failure = e; wake?.(); },
  });
  decoder.configure({ ...config, optimizeForLatency: true });

  let n = 0, procMs = 0, firstTs = NaN, lastTs = NaN;
  const formats = new Set<string>();
  const process = async (f: VideoFrame) => {
    const t = performance.now();
    try {
      formats.add(String(f.format));
      if (path === 'exact') {
        const size = f.allocationSize();
        if (buf.length < size) buf = new Uint8Array(size);
        const layout = await f.copyTo(buf);
        const y = buf.subarray(layout[0].offset);
        if (f.format === 'I420' || f.format === 'NV12' || f.format === 'I420A' || f.format === 'I422' || f.format === 'I444') {
          boxDownscale(y, f.visibleRect!.width, f.visibleRect!.height, layout[0].stride, w, h, small);
        }
      } else {
        ctx.drawImage(f, 0, 0, w, h);
        ctx.getImageData(0, 0, w, h);
      }
      if (Number.isNaN(firstTs)) firstTs = f.timestamp / 1e6;
      lastTs = f.timestamp / 1e6;
      n++;
    } finally {
      f.close();
      procMs += performance.now() - t;
    }
  };

  const t0 = performance.now();
  let pkt = await sink.getFirstPacket();
  while (pkt && pkt.timestamp <= limitS) {
    if (failure) throw failure;
    if (pkt.type === 'key') decoder.decode(pkt.toEncodedVideoChunk());
    while (decoder.decodeQueueSize > 4 || frames.length > 2) {
      if (frames.length) await process(frames.shift()!);
      else await new Promise<void>((r) => { wake = r; decoder.addEventListener('dequeue', () => r(), { once: true }); });
      if (failure) throw failure;
    }
    while (frames.length) await process(frames.shift()!);
    pkt = await sink.getNextKeyPacket(pkt);
    if (n && n % 100 === 0) log(`  ${n}장... ${lastTs.toFixed(0)}초`);
  }
  await decoder.flush();
  while (frames.length) await process(frames.shift()!);
  decoder.close();
  const wallS = (performance.now() - t0) / 1000;
  const spanS = n ? lastTs - firstTs + 1 : 0;
  const r = {
    keyframes: n, videoSpanS: +spanS.toFixed(1), wallS: +wallS.toFixed(2),
    msPerKeyframe: n ? +((wallS * 1000) / n).toFixed(1) : null,
    processMsPerKeyframe: n ? +(procMs / n).toFixed(1) : null,
    xRealtime: wallS ? +(spanS / wallS).toFixed(1) : null,
    frameFormats: [...formats],
  };
  log('  ' + JSON.stringify(r));
  return r;
}

async function fullDecodeBench(track: InputVideoTrack, config: VideoDecoderConfig, seconds: number) {
  const sink = new EncodedPacketSink(track);
  let n = 0;
  let failure: unknown = null;
  const decoder = new VideoDecoder({
    output: (f) => { n++; f.close(); },
    error: (e) => { failure = e; },
  });
  decoder.configure(config);
  const t0 = performance.now();
  let pkt = await sink.getFirstPacket();
  while (pkt && pkt.timestamp < seconds) {
    if (failure) throw failure;
    decoder.decode(pkt.toEncodedVideoChunk());
    if (decoder.decodeQueueSize > 8) {
      await new Promise<void>((r) => decoder.addEventListener('dequeue', () => r(), { once: true }));
    }
    pkt = await sink.getNextPacket(pkt);
  }
  await decoder.flush();
  decoder.close();
  const wallS = (performance.now() - t0) / 1000;
  const r = { frames: n, wallS: +wallS.toFixed(2), fps: +(n / wallS).toFixed(1), xRealtime: +(seconds / wallS).toFixed(2) };
  log('  ' + JSON.stringify(r));
  return r;
}
