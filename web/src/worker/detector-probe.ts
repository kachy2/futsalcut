/// <reference lib="webworker" />
// Detector self-test for the diagnostics panel: load the model, run one inference, report backend and timings.
import { createDetector } from './detector';

self.onmessage = async (ev: MessageEvent) => {
  const t0 = performance.now();
  try {
    const det = await createDetector(ev.data.modelUrl as string, ev.data.prefer ?? 'wasm', ev.data.threads,
      { webgpuWaitMs: ev.data.webgpuWaitS ? ev.data.webgpuWaitS * 1000 : undefined, anyGpu: ev.data.anyGpu });
    const t1 = performance.now();
    await det.detect(new Uint8ClampedArray(640 * 640 * 4)); // warm-up
    const t2 = performance.now();
    const runs = 3;
    for (let i = 0; i < runs; i++) await det.detect(new Uint8ClampedArray(640 * 640 * 4));
    const t3 = performance.now();
    postMessage({ ok: true, asked: ev.data.prefer ?? 'wasm', backend: det.backend, fallbackReason: det.fallbackReason, gpuCheck: det.gpuCheck, loadMs: Math.round(t1 - t0), firstMs: Math.round(t2 - t1),
      inferMs: Math.round((t3 - t2) / runs), threads: det.threads, webgpu: 'gpu' in navigator, isolated: self.crossOriginIsolated });
    await det.release();
  } catch (e) {
    postMessage({ ok: false, error: String((e as Error)?.stack ?? e).slice(0, 2000) });
  }
};
