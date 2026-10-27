/**
 * D-FINE-N person detector (Apache-2.0) via onnxruntime-web: WASM by default, WebGPU only when asked for (experimental).
 * A WebGPU backend that fails at creation, in the warm-up run or later during inference is replaced by WASM and the reason is kept.
 * Input: a frame drawn (stretched, display orientation) into a 640x640 RGB canvas — the same
 * preprocessing as the HF RTDetrImageProcessor (resize, /255, no normalization).
 */
import * as ort from 'onnxruntime-web/webgpu';

export interface Detector {
  readonly backend: string;
  /** why WebGPU was given up, if it was */
  readonly fallbackReason: string | null;
  detect(rgba: Uint8ClampedArray): Promise<number[][]>;
  release(): Promise<void>;
}

const PERSON = 0;
const MIN_SCORE = 0.15;

export async function createDetector(modelUrl: string, prefer: 'webgpu' | 'wasm' = 'wasm'): Promise<Detector> {
  // the runtime's .wasm is bundled by Vite (referenced from onnxruntime-web/webgpu)
  ort.env.wasm.numThreads = Math.min(4, (self.crossOriginIsolated ? navigator.hardwareConcurrency : 1) || 1);
  const model = new Uint8Array(await (await fetch(modelUrl)).arrayBuffer());
  const input = new Float32Array(3 * 640 * 640);
  const makeSession = (ep: 'webgpu' | 'wasm') =>
    ort.InferenceSession.create(model, { executionProviders: [ep], graphOptimizationLevel: 'all' });
  const infer = async (session: ort.InferenceSession, rgba: Uint8ClampedArray): Promise<number[][]> => {
    const plane = 640 * 640;
    for (let i = 0; i < plane; i++) {
      input[i] = rgba[i * 4] / 255;
      input[plane + i] = rgba[i * 4 + 1] / 255;
      input[2 * plane + i] = rgba[i * 4 + 2] / 255;
    }
    const out = await session.run({ pixel_values: new ort.Tensor('float32', input, [1, 3, 640, 640]) });
    const logits = out.logits.data as Float32Array; // [1, 300, 80]
    const boxes = out.pred_boxes.data as Float32Array; // [1, 300, 4]
    const nq = out.logits.dims[1];
    const nc = out.logits.dims[2];
    const res: number[][] = [];
    for (let q = 0; q < nq; q++) {
      const p = 1 / (1 + Math.exp(-logits[q * nc + PERSON]));
      if (p < MIN_SCORE) continue;
      const cx = boxes[q * 4];
      const cy = boxes[q * 4 + 1];
      const w = boxes[q * 4 + 2];
      const h = boxes[q * 4 + 3];
      res.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, p]);
    }
    return res;
  };

  let ep: 'webgpu' | 'wasm' = 'wasm';
  let session: ort.InferenceSession | null = null;
  let fallbackReason: string | null = null;
  const describe = (e: unknown) => String((e as Error)?.message ?? e).slice(0, 400);
  if (prefer === 'webgpu' && 'gpu' in navigator) {
    try {
      const warmUp = (async () => {
        const gpuSession = await makeSession('webgpu');
        await infer(gpuSession, new Uint8ClampedArray(640 * 640 * 4)); // some adapters only fail (or hang) when a kernel runs
        return gpuSession;
      })();
      let timer = 0;
      const timeout = new Promise<never>((_, reject) => { timer = self.setTimeout(() => reject(new Error('timed out after 60 s')), 60_000); });
      try {
        session = await Promise.race([warmUp, timeout]);
      } finally {
        self.clearTimeout(timer);
      }
      ep = 'webgpu';
    } catch (e) {
      fallbackReason = `webgpu: ${describe(e)}`;
    }
  }
  if (!session) session = await makeSession('wasm');
  let current = session;

  return {
    get backend() { return fallbackReason ? 'wasm (webgpu failed)' : ep; },
    get fallbackReason() { return fallbackReason; },
    async detect(rgba: Uint8ClampedArray): Promise<number[][]> {
      try {
        return await infer(current, rgba);
      } catch (e) {
        if (ep !== 'webgpu') throw e;
        // the GPU backend broke while running: continue with WASM from this very frame
        fallbackReason = `webgpu: ${describe(e)}`;
        await current.release().catch(() => undefined);
        ep = 'wasm';
        current = await makeSession('wasm');
        return infer(current, rgba);
      }
    },
    release: () => current.release(),
  };
}
