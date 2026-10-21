/**
 * D-FINE-N person detector (Apache-2.0) via onnxruntime-web: WebGPU when available, WASM otherwise.
 * Input: a frame drawn (stretched, display orientation) into a 640x640 RGB canvas — the same
 * preprocessing as the HF RTDetrImageProcessor (resize, /255, no normalization).
 */
import * as ort from 'onnxruntime-web/webgpu';

export interface Detector { backend: string; detect(rgba: Uint8ClampedArray): Promise<number[][]>; release(): Promise<void> }

const PERSON = 0;
const MIN_SCORE = 0.15;

export async function createDetector(modelUrl: string, prefer: 'webgpu' | 'wasm' = 'webgpu'): Promise<Detector> {
  // the runtime's .wasm is bundled by Vite (referenced from onnxruntime-web/webgpu)
  ort.env.wasm.numThreads = Math.min(4, (self.crossOriginIsolated ? navigator.hardwareConcurrency : 1) || 1);
  const model = new Uint8Array(await (await fetch(modelUrl)).arrayBuffer());
  const tryOrder = prefer === 'webgpu' && 'gpu' in navigator ? ['webgpu', 'wasm'] : ['wasm'];
  let lastErr: unknown = null;
  for (const ep of tryOrder) {
    try {
      const session = await ort.InferenceSession.create(model, { executionProviders: [ep], graphOptimizationLevel: 'all' });
      const input = new Float32Array(3 * 640 * 640);
      return {
        backend: ep,
        async detect(rgba: Uint8ClampedArray): Promise<number[][]> {
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
        },
        release: () => session.release(),
      };
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr ?? new Error('no execution provider available');
}
