/**
 * D-FINE-N person detector (Apache-2.0) via onnxruntime-web: WASM by default, WebGPU only when asked for (experimental).
 * WebGPU is used only if it creates a session, runs a fixed test frame within the time limit and agrees with WASM on that frame;
 * a backend that fails there, or later during inference, is replaced by WASM and the reason is kept.
 * Input: a frame drawn (stretched, display orientation) into a 640x640 RGB canvas — the same
 * preprocessing as the HF RTDetrImageProcessor (resize, /255, no normalization).
 */
import * as ort from 'onnxruntime-web/webgpu';
import { reshapeInitializers } from '../core/onnx-patch';

export interface Detector {
  readonly backend: string;
  /** why WebGPU was given up, if it was */
  readonly fallbackReason: string | null;
  /** WebGPU only: largest difference to the WASM results on the test frame, for the diagnostics */
  readonly gpuCheck: string | null;
  /** WASM threads in use (onnxruntime-web drops to 1 when the page is not cross-origin isolated) */
  readonly threads: number;
  detect(rgba: Uint8ClampedArray): Promise<number[][]>;
  release(): Promise<void>;
}

export interface DetectorOptions {
  /** how long WebGPU may take to create its session and run the test frame (a software adapter needs minutes) */
  webgpuWaitMs?: number;
}

const PERSON = 0;
const MIN_SCORE = 0.15;
/** onnxruntime-web 1.29's WebGPU backend cannot run a MatMul whose weight is a 1-D tensor ("SizeToDimension ... Invalid dimension of
 * 4294967295"; it computes rank - 2 on an unsigned number). D-FINE's three `integral` layers project the 33 distribution bins with
 * such a weight; one more axis of length 1 gives the same numbers (WASM results differ by < 2e-5). Only the WebGPU session gets this. */
const WEBGPU_FIX: Record<string, number[]> = { '/m/model/decoder/Concat_output_0': [33, 1] };
/** WebGPU must match WASM on the test frame this closely (a software WebGPU adapter showed 8e-5 on logits, 1e-5 on boxes) */
const LOGIT_TOL = 0.02;
const BOX_TOL = 0.005;

interface Raw { logits: Float32Array; boxes: Float32Array; nq: number; nc: number }

/** A fixed noise frame: the same input for every backend and every run. */
function testFrame(): Uint8ClampedArray {
  const px = new Uint8ClampedArray(640 * 640 * 4);
  let s = 12345;
  for (let i = 0; i < px.length; i++) {
    if (i % 4 === 3) { px[i] = 255; continue; }
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    px[i] = s >>> 24;
  }
  return px;
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  let timer = 0;
  const limit = new Promise<never>((_, reject) => { timer = self.setTimeout(() => reject(new Error(`timed out after ${Math.round(ms / 1000)} s`)), ms); });
  return Promise.race([p, limit]).finally(() => self.clearTimeout(timer));
}

const maxDiff = (a: Float32Array, b: Float32Array) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

export async function createDetector(modelUrl: string, prefer: 'webgpu' | 'wasm' = 'wasm', threads?: number,
  opts: DetectorOptions = {}): Promise<Detector> {
  // the runtime's .mjs and .wasm are plain files in <app>/ort/ (prepare-assets.mjs copies them); the model is in <app>/models/
  ort.env.wasm.wasmPaths = new URL('../ort/', modelUrl).href;
  const cores = navigator.hardwareConcurrency || 1;
  ort.env.wasm.numThreads = threads ?? (self.crossOriginIsolated ? Math.min(4, Math.max(1, cores >> 1)) : 1);
  const model = new Uint8Array(await (await fetch(modelUrl)).arrayBuffer());
  const input = new Float32Array(3 * 640 * 640);
  const makeSession = (ep: 'webgpu' | 'wasm') =>
    ort.InferenceSession.create(ep === 'webgpu' ? (reshapeInitializers(model, WEBGPU_FIX) ?? model) : model,
      { executionProviders: [ep], graphOptimizationLevel: 'all' });
  const run = async (session: ort.InferenceSession, rgba: Uint8ClampedArray, copy = false): Promise<Raw> => {
    const plane = 640 * 640;
    for (let i = 0; i < plane; i++) {
      input[i] = rgba[i * 4] / 255;
      input[plane + i] = rgba[i * 4 + 1] / 255;
      input[2 * plane + i] = rgba[i * 4 + 2] / 255;
    }
    const out = await session.run({ pixel_values: new ort.Tensor('float32', input, [1, 3, 640, 640]) });
    const logits = out.logits.data as Float32Array; // [1, 300, 80]
    const boxes = out.pred_boxes.data as Float32Array; // [1, 300, 4]
    return { logits: copy ? logits.slice() : logits, boxes: copy ? boxes.slice() : boxes, nq: out.logits.dims[1], nc: out.logits.dims[2] };
  };
  const toBoxes = ({ logits, boxes, nq, nc }: Raw): number[][] => {
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
  let current: ort.InferenceSession | null = null;
  let standby: ort.InferenceSession | null = null; // the WASM session that proved the WebGPU results, kept for a fallback
  let fallbackReason: string | null = null;
  let gpuCheck: string | null = null;
  const describe = (e: unknown) => String((e as Error)?.message ?? e).slice(0, 400);
  if (prefer === 'webgpu' && 'gpu' in navigator) {
    const gpu = { session: null as ort.InferenceSession | null };
    const frame = testFrame();
    // some adapters only fail (or hang) when a kernel runs, so the session is not trusted before the test frame has been run
    const warmUp = (async () => {
      gpu.session = await makeSession('webgpu');
      return run(gpu.session, frame, true);
    })();
    try {
      const onGpu = await withTimeout(warmUp, opts.webgpuWaitMs ?? 60_000);
      const wasm = await makeSession('wasm');
      const onCpu = await run(wasm, frame, true);
      const dl = maxDiff(onGpu.logits, onCpu.logits);
      const db = maxDiff(onGpu.boxes, onCpu.boxes);
      gpuCheck = `logits ${dl.toExponential(1)}, boxes ${db.toExponential(1)}`;
      if (!(dl <= LOGIT_TOL && db <= BOX_TOL)) { // also catches NaN
        await wasm.release().catch(() => undefined);
        throw new Error(`results differ from WASM (${gpuCheck})`);
      }
      current = gpu.session;
      standby = wasm;
      ep = 'webgpu';
    } catch (e) {
      fallbackReason = `webgpu: ${describe(e)}`;
      void warmUp.then(() => undefined, () => undefined).then(() => gpu.session?.release()).catch(() => undefined); // a session still being created
    }
  }
  if (!current) current = await makeSession('wasm');

  return {
    get backend() { return fallbackReason ? 'wasm (webgpu failed)' : ep; },
    get fallbackReason() { return fallbackReason; },
    get gpuCheck() { return gpuCheck; },
    get threads() { return ort.env.wasm.numThreads ?? 1; },
    async detect(rgba: Uint8ClampedArray): Promise<number[][]> {
      try {
        return toBoxes(await run(current!, rgba));
      } catch (e) {
        if (ep !== 'webgpu') throw e;
        // the GPU backend broke while running: continue with WASM from this very frame
        fallbackReason = `webgpu: ${describe(e)}`;
        await current!.release().catch(() => undefined);
        ep = 'wasm';
        current = standby ?? (await makeSession('wasm'));
        standby = null;
        return toBoxes(await run(current, rgba));
      }
    },
    async release() {
      await current?.release().catch(() => undefined);
      await standby?.release().catch(() => undefined);
    },
  };
}
