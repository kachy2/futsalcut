/**
 * The person detector as the analysis worker uses it. It lives in a worker of its own, so inference (graphics chip or CPU threads)
 * goes on while the analysis worker decodes the next frame; where that worker cannot be started the detector runs in the calling
 * thread instead (same results, no overlap). `submit` hands a 640x640 RGBA frame over (its buffer is transferred) and may be called
 * again before the answer is back; answers come in the order of the calls.
 */
import type { DetectorOptions } from './detector';

export interface Detection {
  /** normalized x1, y1, x2, y2, score */
  boxes: number[][];
  /** time the inference took (from taking the frame to having the boxes), not counting the wait behind earlier calls */
  ms: number;
}

export interface PipelinedDetector {
  /** true: runs in its own worker, so calls overlap with the caller's work */
  readonly remote: boolean;
  readonly backend: string;
  readonly fallbackReason: string | null;
  readonly gpuCheck: string | null;
  readonly threads: number;
  /** cross-origin isolated where the inference runs (WASM threads need that) */
  readonly isolated: boolean;
  submit(rgba: Uint8ClampedArray): Promise<Detection>;
  release(): Promise<void>;
}

export interface DetectorState { backend: string; fallbackReason: string | null; gpuCheck: string | null; threads: number; isolated: boolean }

export type DetectorRequest =
  | { id: number; type: 'init'; modelUrl: string; prefer: 'auto' | 'webgpu' | 'wasm'; threads?: number; opts: DetectorOptions }
  | { id: number; type: 'detect'; rgba: Uint8ClampedArray }
  | { id: number; type: 'release' };

export interface DetectorReply { id: number; error?: string; state?: DetectorState; boxes?: number[][]; ms?: number }

type WithoutId<T> = T extends unknown ? Omit<T, 'id'> : never;

/** The worker script did not load or crashed before it answered: a case for falling back to the calling thread. */
class WorkerStartError extends Error {}

export async function createPipelinedDetector(modelUrl: string, prefer: 'auto' | 'webgpu' | 'wasm', threads: number | undefined,
  opts: DetectorOptions, separateWorker = true): Promise<PipelinedDetector> {
  if (separateWorker) {
    try {
      return await inWorker(modelUrl, prefer, threads, opts);
    } catch (e) {
      if (!(e instanceof WorkerStartError)) throw e; // the detector itself failed: the thread would fail the same way
      console.warn('detector worker unavailable, running the detector in the analysis worker', e);
    }
  }
  return inThread(modelUrl, prefer, threads, opts);
}

async function inThread(modelUrl: string, prefer: 'auto' | 'webgpu' | 'wasm', threads: number | undefined,
  opts: DetectorOptions): Promise<PipelinedDetector> {
  const { createDetector } = await import('./detector'); // only here: the runtime is not part of this worker otherwise
  const d = await createDetector(modelUrl, prefer, threads, opts);
  return {
    remote: false,
    get backend() { return d.backend; },
    get fallbackReason() { return d.fallbackReason; },
    get gpuCheck() { return d.gpuCheck; },
    get threads() { return d.threads; },
    isolated: self.crossOriginIsolated,
    async submit(rgba) {
      const t0 = performance.now();
      const boxes = await d.detect(rgba);
      return { boxes, ms: performance.now() - t0 };
    },
    release: () => d.release(),
  };
}

function inWorker(modelUrl: string, prefer: 'auto' | 'webgpu' | 'wasm', threads: number | undefined,
  opts: DetectorOptions): Promise<PipelinedDetector> {
  let worker: Worker;
  try {
    worker = new Worker(new URL('./detector-worker.ts', import.meta.url), { type: 'module' });
  } catch (e) {
    return Promise.reject(new WorkerStartError(String(e)));
  }
  let heard = false; // the worker answered at least once: its script is running
  let dead: Error | null = null;
  let state: DetectorState | null = null;
  let seq = 0;
  const waiting = new Map<number, { resolve(m: DetectorReply): void; reject(e: Error): void }>();
  const failAll = (e: Error) => {
    dead ??= e;
    for (const w of waiting.values()) w.reject(e);
    waiting.clear();
  };
  worker.onmessage = (ev: MessageEvent<DetectorReply>) => {
    heard = true;
    const m = ev.data;
    const w = waiting.get(m.id);
    if (!w) return;
    waiting.delete(m.id);
    if (m.state) state = m.state;
    if (m.error !== undefined) w.reject(new Error(m.error));
    else w.resolve(m);
  };
  worker.onerror = (ev) => {
    const why = (ev as ErrorEvent).message || 'detector worker failed';
    failAll(heard ? new Error(why) : new WorkerStartError(why));
  };
  const call = (msg: WithoutId<DetectorRequest>, transfer: Transferable[] = []) => new Promise<DetectorReply>((resolve, reject) => {
    if (dead) { reject(dead); return; } // an answer would never come
    const id = ++seq;
    waiting.set(id, { resolve, reject });
    try {
      worker.postMessage({ ...msg, id }, transfer);
    } catch (e) {
      waiting.delete(id);
      reject(heard ? e : new WorkerStartError(String(e)));
    }
  });

  return call({ type: 'init', modelUrl, prefer, threads, opts }).then(() => {
    const now = () => state!;
    const detector: PipelinedDetector = {
      remote: true,
      get backend() { return now().backend; },
      get fallbackReason() { return now().fallbackReason; },
      get gpuCheck() { return now().gpuCheck; },
      get threads() { return now().threads; },
      get isolated() { return now().isolated; },
      async submit(rgba) {
        const m = await call({ type: 'detect', rgba }, [rgba.buffer]);
        return { boxes: m.boxes!, ms: m.ms! };
      },
      async release() {
        let timer: ReturnType<typeof setTimeout> | undefined;
        try {
          const patience = new Promise<void>((r) => { timer = setTimeout(r, 5000); }); // a stuck worker must not block the next analysis
          await Promise.race([call({ type: 'release' }), patience]);
        } catch { /* the worker is gone already */ } finally {
          clearTimeout(timer);
          worker.terminate();
        }
      },
    };
    return detector;
  }, (e) => {
    worker.terminate();
    throw e;
  });
}
