/**
 * The luma helper as the analysis worker uses it: it hands over a frame's copied planes and gets back the exact 640-px luma
 * and, for 8-bit YUV, the timeline thumbnail made from the same planes. The copied buffer comes back with the answer so that
 * the next frame can be copied into it. Answers may arrive in any order; the caller keeps the sample order.
 */
import type { RawPlanes } from './luma';

export type LumaRequest =
  | { type: 'ping' }
  | { type: 'shrink'; id: number; raw: RawPlanes; rotation: number; thumb: boolean };

export interface LumaDone {
  type: 'done';
  id: number;
  luma: Uint8Array;
  w: number;
  h: number;
  format: string;
  buf: ArrayBuffer;
  thumb: Blob | null;
  shrinkS: number;
  thumbS: number;
}

export type LumaReply = { type: 'pong' } | LumaDone | { type: 'error'; message: string };

export class LumaWorker {
  private waiting = new Map<number, { resolve(d: LumaDone): void; reject(e: Error): void }>();
  private failure: Error | null = null;

  constructor(private worker: Worker) {
    worker.onmessage = (ev: MessageEvent<LumaReply>) => {
      const m = ev.data;
      if (m.type === 'done') {
        const w = this.waiting.get(m.id);
        this.waiting.delete(m.id);
        w?.resolve(m);
      } else if (m.type === 'error') {
        this.fail(new Error(m.message));
      }
    };
    worker.onerror = (ev) => this.fail(new Error((ev as ErrorEvent).message || 'luma helper failed'));
  }

  private fail(e: Error): void {
    this.failure ??= e;
    for (const w of this.waiting.values()) w.reject(this.failure);
    this.waiting.clear();
  }

  /** The buffer of `raw` is handed over (transferred) and comes back in the answer. */
  shrink(id: number, raw: RawPlanes, rotation: number, thumb: boolean): Promise<LumaDone> {
    if (this.failure) return Promise.reject(this.failure);
    return new Promise<LumaDone>((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      this.worker.postMessage({ type: 'shrink', id, raw, rotation, thumb } satisfies LumaRequest, [raw.buf]);
    });
  }

  /** Stops the helper; answers still on their way are dropped (nobody waits for them any more). */
  close(): void {
    this.worker.terminate();
    this.waiting.clear();
  }
}

/** A running luma helper, or null if it cannot be started (it has to answer a ping first). */
export function openLumaWorker(): Promise<LumaWorker | null> {
  return new Promise((resolve) => {
    let worker: Worker;
    try {
      worker = new Worker(new URL('./luma-worker.ts', import.meta.url), { type: 'module' });
    } catch (e) {
      console.warn('luma helper unavailable, shrinking in the analysis worker', e);
      resolve(null);
      return;
    }
    const giveUp = (why: unknown) => {
      console.warn('luma helper unavailable, shrinking in the analysis worker', why);
      clearTimeout(timer);
      worker.terminate();
      resolve(null);
    };
    const timer = setTimeout(() => giveUp('no answer'), 10_000);
    worker.onerror = (ev) => giveUp((ev as ErrorEvent).message);
    worker.onmessage = () => {
      clearTimeout(timer);
      resolve(new LumaWorker(worker));
    };
    worker.postMessage({ type: 'ping' } satisfies LumaRequest);
  });
}
