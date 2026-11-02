/**
 * What happens to every decoded sample besides storing it: streaming pass 1 over its luma and a thumbnail from its frame.
 * `openSide` puts that into a helper worker, so the analysis worker can go on decoding; if the helper cannot be started the work
 * is done in the calling thread (same results, no overlap).
 */
import type { Pass1 } from '../core/features';
import { polygonMask, type Point } from '../core/roi';
import { StreamingPass1 } from '../core/streaming';
import { Thumbnailer } from './thumbs';

export interface SideResult {
  pass1: Pass1[];
  /** seconds the pass-1 and the thumbnail work took, in whichever thread did it */
  pass1S: number;
  thumbS: number;
}

export interface SideWork {
  /** true: a helper worker does the work beside the caller */
  readonly remote: boolean;
  /** with the first sample: its size, and the zone pass 1 looks at */
  start(W: number, H: number, zone: Point[]): void;
  /** One sample: `luma` goes to pass 1, a thumbnail is drawn from `frame` if there is one (the caller closes its own reference
   * afterwards; without a frame the thumbnail is made elsewhere). Returns once the helper has taken it over; waits while the helper is behind. */
  push(luma: Uint8Array, frame: VideoFrame | null, rotation: number): Promise<void>;
  /** the end of the stream: the pass-1 results; every thumbnail has been delivered by then */
  end(): Promise<SideResult>;
  close(): void;
}

export type SideRequest =
  | { type: 'ping' }
  | { type: 'start'; W: number; H: number; zone: Point[] }
  | { type: 'sample'; luma: Uint8Array; frame?: VideoFrame; rotation: number }
  | { type: 'end' };

export type SideReply =
  | { type: 'pong' }
  | { type: 'ack' }
  | { type: 'thumbs'; start: number; blobs: Blob[] }
  | ({ type: 'result' } & SideResult)
  | { type: 'error'; message: string };

type Deliver = (start: number, blobs: Blob[]) => void;
const lap = (since: number) => (performance.now() - since) / 1000;

class InlineSide implements SideWork {
  readonly remote = false;
  private sp: StreamingPass1 | null = null;
  private thumbs: Thumbnailer;
  private pass1S = 0;
  private thumbS = 0;

  constructor(deliver: Deliver) {
    this.thumbs = new Thumbnailer(deliver);
  }

  start(W: number, H: number, zone: Point[]): void {
    this.sp = new StreamingPass1(W, H, polygonMask(zone, W, H));
  }

  async push(luma: Uint8Array, frame: VideoFrame | null, rotation: number): Promise<void> {
    let s = performance.now();
    this.sp!.push(luma);
    this.pass1S += lap(s);
    if (!frame) return;
    s = performance.now();
    await this.thumbs.add(frame, rotation);
    this.thumbS += lap(s);
  }

  async end(): Promise<SideResult> {
    const s = performance.now();
    const pass1 = this.sp!.end();
    this.pass1S += lap(s);
    this.thumbs.flush();
    return { pass1, pass1S: this.pass1S, thumbS: this.thumbS };
  }

  close(): void { /* nothing to stop */ }
}

/** Samples handed over and not yet taken care of: each one holds a copy of its luma and a reference to a decoder frame. */
const MAX_BEHIND = 3;

class RemoteSide implements SideWork {
  readonly remote = true;
  private behind = 0;
  private waiters: (() => void)[] = [];
  private failure: Error | null = null;
  private result: { resolve(r: SideResult): void; reject(e: Error): void } | null = null;

  constructor(private worker: Worker, private deliver: Deliver) {
    worker.onmessage = (ev: MessageEvent<SideReply>) => {
      const m = ev.data;
      if (m.type === 'ack') { this.behind--; this.wake(); }
      else if (m.type === 'thumbs') this.deliver(m.start, m.blobs);
      else if (m.type === 'result') this.result?.resolve(m);
      else if (m.type === 'error') this.fail(new Error(m.message));
    };
    worker.onerror = (ev) => this.fail(new Error((ev as ErrorEvent).message || 'helper worker failed'));
  }

  private fail(e: Error): void {
    this.failure ??= e;
    this.result?.reject(this.failure);
    this.wake();
  }

  private wake(): void {
    for (const w of this.waiters.splice(0)) w();
  }

  start(W: number, H: number, zone: Point[]): void {
    this.worker.postMessage({ type: 'start', W, H, zone } satisfies SideRequest);
  }

  async push(luma: Uint8Array, frame: VideoFrame | null, rotation: number): Promise<void> {
    while (this.behind >= MAX_BEHIND && !this.failure) await new Promise<void>((r) => this.waiters.push(r));
    if (this.failure) throw this.failure;
    this.behind++;
    try {
      // luma is copied, a frame shares the decoder's picture
      this.worker.postMessage((frame ? { type: 'sample', luma, frame, rotation } : { type: 'sample', luma, rotation }) satisfies SideRequest);
    } catch (e) {
      this.behind--;
      throw new Error(`cannot hand a frame to the helper worker: ${(e as Error)?.message ?? e}`);
    }
  }

  end(): Promise<SideResult> {
    return new Promise<SideResult>((resolve, reject) => {
      if (this.failure) { reject(this.failure); return; }
      this.result = { resolve, reject };
      this.worker.postMessage({ type: 'end' } satisfies SideRequest);
    });
  }

  close(): void {
    this.worker.terminate();
    this.wake();
  }
}

/** A running helper worker, or null if it cannot be started (it has to answer a ping first). */
function startHelper(): Promise<Worker | null> {
  return new Promise((resolve) => {
    let worker: Worker;
    try {
      worker = new Worker(new URL('./side-worker.ts', import.meta.url), { type: 'module' });
    } catch (e) {
      console.warn('helper worker unavailable, working in the analysis worker', e);
      resolve(null);
      return;
    }
    const giveUp = (why: unknown) => {
      console.warn('helper worker unavailable, working in the analysis worker', why);
      clearTimeout(timer);
      worker.terminate();
      resolve(null);
    };
    const timer = setTimeout(() => giveUp('no answer'), 10_000);
    worker.onerror = (ev) => giveUp((ev as ErrorEvent).message);
    worker.onmessage = () => {
      clearTimeout(timer);
      resolve(worker);
    };
    worker.postMessage({ type: 'ping' } satisfies SideRequest);
  });
}

/** `deliver(start, blobs)` receives the thumbnails in batches, in the order of the samples. */
export async function openSide(parallel: boolean, deliver: Deliver): Promise<SideWork> {
  const worker = parallel ? await startHelper() : null;
  return worker ? new RemoteSide(worker, deliver) : new InlineSide(deliver);
}
