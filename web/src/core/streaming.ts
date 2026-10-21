/**
 * Streaming pass 1: feed luma samples one at a time; results come out in order once the background
 * window of their block is complete (at most ~65 samples later) or when the stream ends.
 * Equivalent to python futsalcut.features.run_pass1 over the concatenated sample sequence,
 * while holding only the samples still needed (~130 x 230 KB at 640x360).
 */
import { BG_EVERY, BG_HALF, BG_STEP, blockWindow, lowerMedian, pass1Sample, type Pass1 } from './features';

export class StreamingPass1 {
  private frames = new Map<number, Uint8Array>();
  private bgCache = new Map<number, Uint8Array>();
  private received = 0;
  private processed = 0;
  private ended = false;
  readonly results: Pass1[] = [];

  constructor(readonly W: number, readonly H: number, private zone: Uint8Array,
    private onResult?: (index: number, r: Pass1) => void) {}

  get count(): number { return this.received; }

  push(luma: Uint8Array): void {
    this.frames.set(this.received++, luma);
    this.drain();
  }

  end(): Pass1[] {
    this.ended = true;
    this.drain();
    this.frames.clear();
    this.bgCache.clear();
    return this.results;
  }

  /** Highest sample index the background of `block` needs, assuming the stream continues. */
  private lastNeeded(block: number): number {
    return block * BG_EVERY + Math.floor(BG_EVERY / 2) + BG_HALF;
  }

  private background(block: number): Uint8Array {
    let bg = this.bgCache.get(block);
    if (!bg) {
      const n = this.ended ? this.received : Number.MAX_SAFE_INTEGER;
      const idx = blockWindow(n, block);
      bg = lowerMedian(idx.map((i) => this.frames.get(i)!), this.W * this.H);
      this.bgCache.set(block, bg);
    }
    return bg;
  }

  private drain(): void {
    while (this.processed < this.received) {
      const i = this.processed;
      const block = Math.floor(i / BG_EVERY);
      if (!this.ended && this.received - 1 < this.lastNeeded(block)) return; // window not complete yet
      const r = pass1Sample(this.frames.get(i)!, this.background(block), i ? this.frames.get(i - 1)! : null,
        this.zone, this.W, this.H);
      this.results.push(r);
      this.onResult?.(i, r);
      this.processed++;
      this.evict();
    }
  }

  /** Drop frames no future background window or `prev` reference can use. At the end of the stream the
   * last block's window centre is clamped to n-1 (>= the block start), so keep from blockStart - BG_HALF. */
  private evict(): void {
    const nextBlock = Math.floor(this.processed / BG_EVERY);
    const blockStart = nextBlock * BG_EVERY;
    const keepFrom = Math.min(this.processed - 1, blockStart - BG_HALF);
    for (const k of this.frames.keys()) {
      if (k < keepFrom) this.frames.delete(k);
    }
    for (const b of this.bgCache.keys()) if (b < nextBlock) this.bgCache.delete(b);
  }
}

export { BG_STEP };
