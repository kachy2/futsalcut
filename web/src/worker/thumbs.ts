/** Small JPEG previews of the decoded key frames (the timeline's hover images), handed over in batches. */
import { drawRotated } from './luma';

export const THUMB_W = 192;
export const THUMB_H = 108;
const THUMB_BATCH = 40;

export class Thumbnailer {
  private canvas = new OffscreenCanvas(THUMB_W, THUMB_H);
  private ctx = this.canvas.getContext('2d')!;
  private blobs: Blob[] = [];
  private next = 0;

  /** `deliver(start, blobs)`: `blobs` are the thumbnails of samples start, start + 1, ... */
  constructor(private deliver: (start: number, blobs: Blob[]) => void) {}

  async add(frame: VideoFrame, rotation: number): Promise<void> {
    drawRotated(this.ctx, frame, rotation, THUMB_W, THUMB_H);
    this.blobs.push(await this.canvas.convertToBlob({ type: 'image/jpeg', quality: 0.7 }));
    if (this.blobs.length >= THUMB_BATCH) this.flush();
  }

  /** hand over what is waiting (the last, shorter batch at the end of the stream) */
  flush(): void {
    if (!this.blobs.length) return;
    this.deliver(this.next, this.blobs);
    this.next += this.blobs.length;
    this.blobs = [];
  }
}

/** One thumbnail drawn from a frame, for frames whose thumbnail cannot be made from copied planes (10-bit, RGB, GPU-only). */
export class FrameThumbs {
  private canvas = new OffscreenCanvas(THUMB_W, THUMB_H);
  private ctx = this.canvas.getContext('2d')!;

  /** Draws now, so the frame may be closed right after; the JPEG is encoded in the background. */
  make(frame: VideoFrame, rotation: number): Promise<Blob> {
    drawRotated(this.ctx, frame, rotation, THUMB_W, THUMB_H);
    return this.canvas.convertToBlob({ type: 'image/jpeg', quality: 0.7 });
  }
}

/** Thumbnails that arrive with their sample index, in order, handed over in batches of consecutive samples. */
export class ThumbBatch {
  private start = 0;
  private blobs: Blob[] = [];

  constructor(private deliver: (start: number, blobs: Blob[]) => void) {}

  add(index: number, blob: Blob): void {
    if (this.blobs.length && index !== this.start + this.blobs.length) this.flush();
    if (!this.blobs.length) this.start = index;
    this.blobs.push(blob);
    if (this.blobs.length >= THUMB_BATCH) this.flush();
  }

  flush(): void {
    if (!this.blobs.length) return;
    this.deliver(this.start, this.blobs);
    this.blobs = [];
  }
}
