/// <reference lib="webworker" />
/**
 * Helper of the analysis worker (see luma-client.ts): shrinks a frame's copied planes to the exact 640-px luma sample and makes
 * the timeline thumbnail from the same planes, while the analysis worker decodes and copies the next frame.
 * Requests are handled one at a time.
 */
import { planesToRgba, shrinkPlanes } from './luma';
import type { LumaReply, LumaRequest } from './luma-client';
import { THUMB_H, THUMB_W } from './thumbs';

const canvas = new OffscreenCanvas(THUMB_W, THUMB_H);
const ctx = canvas.getContext('2d')!;
const send = (m: LumaReply, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);
let queue: Promise<void> = Promise.resolve();

async function handle(m: LumaRequest): Promise<void> {
  if (m.type === 'ping') {
    send({ type: 'pong' });
    return;
  }
  try {
    const s0 = performance.now();
    const luma = shrinkPlanes(m.raw, m.rotation);
    const s1 = performance.now();
    let thumb: Blob | null = null;
    if (m.thumb) {
      ctx.putImageData(new ImageData(planesToRgba(m.raw, m.rotation, THUMB_W, THUMB_H) as Uint8ClampedArray<ArrayBuffer>, THUMB_W, THUMB_H), 0, 0);
      thumb = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.7 });
    }
    const s2 = performance.now();
    send({ type: 'done', id: m.id, luma: luma.data, w: luma.w, h: luma.h, format: luma.format, buf: m.raw.buf, thumb,
      shrinkS: (s1 - s0) / 1000, thumbS: (s2 - s1) / 1000 }, [luma.data.buffer as ArrayBuffer, m.raw.buf]);
  } catch (e) {
    send({ type: 'error', message: String((e as Error)?.message ?? e) });
  }
}

self.onmessage = (ev: MessageEvent<LumaRequest>) => {
  queue = queue.then(() => handle(ev.data)); // handle never rejects
};
