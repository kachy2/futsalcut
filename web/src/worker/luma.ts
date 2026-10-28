/**
 * VideoFrame -> 640-px luma sample. Exact path (parity contract, plan §3.1): copy the decoder's Y plane,
 * integer box downscale in coded orientation, then rotate to display orientation.
 */
import { boxDownscale, targetSize } from '../core/downscale';
import { rotateCw } from '../core/rotate';

const Y_8BIT = new Set(['I420', 'I420A', 'I422', 'I422A', 'I444', 'I444A', 'NV12']);
const Y_16BIT = new Set(['I420P10', 'I420AP10', 'I422P10', 'I422AP10', 'I444P10', 'I444AP10',
  'I420P12', 'I420AP12', 'I422P12', 'I422AP12', 'I444P12', 'I444AP12']);

let scratch = new Uint8Array(0);

/** Seconds spent so far in the two parts of the exact path: copying the frame out of the decoder, shrinking and rotating the luma (diagnostics). */
export const lumaClock = { copy: 0, shrink: 0 };

export interface LumaSample { data: Uint8Array; w: number; h: number; path: 'exact' | 'fallback'; format: string }

export async function frameToLuma(frame: VideoFrame, rotation: number, longSide = 640): Promise<LumaSample> {
  const vw = frame.visibleRect?.width ?? frame.codedWidth;
  const vh = frame.visibleRect?.height ?? frame.codedHeight;
  const [rw, rh] = targetSize(vw, vh, longSide);
  const fmt = frame.format ?? 'null';
  let small: Uint8Array;
  let path: 'exact' | 'fallback' = 'exact';
  let shrinkFrom = performance.now(); // restarted after the copy
  if (Y_8BIT.has(fmt) || Y_16BIT.has(fmt)) {
    const size = frame.allocationSize();
    if (scratch.length < size) scratch = new Uint8Array(size);
    const c0 = performance.now();
    const layout = await frame.copyTo(scratch);
    shrinkFrom = performance.now();
    lumaClock.copy += (shrinkFrom - c0) / 1000;
    const { offset, stride } = layout[0];
    if (Y_8BIT.has(fmt)) {
      small = boxDownscale(scratch.subarray(offset), vw, vh, stride, rw, rh);
    } else {
      const bits = fmt.endsWith('P12') ? 4 : 2; // to 8 bit: 10-bit >> 2, 12-bit >> 4
      const y8 = new Uint8Array(vw * vh);
      const view = new DataView(scratch.buffer, scratch.byteOffset + offset);
      for (let r = 0; r < vh; r++) {
        for (let c = 0; c < vw; c++) y8[r * vw + c] = view.getUint16(r * stride + c * 2, true) >> bits;
      }
      small = boxDownscale(y8, vw, vh, vw, rw, rh);
    }
  } else {
    // RGB or opaque GPU frames: draw at target size and compute BT.601 luma (not bit-exact)
    path = 'fallback';
    const canvas = new OffscreenCanvas(rw, rh);
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(frame, 0, 0, rw, rh);
    const px = ctx.getImageData(0, 0, rw, rh).data;
    small = new Uint8Array(rw * rh);
    for (let i = 0; i < rw * rh; i++) {
      small[i] = Math.round(16 + (65.738 * px[i * 4] + 129.057 * px[i * 4 + 1] + 25.064 * px[i * 4 + 2]) / 256);
    }
  }
  const r = rotateCw(small, rw, rh, rotation);
  lumaClock.shrink += (performance.now() - shrinkFrom) / 1000;
  return { data: r.data, w: r.w, h: r.h, path, format: fmt };
}

/** Draw a frame in display orientation onto a canvas of the given size (thumbnails, detector input). */
export function drawRotated(ctx: OffscreenCanvasRenderingContext2D, frame: VideoFrame, rotation: number,
  outW: number, outH: number): void {
  ctx.save();
  ctx.translate(outW / 2, outH / 2);
  ctx.rotate((rotation * Math.PI) / 180);
  const swap = rotation % 180 !== 0;
  const dw = swap ? outH : outW;
  const dh = swap ? outW : outH;
  ctx.drawImage(frame, -dw / 2, -dh / 2, dw, dh);
  ctx.restore();
}
