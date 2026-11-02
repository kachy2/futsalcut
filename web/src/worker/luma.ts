/**
 * VideoFrame -> 640-px luma sample. Exact path (parity contract, plan §3.1): copy the decoder's Y plane,
 * integer box downscale in coded orientation, then rotate to display orientation.
 * The copy (which needs the frame) and the shrink (which needs only the copied bytes) are separate steps, so that the
 * shrink can run in a helper worker while the analysis worker copies the next frame; together they are frameToLuma.
 */
import { boxDownscale, targetSize } from '../core/downscale';
import { rotateCw } from '../core/rotate';

const Y_8BIT = new Set(['I420', 'I420A', 'I422', 'I422A', 'I444', 'I444A', 'NV12']);
const Y_16BIT = new Set(['I420P10', 'I420AP10', 'I422P10', 'I422AP10', 'I444P10', 'I444AP10',
  'I420P12', 'I420AP12', 'I422P12', 'I422AP12', 'I444P12', 'I444AP12']);

let scratch = new ArrayBuffer(0);

/** Seconds spent so far in the two parts of the exact path: copying the frame out of the decoder, shrinking and rotating the luma (diagnostics). */
export const lumaClock = { copy: 0, shrink: 0 };

export interface LumaSample { data: Uint8Array; w: number; h: number; path: 'exact' | 'fallback'; format: string }

/** A frame's planes copied out of the decoder: the bytes, where each plane starts, and how to read the colours. */
export interface RawPlanes {
  buf: ArrayBuffer;
  fmt: string;
  layout: { offset: number; stride: number }[];
  vw: number;
  vh: number;
  /** BT.709 for HD and up unless the frame says otherwise; only the thumbnail uses it */
  bt709: boolean;
  fullRange: boolean;
}

/** Copy the planes of a frame whose luma can be read exactly (8- or 16-bit YUV); null for RGB or opaque GPU frames.
 * `buf` is reused when it is large enough. */
export async function copyPlanes(frame: VideoFrame, buf?: ArrayBuffer): Promise<RawPlanes | null> {
  const fmt = frame.format ?? 'null';
  if (!Y_8BIT.has(fmt) && !Y_16BIT.has(fmt)) return null;
  const vw = frame.visibleRect?.width ?? frame.codedWidth;
  const vh = frame.visibleRect?.height ?? frame.codedHeight;
  const size = frame.allocationSize();
  const target = buf && buf.byteLength >= size ? buf : new ArrayBuffer(size);
  const c0 = performance.now();
  const layout = await frame.copyTo(new Uint8Array(target, 0, size));
  lumaClock.copy += (performance.now() - c0) / 1000;
  const matrix = frame.colorSpace?.matrix;
  return {
    buf: target, fmt, layout: layout.map((p) => ({ offset: p.offset, stride: p.stride })), vw, vh,
    bt709: matrix ? matrix === 'bt709' : vh >= 720, fullRange: !!frame.colorSpace?.fullRange,
  };
}

/** The exact 640-px luma from copied planes: integer box downscale of the Y plane, then rotation to display orientation. */
export function shrinkPlanes(raw: RawPlanes, rotation: number, longSide = 640): LumaSample {
  const s0 = performance.now();
  const [rw, rh] = targetSize(raw.vw, raw.vh, longSide);
  const { offset, stride } = raw.layout[0];
  let small: Uint8Array;
  if (Y_8BIT.has(raw.fmt)) {
    small = boxDownscale(new Uint8Array(raw.buf, offset), raw.vw, raw.vh, stride, rw, rh);
  } else {
    const bits = raw.fmt.endsWith('P12') ? 4 : 2; // to 8 bit: 10-bit >> 2, 12-bit >> 4
    const y8 = new Uint8Array(raw.vw * raw.vh);
    const view = new DataView(raw.buf, offset);
    for (let r = 0; r < raw.vh; r++) {
      for (let c = 0; c < raw.vw; c++) y8[r * raw.vw + c] = view.getUint16(r * stride + c * 2, true) >> bits;
    }
    small = boxDownscale(y8, raw.vw, raw.vh, raw.vw, rw, rh);
  }
  const r = rotateCw(small, rw, rh, rotation);
  lumaClock.shrink += (performance.now() - s0) / 1000;
  return { data: r.data, w: r.w, h: r.h, path: 'exact', format: raw.fmt };
}

export async function frameToLuma(frame: VideoFrame, rotation: number, longSide = 640): Promise<LumaSample> {
  const raw = await copyPlanes(frame, scratch);
  if (raw) {
    scratch = raw.buf;
    return shrinkPlanes(raw, rotation, longSide);
  }
  // RGB or opaque GPU frames: draw at target size and compute BT.601 luma (not bit-exact)
  const s0 = performance.now();
  const vw = frame.visibleRect?.width ?? frame.codedWidth;
  const vh = frame.visibleRect?.height ?? frame.codedHeight;
  const [rw, rh] = targetSize(vw, vh, longSide);
  const canvas = new OffscreenCanvas(rw, rh);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(frame, 0, 0, rw, rh);
  const px = ctx.getImageData(0, 0, rw, rh).data;
  const small = new Uint8Array(rw * rh);
  for (let i = 0; i < rw * rh; i++) {
    small[i] = Math.round(16 + (65.738 * px[i * 4] + 129.057 * px[i * 4 + 1] + 25.064 * px[i * 4 + 2]) / 256);
  }
  const r = rotateCw(small, rw, rh, rotation);
  lumaClock.shrink += (performance.now() - s0) / 1000;
  return { data: r.data, w: r.w, h: r.h, path: 'fallback', format: frame.format ?? 'null' };
}

/** Chroma layout of the 8-bit formats a thumbnail can be made from without the frame: [x subsampling, y subsampling, interleaved]. */
const CHROMA: Record<string, [number, number, boolean]> = {
  NV12: [2, 2, true], I420: [2, 2, false], I420A: [2, 2, false], I422: [2, 1, false], I422A: [2, 1, false], I444: [1, 1, false], I444A: [1, 1, false],
};

export const canThumbFromPlanes = (raw: RawPlanes) => raw.fmt in CHROMA;

/**
 * A small RGBA picture (display orientation, `outW` x `outH`, stretched like drawRotated) from copied 8-bit YUV planes,
 * so that the timeline thumbnail needs no second drawing of the frame on the graphics chip. Each output pixel averages a
 * 4 x 4 grid of luma samples inside its block and takes the chroma at the block centre.
 */
export function planesToRgba(raw: RawPlanes, rotation: number, outW: number, outH: number): Uint8ClampedArray {
  const [sx, sy, interleaved] = CHROMA[raw.fmt];
  const swap = rotation % 180 !== 0;
  const cw = swap ? outH : outW; // coded-orientation size of the picture before rotation
  const ch = swap ? outW : outH;
  const bytes = new Uint8Array(raw.buf);
  const [yl, ul, vl] = raw.layout;
  const coded = new Uint8ClampedArray(cw * ch * 4);
  const kr = raw.bt709 ? 0.2126 : 0.299;
  const kb = raw.bt709 ? 0.0722 : 0.114;
  const kg = 1 - kr - kb;
  const rv = 2 * (1 - kr);
  const bu = 2 * (1 - kb);
  const gu = (bu * kb) / kg;
  const gv = (rv * kr) / kg;
  const yScale = raw.fullRange ? 1 : 255 / 219;
  const cScale = raw.fullRange ? 1 : 255 / 224;
  const yOff = raw.fullRange ? 0 : 16;
  for (let oy = 0; oy < ch; oy++) {
    const y0 = Math.floor((oy * raw.vh) / ch);
    const y1 = Math.max(y0 + 1, Math.floor(((oy + 1) * raw.vh) / ch));
    for (let ox = 0; ox < cw; ox++) {
      const x0 = Math.floor((ox * raw.vw) / cw);
      const x1 = Math.max(x0 + 1, Math.floor(((ox + 1) * raw.vw) / cw));
      let sum = 0;
      let n = 0;
      for (let k = 0; k < 4; k++) {
        const yy = y0 + Math.floor(((y1 - y0) * (2 * k + 1)) / 8);
        for (let j = 0; j < 4; j++) {
          const xx = x0 + Math.floor(((x1 - x0) * (2 * j + 1)) / 8);
          sum += bytes[yl.offset + yy * yl.stride + xx];
          n++;
        }
      }
      const cxp = Math.floor((x0 + x1) / 2 / sx);
      const cyp = Math.floor((y0 + y1) / 2 / sy);
      const u = interleaved ? bytes[ul.offset + cyp * ul.stride + cxp * 2] : bytes[ul.offset + cyp * ul.stride + cxp];
      const v = interleaved ? bytes[ul.offset + cyp * ul.stride + cxp * 2 + 1] : bytes[vl.offset + cyp * vl.stride + cxp];
      const Y = (sum / n - yOff) * yScale;
      const U = (u - 128) * cScale;
      const V = (v - 128) * cScale;
      const o = (oy * cw + ox) * 4;
      coded[o] = Y + rv * V;
      coded[o + 1] = Y - gu * U - gv * V;
      coded[o + 2] = Y + bu * U;
      coded[o + 3] = 255;
    }
  }
  return rotateRgba(coded, cw, ch, rotation);
}

/** Clockwise rotation of an RGBA picture by 0/90/180/270 degrees (same mapping as rotateCw). */
export function rotateRgba(src: Uint8ClampedArray, w: number, h: number, degrees: number): Uint8ClampedArray {
  const k = (((degrees / 90) % 4) + 4) % 4;
  if (k === 0) return src;
  const out = new Uint8ClampedArray(src.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const s = (y * w + x) * 4;
      let d: number;
      if (k === 1) d = (x * h + (h - 1 - y)) * 4;       // 90 cw: output is h wide
      else if (k === 2) d = ((h - 1 - y) * w + (w - 1 - x)) * 4;
      else d = ((w - 1 - x) * h + y) * 4;               // 270 cw
      out[d] = src[s];
      out[d + 1] = src[s + 1];
      out[d + 2] = src[s + 2];
      out[d + 3] = src[s + 3];
    }
  }
  return out;
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
