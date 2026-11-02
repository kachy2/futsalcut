// The luma path split into copy and shrink (so the shrink can run in a helper worker) must stay the exact path, and the thumbnail
// made from copied YUV planes must show the right picture in the right orientation.
import { describe, expect, it } from 'vitest';
import { boxDownscale } from '../../src/core/downscale';
import { rotateCw } from '../../src/core/rotate';
import { planesToRgba, rotateRgba, shrinkPlanes, type RawPlanes } from '../../src/worker/luma';

type Plane = (x: number, y: number) => number;

function nv12(w: number, h: number, Y: Plane, U: Plane, V: Plane): RawPlanes {
  const ySize = w * h;
  const buf = new ArrayBuffer(ySize + ySize / 2);
  const b = new Uint8Array(buf);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) b[y * w + x] = Y(x, y);
  for (let j = 0; j < h / 2; j++) {
    for (let i = 0; i < w / 2; i++) {
      b[ySize + j * w + 2 * i] = U(i, j);
      b[ySize + j * w + 2 * i + 1] = V(i, j);
    }
  }
  return { buf, fmt: 'NV12', layout: [{ offset: 0, stride: w }, { offset: ySize, stride: w }], vw: w, vh: h, bt709: true, fullRange: false };
}

function i420(w: number, h: number, Y: Plane, U: Plane, V: Plane): RawPlanes {
  const ySize = w * h;
  const cw = w / 2;
  const cSize = cw * (h / 2);
  const buf = new ArrayBuffer(ySize + 2 * cSize);
  const b = new Uint8Array(buf);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) b[y * w + x] = Y(x, y);
  for (let j = 0; j < h / 2; j++) {
    for (let i = 0; i < cw; i++) {
      b[ySize + j * cw + i] = U(i, j);
      b[ySize + cSize + j * cw + i] = V(i, j);
    }
  }
  return { buf, fmt: 'I420', layout: [{ offset: 0, stride: w }, { offset: ySize, stride: cw }, { offset: ySize + cSize, stride: cw }],
    vw: w, vh: h, bt709: true, fullRange: false };
}

let seed = 9;
const rnd = () => ((seed = (seed * 16807) % 2147483647) % 256);

describe('luma split into copy and shrink', () => {
  it('shrinkPlanes is the exact path: box downscale of the Y plane, then rotation', () => {
    const w = 128;
    const h = 72;
    const ys = Array.from({ length: w * h }, rnd);
    const raw = nv12(w, h, (x, y) => ys[y * w + x], () => 128, () => 128);
    for (const rot of [0, 90, 180, 270]) {
      const got = shrinkPlanes(raw, rot, 64);
      const small = boxDownscale(Uint8Array.from(ys), w, h, w, 64, 36);
      const want = rotateCw(small, 64, 36, rot);
      expect([got.w, got.h]).toEqual([want.w, want.h]);
      expect(Array.from(got.data)).toEqual(Array.from(want.data));
      expect(got.path).toBe('exact');
    }
  });
});

describe('thumbnail from YUV planes', () => {
  it('turns mid grey (limited range) into RGB 130', () => {
    const px = planesToRgba(nv12(64, 36, () => 128, () => 128, () => 128), 0, 16, 9);
    expect(px.length).toBe(16 * 9 * 4);
    for (let i = 0; i < px.length; i += 4) expect([px[i], px[i + 1], px[i + 2], px[i + 3]]).toEqual([130, 130, 130, 255]);
  });

  it('turns BT.709 red back into red', () => {
    const px = planesToRgba(nv12(64, 36, () => 63, () => 102, () => 240), 0, 8, 4);
    expect(px[0]).toBeGreaterThan(250);
    expect(px[1]).toBeLessThan(5);
    expect(px[2]).toBeLessThan(5);
  });

  it('gives the same picture from NV12 and I420 planes of the same frame', () => {
    const w = 96;
    const h = 54;
    const ys = Array.from({ length: w * h }, rnd);
    const us = Array.from({ length: (w / 2) * (h / 2) }, rnd);
    const vs = Array.from({ length: (w / 2) * (h / 2) }, rnd);
    const Y: Plane = (x, y) => ys[y * w + x];
    const U: Plane = (x, y) => us[y * (w / 2) + x];
    const V: Plane = (x, y) => vs[y * (w / 2) + x];
    for (const rot of [0, 90, 180]) {
      expect(Array.from(planesToRgba(nv12(w, h, Y, U, V), rot, 24, 12))).toEqual(Array.from(planesToRgba(i420(w, h, Y, U, V), rot, 24, 12)));
    }
  });

  it('puts the picture in display orientation: a bright top-left corner ends up where rotateCw puts it', () => {
    const w = 64;
    const h = 32;
    const raw = nv12(w, h, (x, y) => (x < 8 && y < 8 ? 235 : 16), () => 128, () => 128);
    const outW = 16;
    const outH = 8;
    for (const rot of [0, 90, 180, 270]) {
      const px = planesToRgba(raw, rot, outW, outH);
      const bright = (x: number, y: number) => px[(y * outW + x) * 4] > 200;
      const corner = { 0: [0, 0], 90: [outW - 1, 0], 180: [outW - 1, outH - 1], 270: [0, outH - 1] }[rot]!;
      expect(bright(corner[0], corner[1])).toBe(true);
      expect(bright(outW - 1 - corner[0], outH - 1 - corner[1])).toBe(false);
    }
  });

  it('rotates RGBA the same way rotateCw rotates luma', () => {
    const w = 5;
    const h = 3;
    const lum = Uint8Array.from({ length: w * h }, (_, i) => i * 10);
    const rgba = new Uint8ClampedArray(w * h * 4);
    lum.forEach((v, i) => { rgba[i * 4] = v; rgba[i * 4 + 3] = 255; });
    for (const rot of [0, 90, 180, 270]) {
      const r = rotateRgba(rgba, w, h, rot);
      const want = rotateCw(lum, w, h, rot).data;
      expect(Array.from(want, (_, i) => r[i * 4])).toEqual(Array.from(want));
    }
  });
});
