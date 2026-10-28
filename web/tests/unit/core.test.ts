import { describe, expect, it } from 'vitest';
import { boxDownscale, targetSize } from '../../src/core/downscale';
import { clipBelow, pointInPolygon, polygonMask } from '../../src/core/roi';
import { DEFAULT_POST, keptSeconds, select, smooth, thresholdForBudget } from '../../src/core/select';

describe('select (twin of python tests/test_select.py)', () => {
  it('smooths 2 of 3', () => {
    const d = [0, 1, 0, 1, 1, 0, 0, 1, 0].map(Boolean);
    expect(smooth(d, 3, 2).map(Number)).toEqual([0, 0, 1, 1, 1, 0, 0, 0, 0]);
  });
  it('merges, drops and pads', () => {
    const t = Array.from({ length: 60 }, (_, i) => i);
    const s = new Array(60).fill(0);
    for (let i = 10; i < 16; i++) s[i] = 1;
    for (let i = 19; i < 22; i++) s[i] = 1;
    s[40] = s[41] = 1;
    const p = { smoothWindow: 1, smoothNeed: 1, mergeGap: 5, minLen: 4, padBefore: 2, padAfter: 2 };
    expect(select(s, t, 0.5, p)).toEqual([[8, 24]]);
  });
  it('budget search respects the target', () => {
    let x = 1;
    const rand = () => ((x = (x * 16807) % 2147483647) / 2147483647);
    const t = Array.from({ length: 600 }, (_, i) => i);
    const s = t.map(() => rand());
    const { intervals } = thresholdForBudget(s, t, 120, DEFAULT_POST);
    expect(keptSeconds(intervals)).toBeLessThanOrEqual(120);
  });
});

describe('roi', () => {
  it('point in polygon and clipping', () => {
    const sq: [number, number][] = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]];
    expect(pointInPolygon(sq, 0.5, 0.5)).toBe(true);
    expect(pointInPolygon(sq, 0.05, 0.5)).toBe(false);
    const c = clipBelow(sq, 0.5);
    expect(Math.min(...c.map((p) => p[1]))).toBeCloseTo(0.5);
    const m = polygonMask(sq, 10, 10);
    expect(m.reduce((a, b) => a + b, 0)).toBe(64);
  });
});

describe('downscale (twin of python box_downscale)', () => {
  it('rounds half up and matches a naive box mean', () => {
    expect(boxDownscale(new Uint8Array([0, 1, 0, 1]), 2, 2, 2, 1, 1)[0]).toBe(1);
    const W = 61;
    const H = 37;
    const src = new Uint8Array(W * H).map((_, i) => (i * 37 + 11) % 256);
    const out = boxDownscale(src, W, H, W, 16, 9);
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 16; c++) {
        const r0 = Math.floor((r * H) / 9), r1 = Math.floor(((r + 1) * H) / 9);
        const c0 = Math.floor((c * W) / 16), c1 = Math.floor(((c + 1) * W) / 16);
        let s = 0;
        for (let y = r0; y < r1; y++) for (let x = c0; x < c1; x++) s += src[y * W + x];
        const n = (r1 - r0) * (c1 - c0);
        expect(out[r * 16 + c]).toBe(Math.floor((s + (n >> 1)) / n));
      }
    }
    expect(targetSize(3840, 2160)).toEqual([640, 360]);
    expect(targetSize(4032, 3024)).toEqual([640, 480]);
  });
});

import { calibrate, ensembleProb, pickQuestions, quickCalibrate } from '../../src/core/ensemble';

describe('ensemble (twin of python tests/test_ensemble.py)', () => {
  it('calibrator is monotone and combines', () => {
    const c = { scale: 1, a: 2, b: -3 };
    expect(calibrate(c, 0)).toBeLessThan(calibrate(c, 5));
    const m = { names: ['x'], calibrators: { x: c }, weights: [2], bias: -6 };
    const p = ensembleProb(m, { x: [0, 1, 10, 100] });
    expect(p[3]).toBeGreaterThan(p[0]);
  });
  it('questions are spread; quick calibration follows answers with a margin', () => {
    const t = Array.from({ length: 300 }, (_, i) => i);
    const a = t.map((v) => (v < 150 ? 0.9 : 0.1));
    const b = t.map((v) => (v < 150 ? 0.1 : 0.9));
    const q = pickQuestions({ a, b }, t, 5, 20);
    expect(q.length).toBe(5);
    const answers = new Map(q.map((i) => [i, t[i] >= 150] as [number, boolean]));
    const r = quickCalibrate({ a, b }, answers, 'a');
    expect(r.name).toBe('b');
    expect([r.agree, r.total]).toEqual([5, 5]);
  });
});

import { decodeEta, itemEta } from '../../src/core/progress';

describe('progress estimates', () => {
  it('uses the worker guess until two items are done, then the stage pace', () => {
    expect(itemEta(0, 200, 0, 1.5)).toEqual({ perItemS: 1.5, leftS: 300 });
    expect(itemEta(1, 200, 1.4, 1.5)?.perItemS).toBe(1.5);
    const e = itemEta(50, 200, 100, 1.5)!;
    expect(e.perItemS).toBe(2);
    expect(e.leftS).toBe(300);
    expect(itemEta(0, 10)).toBeNull();
    expect(itemEta(10, 10, 20)?.leftS).toBe(0);
  });
  it('estimates the decode stage from the video time covered', () => {
    expect(decodeEta(0, 700, 5)).toBeNull();
    const e = decodeEta(100, 700, 50)!;
    expect(e.rate).toBe(2);
    expect(e.leftS).toBe(300);
  });
});
