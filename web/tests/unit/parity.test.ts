// Python <-> TypeScript parity on the synthetic fixture (tools/make_fixtures.sh regenerates luma.u8).
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { blockWindow, lowerMedian, pass1Sample, pass2, perspectiveFromBlobs, perspectiveFromBoxes, BG_EVERY,
  type Pass1 } from '../../src/core/features';
import { cascadeProb, detectorCount, lightProb, postParams, type ModelFile } from '../../src/core/model';
import { ensembleProb } from '../../src/core/ensemble';
import { polygonMask, type Point } from '../../src/core/roi';
import { StreamingPass1 } from '../../src/core/streaming';
import { select } from '../../src/core/select';

const DIR = process.env.PARITY_DIR ?? resolve(__dirname, '../fixtures/synthetic');
const has = existsSync(resolve(DIR, 'luma.u8')) && existsSync(resolve(DIR, 'fixture.json'));
const MODEL = resolve(__dirname, '../../../python/src/futsalcut/models/ensemble_v1.json');

const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(1e-12, Math.abs(b));

describe.skipIf(!has)('parity with the Python reference (synthetic fixture)', () => {
  const fx = has ? JSON.parse(readFileSync(resolve(DIR, 'fixture.json'), 'utf8')) : null;
  const { n, W, H, t } = fx?.meta ?? {};
  const raw = has ? readFileSync(resolve(DIR, 'luma.u8')) : Buffer.alloc(0);
  const frame = (i: number) => new Uint8Array(raw.buffer, raw.byteOffset + i * W * H, W * H);
  const zonePoly: Point[] = fx?.zone ?? [];
  const zone = has ? polygonMask(zonePoly, W, H) : new Uint8Array(0);
  let p1: Pass1[] = [];

  it('zone mask matches', () => {
    expect(zone.reduce((a, b) => a + b, 0)).toBe(fx.zone_pixels);
  });

  it('pass 1 (background, majority filter, gain, blobs) is identical', () => {
    p1 = [];
    let bg: Uint8Array | null = null;
    let bgBlock = -1;
    for (let i = 0; i < n; i++) {
      const b = Math.floor(i / BG_EVERY);
      if (b !== bgBlock) {
        bg = lowerMedian(blockWindow(n, b).map(frame), W * H);
        bgBlock = b;
      }
      const s = pass1Sample(frame(i), bg!, i ? frame(i - 1) : null, zone, W, H);
      p1.push(s);
      const e = fx.pass1[i];
      expect(Array.from(s.rowFg)).toEqual(e.rowFg);
      expect(Array.from(s.rowFgGain)).toEqual(e.rowFgGain);
      expect(s.diffZone).toBe(e.diffZone);
      expect(s.blobs.map((q) => [q.x, q.y, q.w, q.h, q.a, q.footY, q.inZone ? 1 : 0])).toEqual(e.blobs);
    }
  }, 300_000);

  it('streaming pass 1 (bounded memory) equals the batch reference', () => {
    const sp = new StreamingPass1(W, H, zone);
    for (let i = 0; i < n; i++) sp.push(new Uint8Array(frame(i)));
    const res = sp.end();
    expect(res.length).toBe(n);
    res.forEach((s, i) => {
      expect(Array.from(s.rowFgGain)).toEqual(fx.pass1[i].rowFgGain);
      expect(s.blobs.length).toBe(fx.pass1[i].blobs.length);
    });
  }, 300_000);

  it('perspective fits and pass 2 scores match', () => {
    const zoneTop = Math.min(...zonePoly.map((p) => p[1])) * H;
    const pb = perspectiveFromBlobs(p1, zoneTop);
    if (fx.perspective_blobs) {
      expect(rel(pb!.a, fx.perspective_blobs[0])).toBeLessThan(1e-12);
      expect(Math.abs(pb!.c - fx.perspective_blobs[1])).toBeLessThan(1e-9);
    } else expect(pb).toBeNull();
    const pd = perspectiveFromBoxes(fx.boxes, H, zoneTop);
    expect(pd).not.toBeNull();
    expect(rel(pd!.a, fx.perspective_boxes[0])).toBeLessThan(1e-12);
    const s = pass2(p1, { a: fx.fixed_perspective[0], c: fx.fixed_perspective[1], source: 'fixed' }, H);
    for (const k of Object.keys(fx.scores)) {
      const got = (s as unknown as Record<string, Float64Array>)[k];
      fx.scores[k].forEach((v: number, i: number) => expect(rel(got[i], v)).toBeLessThan(1e-12));
    }
  });

  it('detector counts, ensemble probabilities and selection match', () => {
    const model: ModelFile = JSON.parse(readFileSync(MODEL, 'utf8'));
    const det = fx.boxes.map((bs: number[][]) => detectorCount(bs, zonePoly, model.detector.min_score));
    det.forEach((v: number, i: number) => expect(rel(v, fx.detector[i])).toBeLessThan(1e-12));
    const s = pass2(p1, { a: fx.fixed_perspective[0], c: fx.fixed_perspective[1], source: 'fixed' }, H);
    const light = lightProb(model, s);
    light.forEach((v, i) => expect(Math.abs(v - fx.light[i])).toBeLessThan(1e-12));
    const full = ensembleProb(model.full, { ...(s as unknown as Record<string, Float64Array>), dfine_count: det });
    full.forEach((v, i) => expect(Math.abs(v - fx.full[i])).toBeLessThan(1e-12));
    expect(select(light, t, model.threshold, postParams(model))).toEqual(fx.intervals_light);
    // cascade uses the full model exactly where the light model is unsure
    const detMap = new Map<number, number>(det.map((v: number, i: number) => [i, v] as [number, number]));
    const casc = cascadeProb(model, s, light, detMap);
    const [lo, hi] = model.detector.cascade;
    casc.forEach((v, i) => expect(v).toBe(light[i] > lo && light[i] < hi ? full[i] : light[i]));
  });
});
