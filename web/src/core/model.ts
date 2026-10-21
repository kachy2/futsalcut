/**
 * Shipped ensemble model (python/src/futsalcut/models/ensemble_v1.json, fitted by eval/fit_model.py)
 * and the auto-mode pipeline: light probabilities -> detector only where unsure -> final probabilities.
 */
import { ensembleProb, type EnsembleModel } from './ensemble';
import { pointInPolygon, type Point } from './roi';
import type { PostParams } from './select';
import type { Scores } from './features';

export interface ModelFile {
  version: number;
  threshold: number;
  post: { smooth_window: number; smooth_need: number; merge_gap: number; min_len: number; pad_before: number; pad_after: number };
  detector: { min_score: number; perspective_stride: number; cascade: [number, number] };
  light: EnsembleModel;
  full: EnsembleModel;
}

export function postParams(m: ModelFile): PostParams {
  const p = m.post;
  return { smoothWindow: p.smooth_window, smoothNeed: p.smooth_need, mergeGap: p.merge_gap, minLen: p.min_len,
    padBefore: p.pad_before, padAfter: p.pad_after };
}

/** Detector score per sample: confident in-zone persons + 0.99-capped confidence mass (tie-break). */
export function detectorCount(boxes: number[][], zone: Point[], minScore: number): number {
  let hard = 0;
  let soft = 0;
  for (const [x1, , x2, y2, s] of boxes) {
    if (!pointInPolygon(zone, (x1 + x2) / 2, Math.min(0.9999, y2))) continue;
    if (s >= minScore) hard++;
    soft += s;
  }
  return hard + Math.min(0.99, soft / 10);
}

export function lightProb(m: ModelFile, s: Scores): Float64Array {
  return ensembleProb(m.light, s as unknown as Record<string, ArrayLike<number>>);
}

/** Samples whose light probability is inside the cascade band (need the detector). */
export function unsureSamples(m: ModelFile, light: ArrayLike<number>): number[] {
  const [lo, hi] = m.detector.cascade;
  const out: number[] = [];
  for (let i = 0; i < light.length; i++) if (light[i] > lo && light[i] < hi) out.push(i);
  return out;
}

/** Final probabilities: full ensemble where a detector count is available and the sample is unsure. */
export function cascadeProb(m: ModelFile, s: Scores, light: Float64Array, detector: Map<number, number>): Float64Array {
  const out = Float64Array.from(light);
  const [lo, hi] = m.detector.cascade;
  const idx = [...detector.keys()].filter((i) => light[i] > lo && light[i] < hi).sort((a, b) => a - b);
  if (!idx.length) return out;
  const sub: Record<string, number[]> = {};
  for (const name of m.full.names) sub[name] = [];
  for (const i of idx) {
    for (const name of m.full.names) {
      sub[name].push(name === 'dfine_count' ? detector.get(i)! : (s as unknown as Record<string, Float64Array>)[name][i]);
    }
  }
  const p = ensembleProb(m.full, sub);
  idx.forEach((i, k) => { out[i] = p[k]; });
  return out;
}
