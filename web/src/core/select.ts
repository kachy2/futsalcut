/**
 * From per-sample scores to kept intervals — twin of python/src/futsalcut/select.py.
 * Sample i covers [t[i], t[i+1]) (the last one covers [t[n-1], t[n-1] + 1)).
 */
export interface PostParams {
  smoothWindow: number;
  smoothNeed: number;
  mergeGap: number;
  minLen: number;
  padBefore: number;
  padAfter: number;
}

export const DEFAULT_POST: PostParams = {
  smoothWindow: 3, smoothNeed: 2, mergeGap: 5, minLen: 4, padBefore: 2, padAfter: 2,
};

export type Interval = [number, number];

export function sampleEnds(t: ArrayLike<number>): Float64Array {
  const n = t.length;
  const end = new Float64Array(n);
  for (let i = 0; i < n - 1; i++) end[i] = t[i + 1];
  if (n) end[n - 1] = t[n - 1] + 1;
  return end;
}

/** k-of-n majority in a centred window; at the edges the window and the requirement shrink together. */
export function smooth(dec: ArrayLike<boolean>, window = 3, need = 2): boolean[] {
  const n = dec.length;
  if (window <= 1) return Array.from(dec, Boolean);
  const half = Math.floor(window / 2);
  const c = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) c[i + 1] = c[i] + (dec[i] ? 1 : 0);
  const out: boolean[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half);
    const hi = Math.min(n, i + half + 1);
    out[i] = c[hi] - c[lo] >= Math.min(need, hi - lo);
  }
  return out;
}

export function runsToIntervals(mask: ArrayLike<boolean>, t: ArrayLike<number>): Interval[] {
  const ends = sampleEnds(t);
  const out: Interval[] = [];
  let i = 0;
  const n = mask.length;
  while (i < n) {
    if (mask[i]) {
      let j = i;
      while (j + 1 < n && mask[j + 1]) j++;
      out.push([t[i], ends[j]]);
      i = j + 1;
    } else i++;
  }
  return out;
}

export function postprocess(intervals: Interval[], duration: number, p: PostParams): Interval[] {
  const merged: Interval[] = [];
  for (const [s, e] of intervals) {
    const last = merged[merged.length - 1];
    if (last && s - last[1] <= p.mergeGap) last[1] = Math.max(last[1], e);
    else merged.push([s, e]);
  }
  const padded: Interval[] = [];
  for (const [s, e] of merged) {
    if (e - s < p.minLen) continue;
    const s2 = Math.max(0, s - p.padBefore);
    const e2 = Math.min(duration, e + p.padAfter);
    const last = padded[padded.length - 1];
    if (last && s2 <= last[1]) last[1] = Math.max(last[1], e2);
    else padded.push([s2, e2]);
  }
  return padded;
}

/** Keyframe timestamps jitter by a few ms around the nominal grid. */
export const SNAP_TOL = 0.05;

/** start -> latest sample time <= start + SNAP_TOL; end -> earliest sample end >= end - SNAP_TOL. */
export function snapToSamples(intervals: Interval[], t: ArrayLike<number>): Interval[] {
  const ends = sampleEnds(t);
  const n = t.length;
  const out: Interval[] = [];
  for (const [s, e] of intervals) {
    let i = upperBound(t, s + SNAP_TOL) - 1;
    let j = lowerBound(ends, e - SNAP_TOL);
    i = Math.max(0, i);
    j = Math.min(n - 1, j);
    const s2 = t[i];
    const e2 = ends[j];
    const last = out[out.length - 1];
    if (last && s2 <= last[1]) last[1] = Math.max(last[1], e2);
    else out.push([s2, e2]);
  }
  return out;
}

function upperBound(a: ArrayLike<number>, x: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m] <= x) lo = m + 1;
    else hi = m;
  }
  return lo;
}

function lowerBound(a: ArrayLike<number>, x: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m] < x) lo = m + 1;
    else hi = m;
  }
  return lo;
}

export function select(score: ArrayLike<number>, t: ArrayLike<number>, threshold: number,
  p: PostParams = DEFAULT_POST, duration?: number): Interval[] {
  const dur = duration ?? sampleEnds(t)[t.length - 1];
  const dec = Array.from(score, (s) => s >= threshold);
  return snapToSamples(postprocess(runsToIntervals(smooth(dec, p.smoothWindow, p.smoothNeed), t), dur, p), t);
}

export function keptSeconds(iv: Interval[]): number {
  let s = 0;
  for (const [a, b] of iv) s += b - a;
  return s;
}

/** Lowest candidate threshold whose kept duration does not exceed targetS (binary search; kept
 * duration is non-increasing in the threshold). Falls back to "keep nothing". */
export function thresholdForBudget(score: ArrayLike<number>, t: ArrayLike<number>, targetS: number,
  p: PostParams = DEFAULT_POST, duration?: number): { threshold: number; intervals: Interval[] } {
  const cands = Float64Array.from(new Set(Array.from(score))).sort();
  let lo = 0;
  let hi = cands.length - 1;
  let best: { threshold: number; intervals: Interval[] } | null = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const iv = select(score, t, cands[mid], p, duration);
    if (keptSeconds(iv) <= targetS) {
      best = { threshold: cands[mid], intervals: iv };
      hi = mid - 1;
    } else lo = mid + 1;
  }
  if (!best) {
    const thr = cands.length ? nextUp(cands[cands.length - 1]) : Infinity;
    best = { threshold: thr, intervals: select(score, t, thr, p, duration) };
  }
  return best;
}

function nextUp(x: number): number {
  if (x === Infinity) return x;
  const buf = new DataView(new ArrayBuffer(8));
  if (x === 0) return Number.MIN_VALUE;
  buf.setFloat64(0, x);
  const bits = buf.getBigUint64(0);
  buf.setBigUint64(0, x > 0 ? bits + 1n : bits - 1n);
  return buf.getFloat64(0);
}
