/**
 * Light scorers — twin of python/src/futsalcut/features.py (see docs/algorithms.md §5).
 * Pass 1 runs per sample with a sliding window of luma samples; pass 2 runs once the perspective
 * model is known. All arithmetic mirrors the Python reference (integer ops, float64 in fixed order).
 */
export const FG_T = 22;
export const DIFF_T = 25;
export const BG_HALF = 60;
export const BG_STEP = 4;
export const BG_EVERY = 10;
export const MIN_BLOB_AREA = 6;

export interface Blob { x: number; y: number; w: number; h: number; a: number; footY: number; inZone: boolean }

export interface Pass1 {
  rowFg: Uint32Array;
  rowFgGain: Uint32Array;
  diffZone: number;
  blobs: Blob[];
}

/** Sample indices whose lower median forms the background of `block` (n = total sample count). */
export function blockWindow(n: number, block: number): number[] {
  const c = Math.min(n - 1, block * BG_EVERY + Math.floor(BG_EVERY / 2));
  const out: number[] = [];
  for (let i = Math.max(0, c - BG_HALF); i < Math.min(n, c + BG_HALF + 1); i += BG_STEP) out.push(i);
  return out;
}

/** Per-pixel lower median ((len-1)/2-th smallest) of 8-bit frames, via a 256-bin count per pixel. */
export function lowerMedian(frames: Uint8Array[], size: number): Uint8Array {
  const k = (frames.length - 1) >> 1;
  const out = new Uint8Array(size);
  const vals = new Uint8Array(frames.length);
  for (let p = 0; p < size; p++) {
    for (let f = 0; f < frames.length; f++) vals[f] = frames[f][p];
    vals.sort();
    out[p] = vals[k];
  }
  return out;
}

const hsum = { buf: new Uint8Array(0) };

/** 3x3 majority (>= 5 of 9 on), zero padding — separable box sum (horizontal, then vertical). */
export function majority3(mask: Uint8Array, W: number, H: number): Uint8Array {
  const N = W * H;
  if (hsum.buf.length < N) hsum.buf = new Uint8Array(N);
  const h = hsum.buf;
  for (let y = 0; y < H; y++) {
    const r = y * W;
    let prev = 0;
    let cur = mask[r];
    for (let x = 0; x < W; x++) {
      const next = x + 1 < W ? mask[r + x + 1] : 0;
      h[r + x] = prev + cur + next;
      prev = cur;
      cur = next;
    }
  }
  const out = new Uint8Array(N);
  for (let y = 0; y < H; y++) {
    const r = y * W;
    const up = y > 0 ? r - W : -1;
    const dn = y + 1 < H ? r + W : -1;
    for (let x = 0; x < W; x++) {
      const s = h[r + x] + (up >= 0 ? h[up + x] : 0) + (dn >= 0 ? h[dn + x] : 0);
      out[r + x] = s >= 5 ? 1 : 0;
    }
  }
  return out;
}

/** numpy.median of 8-bit values given as a 256-bin histogram (exact; mean of the two middle values). */
export function medianFromHist(hist: Uint32Array, n: number): number {
  if (!n) return NaN;
  const kLo = (n - 1) >> 1;
  const kHi = n >> 1;
  let acc = 0;
  let lo = -1;
  for (let v = 0; v < 256; v++) {
    acc += hist[v];
    if (lo < 0 && acc > kLo) lo = v;
    if (acc > kHi) return (lo + v) / 2;
  }
  return lo;
}

/** numpy.median of a numeric array (mean of the two middle values for even length). */
export function medianF64(values: ArrayLike<number>): number {
  const v = Float64Array.from(values).sort();
  const n = v.length;
  if (!n) return NaN;
  return n % 2 ? v[(n - 1) >> 1] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

/** numpy.percentile(..., method='linear'), including numpy's lerp formulation. */
export function percentileLinear(values: ArrayLike<number>, q: number): number {
  const v = Float64Array.from(values).sort();
  const pos = (v.length - 1) * (q / 100);
  const lo = Math.floor(pos);
  const hi = Math.min(v.length - 1, lo + 1);
  const t = pos - lo;
  const a = v[lo];
  const b = v[hi];
  const d = b - a;
  return t >= 0.5 ? b - d * (1 - t) : a + d * t;
}

/** 8-connected components with bounding boxes and areas (two-pass union-find). */
export function components(mask: Uint8Array, W: number, H: number): { x: number; y: number; w: number; h: number; a: number }[] {
  const labels = new Int32Array(W * H);
  const parent: number[] = [0];
  const find = (a: number): number => {
    while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
  };
  const union = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };
  let next = 1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = y * W + x;
      if (!mask[p]) continue;
      const nb: number[] = [];
      if (x > 0 && labels[p - 1]) nb.push(labels[p - 1]);
      if (y > 0) {
        const up = p - W;
        if (x > 0 && labels[up - 1]) nb.push(labels[up - 1]);
        if (labels[up]) nb.push(labels[up]);
        if (x < W - 1 && labels[up + 1]) nb.push(labels[up + 1]);
      }
      if (!nb.length) {
        parent.push(next);
        labels[p] = next++;
      } else {
        let m = nb[0];
        for (const l of nb) m = Math.min(m, l);
        labels[p] = m;
        for (const l of nb) union(m, l);
      }
    }
  }
  const stats = new Map<number, { x0: number; y0: number; x1: number; y1: number; a: number }>();
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const l = labels[y * W + x];
      if (!l) continue;
      const r = find(l);
      const s = stats.get(r);
      if (!s) stats.set(r, { x0: x, y0: y, x1: x, y1: y, a: 1 });
      else {
        if (x < s.x0) s.x0 = x;
        if (x > s.x1) s.x1 = x;
        if (y > s.y1) s.y1 = y;
        s.a++;
      }
    }
  }
  return [...stats.values()].map((s) => ({ x: s.x0, y: s.y0, w: s.x1 - s.x0 + 1, h: s.y1 - s.y0 + 1, a: s.a }));
}

export function blobsOf(fg: Uint8Array, zone: Uint8Array, W: number, H: number): Blob[] {
  const out: Blob[] = [];
  for (const c of components(fg, W, H)) {
    if (c.a < MIN_BLOB_AREA) continue;
    const footY = Math.min(H - 1, c.y + c.h - 1);
    const footX = Math.min(W - 1, c.x + (c.w >> 1));
    out.push({ ...c, footY, inZone: zone[footY * W + footX] === 1 });
  }
  out.sort((p, q) => p.y - q.y || p.x - q.x || p.h - q.h || p.w - q.w);
  return out;
}

export function pass1Sample(img: Uint8Array, bg: Uint8Array, prev: Uint8Array | null, zone: Uint8Array,
  W: number, H: number): Pass1 {
  const N = W * H;
  const d = new Uint8Array(N);
  const dz = new Uint8Array(N);
  const hImg = new Uint32Array(256);
  const hBg = new Uint32Array(256);
  let zoneCount = 0;
  for (let p = 0; p < N; p++) {
    const on = Math.abs(img[p] - bg[p]) > FG_T ? 1 : 0;
    d[p] = on;
    if (zone[p]) {
      dz[p] = on;
      hImg[img[p]]++;
      hBg[bg[p]]++;
      zoneCount++;
    }
  }
  const g = medianFromHist(hImg, zoneCount) / Math.max(1, medianFromHist(hBg, zoneCount));
  const dg = new Uint8Array(N);
  for (let p = 0; p < N; p++) {
    if (!zone[p]) continue;
    const v = Math.min(255, Math.max(0, bg[p] * g));
    dg[p] = Math.abs(img[p] - v) > FG_T ? 1 : 0;
  }
  const fg = majority3(d, W, H);
  const fgZone = majority3(dz, W, H);
  const fgGain = majority3(dg, W, H);
  const rowFg = new Uint32Array(H);
  const rowFgGain = new Uint32Array(H);
  for (let y = 0; y < H; y++) {
    let a = 0;
    let b = 0;
    for (let x = 0; x < W; x++) {
      a += fgZone[y * W + x];
      b += fgGain[y * W + x];
    }
    rowFg[y] = a;
    rowFgGain[y] = b;
  }
  let diffZone = 0;
  if (prev) {
    let c = 0;
    for (let p = 0; p < N; p++) if (zone[p] && Math.abs(img[p] - prev[p]) > DIFF_T) c++;
    diffZone = c / zoneCount;
  }
  return { rowFg, rowFgGain, diffZone, blobs: blobsOf(fg, zone, W, H) };
}

export interface Perspective { a: number; c: number; source: string }

export const perspH = (p: Perspective, y: number) => Math.max(4, p.a * y + p.c);

export function theilSen(points: [number, number][]): [number, number] {
  const slopes: number[] = [];
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dx = points[j][0] - points[i][0];
      if (Math.abs(dx) > 10) slopes.push((points[j][1] - points[i][1]) / dx);
    }
  }
  const a = slopes.length ? medianF64(slopes) : 0.5;
  const c = medianF64(points.map(([y, h]) => h - a * y));
  return [a, c];
}

export function clampHorizon(a: number, c: number, zoneTopPx: number): [number, number] {
  if (a > 0 && -c / a > zoneTopPx - 8) c = -a * (zoneTopPx - 8);
  return [a, c];
}

export function perspectiveFromBlobs(p1: Pass1[], zoneTopPx: number): Perspective | null {
  const ys: number[] = [];
  const hs: number[] = [];
  for (const s of p1) {
    for (const b of s.blobs) {
      if (b.inZone && b.h >= 8 && b.h >= 1.3 * b.w && b.a >= 0.3 * b.h * b.w) {
        ys.push(b.footY);
        hs.push(b.h);
      }
    }
  }
  if (ys.length < 12) return null;
  const pts: [number, number][] = [];
  const ymax = Math.max(...ys);
  for (let lo = Math.floor(Math.min(...ys) / 16) * 16; lo <= ymax; lo += 16) {
    const sel: number[] = [];
    for (let i = 0; i < ys.length; i++) if (ys[i] >= lo && ys[i] < lo + 16) sel.push(hs[i]);
    if (sel.length >= 6) pts.push([lo + 8, percentileLinear(sel, 75)]);
  }
  if (pts.length < 2) return null;
  const [a, c] = clampHorizon(...theilSen(pts), zoneTopPx);
  return { a, c, source: 'blobs' };
}

/** boxes: per sample, [x1, y1, x2, y2, score] normalized. */
export function perspectiveFromBoxes(boxes: number[][][], H: number, zoneTopPx: number,
  minScore = 0.6, maxPoints = 200): Perspective | null {
  let pts: [number, number][] = [];
  for (const bs of boxes) {
    for (const [, y1, , y2, s] of bs) {
      if (s >= minScore && y2 > y1 && y2 < 0.995) pts.push([y2 * H, (y2 - y1) * H]);
    }
  }
  if (pts.length < 8) return null;
  if (pts.length > maxPoints) {
    const step = pts.length / maxPoints;
    pts = Array.from({ length: maxPoints }, (_, k) => pts[Math.floor(k * step)]);
  }
  const [a, c] = clampHorizon(...theilSen(pts), zoneTopPx);
  return { a, c, source: 'detector' };
}

export interface Scores {
  fg_zone_w: Float64Array;
  fg_zone_w_gain: Float64Array;
  pe_foot: Float64Array;
  count_foot: Float64Array;
  framediff_zone: Float64Array;
}

export function pass2(p1: Pass1[], persp: Perspective, H: number): Scores {
  const wRow = new Float64Array(H);
  for (let y = 0; y < H; y++) {
    const h = perspH(persp, y);
    wRow[y] = 1 / (0.35 * h * h);
  }
  const n = p1.length;
  const out: Scores = {
    fg_zone_w: new Float64Array(n), fg_zone_w_gain: new Float64Array(n), pe_foot: new Float64Array(n),
    count_foot: new Float64Array(n), framediff_zone: new Float64Array(n),
  };
  for (let i = 0; i < n; i++) {
    const s = p1[i];
    let a = 0;
    let b = 0;
    for (let y = 0; y < H; y++) {
      a += s.rowFg[y] * wRow[y];
      b += s.rowFgGain[y] * wRow[y];
    }
    out.fg_zone_w[i] = a;
    out.fg_zone_w_gain[i] = b;
    out.framediff_zone[i] = s.diffZone;
    let pe = 0;
    let cnt = 0;
    for (const bl of s.blobs) {
      if (!bl.inZone) continue;
      const he = perspH(persp, bl.footY);
      pe += Math.min(3, bl.a / (0.3 * he * he));
      if (bl.h < 0.5 * he || bl.a < 0.12 * he * he) continue;
      cnt += bl.w > 0.8 * he ? Math.min(4, Math.max(1, Math.floor(bl.w / (0.42 * he) + 0.5))) : 1;
    }
    out.pe_foot[i] = pe;
    out.count_foot[i] = cnt + Math.min(0.99, pe / 10);
  }
  return out;
}
