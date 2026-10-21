/**
 * Multi-algorithm layer — twin of python/src/futsalcut/ensemble.py (inference side only; the
 * parameters are fitted offline in Python and shipped as JSON).
 */
export interface CalibratorParams { scale: number; a: number; b: number }

export interface EnsembleModel {
  names: string[];
  calibrators: Record<string, CalibratorParams>;
  weights: number[];
  bias: number;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function transform(c: CalibratorParams, s: number): number {
  return Math.log1p(Math.max(s, 0) / c.scale);
}

export function calibrate(c: CalibratorParams, s: number): number {
  return sigmoid(c.a * transform(c, s) + c.b);
}

export function ensembleProb(m: EnsembleModel, scores: Record<string, ArrayLike<number>>): Float64Array {
  const n = scores[m.names[0]].length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let z = m.bias;
    for (let k = 0; k < m.names.length; k++) {
      const name = m.names[k];
      z += m.weights[k] * transform(m.calibrators[name], scores[name][i]);
    }
    out[i] = sigmoid(z);
  }
  return out;
}

/** Samples where calibrated scorers disagree most, at least minGapS apart (one question per scene). */
export function pickQuestions(probs: Record<string, ArrayLike<number>>, t: ArrayLike<number>, k = 10,
  minGapS = 20, exclude?: ArrayLike<boolean>): number[] {
  const names = Object.keys(probs);
  const n = t.length;
  const spread = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if (exclude?.[i]) { spread[i] = -Infinity; continue; }
    let mean = 0;
    for (const nm of names) mean += probs[nm][i];
    mean /= names.length;
    let v = 0;
    for (const nm of names) v += (probs[nm][i] - mean) ** 2;
    spread[i] = Math.sqrt(v / names.length) + 0.5 * (0.5 - Math.abs(mean - 0.5));
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => spread[b] - spread[a] || a - b);
  const chosen: number[] = [];
  for (const i of order) {
    if (!Number.isFinite(spread[i])) break;
    if (chosen.every((j) => Math.abs(t[i] - t[j]) >= minGapS)) chosen.push(i);
    if (chosen.length === k) break;
  }
  return chosen.sort((a, b) => a - b);
}

function bestThreshold(p: number[], ans: boolean[], prior: number): [number, number] {
  const vals = Array.from(new Set(p)).sort((a, b) => a - b);
  const cands = [prior];
  for (let i = 0; i + 1 < vals.length; i++) cands.push((vals[i] + vals[i + 1]) / 2);
  let best: [number, number, number] | null = null; // agree, -distance, threshold
  for (const c of cands) {
    let agree = 0;
    for (let i = 0; i < p.length; i++) if ((p[i] >= c) === ans[i]) agree++;
    const key: [number, number, number] = [agree, -Math.abs(c - prior), c];
    if (!best || key[0] > best[0] || (key[0] === best[0] && key[1] > best[1])) best = key;
  }
  return [best![0], best![2]];
}

/** Tune the default scorer's threshold with the user's answers; switch scorer only with a clear margin. */
export function quickCalibrate(probs: Record<string, ArrayLike<number>>, answers: Map<number, boolean>,
  defaultName = 'ensemble', switchMargin = 2, prior = 0.5): { name: string; threshold: number } {
  const idx = [...answers.keys()].sort((a, b) => a - b);
  const ans = idx.map((i) => answers.get(i)!);
  const pick = (name: string) => bestThreshold(idx.map((i) => probs[name][i]), ans, prior);
  const [dAgree, dThr] = pick(defaultName);
  let best = { name: defaultName, threshold: dThr, agree: dAgree };
  for (const name of Object.keys(probs)) {
    if (name === defaultName) continue;
    const [agree, thr] = pick(name);
    if (agree >= dAgree + switchMargin && agree > best.agree) best = { name, threshold: thr, agree };
  }
  return { name: best.name, threshold: best.threshold };
}
