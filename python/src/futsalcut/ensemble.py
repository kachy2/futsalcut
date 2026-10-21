"""Multi-algorithm layer (plan §3.4): per-scorer calibration, weighted ensemble, and quick calibration.

Everything here is small, deterministic arithmetic so the web port can reproduce it exactly:
- Transform: u = log1p(max(score, 0) / scale) with a fixed per-scorer scale (scores are non-negative
  amounts: fractions, person counts, person-equivalents).
- Calibrator (Platt): P(keep | score) = sigmoid(a * u + b), 2 parameters per scorer, so the
  sensitivity slider means the same thing for every algorithm.
- Ensemble: logistic regression on the transformed scores, fitted by Newton/IRLS with L2.
  (Isotonic calibration fitted on the same data as the combiner over-fits: a pure-noise feature got
  weight; parametric calibration avoids that.)
- Quick calibration: ask the user about the samples where the scorers disagree most (spread in time),
  then pick the scorer (or the ensemble) and threshold that agree best with the answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logreg(X: np.ndarray, y: np.ndarray, l2: float, iters: int = 100) -> np.ndarray:
    """Weights (last = bias) of an L2-regularized logistic regression (bias unpenalized)."""
    X1 = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X1.shape[1])
    reg = np.full(X1.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = _sigmoid(X1 @ w)
        g = X1.T @ (p - y) + reg * w
        Hm = (X1 * (p * (1 - p) + 1e-9)[:, None]).T @ X1 + np.diag(reg)
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


@dataclass
class Calibrator:
    scale: float
    a: float
    b: float

    def transform(self, s: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(np.asarray(s, float), 0.0) / self.scale)

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return _sigmoid(self.a * self.transform(s) + self.b)

    def threshold_for(self, p: float) -> float:
        """Raw score at which the calibrated probability equals p."""
        u = (np.log(p / (1 - p)) - self.b) / self.a
        return float(np.expm1(u) * self.scale)

    @classmethod
    def fit(cls, score: np.ndarray, label: np.ndarray, l2: float = 1e-3) -> "Calibrator":
        s = np.maximum(np.asarray(score, float), 0.0)
        pos = s[s > 0]
        scale = float(np.median(pos)) if len(pos) else 1.0
        u = np.log1p(s / scale)[:, None]
        w = _logreg(u, np.asarray(label, float), l2)
        return cls(scale=scale, a=float(w[0]), b=float(w[1]))


@dataclass
class Ensemble:
    names: list[str]
    calibrators: dict[str, Calibrator]
    weights: list[float]
    bias: float
    meta: dict = field(default_factory=dict)

    def features(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([self.calibrators[k].transform(scores[k]) for k in self.names], axis=1)

    def prob(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        return _sigmoid(self.features(scores) @ np.asarray(self.weights) + self.bias)

    @classmethod
    def fit(cls, scores: dict[str, np.ndarray], label: np.ndarray, names: list[str], l2: float = 1.0) -> "Ensemble":
        y = np.asarray(label, float)
        cal = {k: Calibrator.fit(scores[k], y) for k in names}
        X = np.stack([cal[k].transform(scores[k]) for k in names], axis=1)
        w = _logreg(X, y, l2)
        return cls(names=list(names), calibrators=cal, weights=[float(v) for v in w[:-1]], bias=float(w[-1]))


def pick_questions(probs: dict[str, np.ndarray], t: np.ndarray, k: int = 10, min_gap_s: float = 20.0,
                   exclude: np.ndarray | None = None) -> list[int]:
    """Samples where the calibrated scorers disagree most (std of probabilities, then closeness to 0.5
    of their mean), at least `min_gap_s` apart so each question is a different scene."""
    P = np.stack(list(probs.values()), axis=1)
    spread = P.std(axis=1) + 0.5 * (0.5 - np.abs(P.mean(axis=1) - 0.5))
    if exclude is not None:
        spread = np.where(exclude, -np.inf, spread)
    chosen: list[int] = []
    for i in np.argsort(-spread, kind="stable"):
        if not np.isfinite(spread[i]):
            break
        if all(abs(t[i] - t[j]) >= min_gap_s for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            break
    return sorted(chosen)


def _best_threshold(p: np.ndarray, ans: np.ndarray, prior: float) -> tuple[int, float]:
    """(agreements, threshold) maximizing agreement; ties -> closest to `prior`. Thresholds are
    midpoints between answered probabilities, so they generalize instead of sitting on one sample."""
    vals = np.unique(p)
    cands = np.concatenate([[prior], (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else []])
    best = None
    for c in cands:
        agree = int(((p >= c) == ans).sum())
        key = (agree, -abs(c - prior))
        if best is None or key > best[0]:
            best = (key, float(c))
    return best[0][0], best[1]


def quick_calibrate(probs: dict[str, np.ndarray], answers: dict[int, bool], default: str = "ensemble",
                    switch_margin: int = 2, prior: float = 0.5) -> tuple[str, float]:
    """Use the user's answers to tune the default scorer's threshold; switch to another scorer only if
    it agrees with at least `switch_margin` more answers (a few answers are weak evidence)."""
    idx = np.array(sorted(answers))
    ans = np.array([answers[i] for i in idx])
    d_agree, d_thr = _best_threshold(probs[default][idx], ans, prior)
    best = (default, d_thr, d_agree)
    for name, p in probs.items():
        if name == default:
            continue
        agree, thr = _best_threshold(p[idx], ans, prior)
        if agree >= d_agree + switch_margin and agree > best[2]:
            best = (name, thr, agree)
    return best[0], best[1]
