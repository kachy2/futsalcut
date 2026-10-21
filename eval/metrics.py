"""Evaluation metrics against per-GOP ground truth labels (docs/labeling.md).

Labels per sample: '2', 'M', '1', 'F', '0', 'b', '?'.  Binary views:
  strict: keep = '2'           half: keep = '2' or 'M'
  '?' is ignored in both; 'b' (band count uncertain) is a drop in strict and ignored in half.
Predictions are either a per-sample keep mask (classifier view) or kept intervals in seconds
(final edit view, after padding/merging).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

KEEP = {"strict": {"2"}, "half": {"2", "M"}}
IGNORE = {"strict": {"?"}, "half": {"?", "b"}}


def binary_gt(labels: list[str], mode: str = "strict") -> np.ndarray:
    """1 = keep, 0 = drop, -1 = ignore."""
    keep, ignore = KEEP[mode], IGNORE[mode]
    return np.array([-1 if lb in ignore else (1 if lb in keep else 0) for lb in labels], np.int8)


def load_labels(path: str) -> tuple[list[str], list[float]]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return list(d["labels"]), list(d["t"])


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _prf(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return PRF(round(p, 4), round(r, 4), round(f, 4), tp, fp, fn)


def classifier_prf(gt: np.ndarray, pred: np.ndarray) -> PRF:
    m = gt >= 0
    g, p = gt[m] == 1, pred[m].astype(bool)
    return _prf(int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum()))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is true."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def intervals_to_mask(intervals: list[tuple[float, float]], t: np.ndarray) -> np.ndarray:
    """Sample i (covering [t_i, t_i + 1)) is kept when its midpoint falls in a kept interval."""
    mid = t + 0.5
    m = np.zeros(len(t), bool)
    for s, e in intervals:
        m |= (mid >= s) & (mid < e)
    return m


@dataclass
class EditMetrics:
    prf_band: PRF            # precision ignores kept samples inside the padding band around GT
    event_recall: float      # GT episodes covered >= 50%
    episodes: int
    event_recall_3s: float   # same, only episodes of >= 3 samples (shorter ones fall under min scene length)
    episodes_3s: int
    false_scene_rate: float  # kept scenes overlapping GT (incl. band) < 20%
    scenes: int
    scenes_per_episode: float
    median_start_err_s: float | None
    median_end_err_s: float | None
    kept_ratio: float
    gt_ratio: float


def edit_metrics(gt: np.ndarray, intervals: list[tuple[float, float]], t: np.ndarray,
                 pad_before: float = 2.0, pad_after: float = 2.0) -> EditMetrics:
    kept = intervals_to_mask(intervals, t)
    valid = gt >= 0
    pos = gt == 1
    band = np.zeros_like(pos)
    for s, e in runs(pos):
        lo = np.searchsorted(t, t[s] - pad_before - 1e-6)
        hi = np.searchsorted(t, t[e - 1] + pad_after + 1e-6, side="right")
        band[lo:hi] = True
    band &= ~pos
    tp = int((kept & pos).sum())
    fp = int((kept & ~pos & ~band & valid).sum())
    fn = int((~kept & pos).sum())
    eps = runs(pos)
    covered = [kept[s:e].mean() >= 0.5 for s, e in eps]
    covered3 = [c for (s, e), c in zip(eps, covered) if e - s >= 3]
    scenes = runs(kept)
    pos_or_band = pos | band
    false = [pos_or_band[s:e].mean() < 0.2 for s, e in scenes]
    s_err, e_err = [], []
    for s, e in eps:
        hits = [(a, b) for a, b in scenes if a < e and b > s]
        if hits:
            s_err.append(abs(t[hits[0][0]] - t[s]))
            e_err.append(abs(t[hits[-1][1] - 1] - t[e - 1]))
    return EditMetrics(
        prf_band=_prf(tp, fp, fn),
        event_recall=round(float(np.mean(covered)) if covered else 0.0, 4),
        episodes=len(eps),
        event_recall_3s=round(float(np.mean(covered3)) if covered3 else 0.0, 4),
        episodes_3s=len(covered3),
        false_scene_rate=round(float(np.mean(false)) if false else 0.0, 4),
        scenes=len(scenes),
        scenes_per_episode=round(len(scenes) / len(eps), 3) if eps else 0.0,
        median_start_err_s=round(float(np.median(s_err)), 2) if s_err else None,
        median_end_err_s=round(float(np.median(e_err)), 2) if e_err else None,
        kept_ratio=round(float(kept.mean()), 4),
        gt_ratio=round(float(pos[valid].mean()) if valid.any() else 0.0, 4),
    )


def block_bootstrap(gt: np.ndarray, pred: np.ndarray, metric=lambda g, p: classifier_prf(g, p).f1,
                    block: int = 30, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of a per-sample metric by resampling contiguous blocks (keeps temporal correlation)."""
    rng = np.random.default_rng(seed)
    nb = max(1, len(gt) // block)
    starts = np.arange(nb) * block
    vals = []
    for _ in range(n):
        pick = rng.choice(starts, size=nb, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, len(gt))) for s in pick])
        vals.append(metric(gt[idx], pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def report(labels: list[str], t: list[float], pred_mask: np.ndarray,
           intervals: list[tuple[float, float]] | None = None, pad: tuple[float, float] = (2.0, 2.0),
           split_s: float | None = None) -> dict:
    """Full report for both GT modes; optional train/test split by time (test = t >= split_s)."""
    ta = np.asarray(t, float)
    out: dict = {}
    for mode in ("strict", "half"):
        g = binary_gt(labels, mode)
        sel = np.ones(len(g), bool) if split_s is None else ta >= split_s
        res = {"classifier": asdict(classifier_prf(g[sel], pred_mask[sel])),
               "classifier_f1_ci95": block_bootstrap(g[sel], pred_mask[sel])}
        if intervals is not None:
            res["edit"] = asdict(edit_metrics(g[sel], intervals, ta[sel], *pad))
        out[mode] = res
    return out
