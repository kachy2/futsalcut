"""From per-sample scores to kept intervals (shared by all scorers; parity contract with the web port).

Sample i covers [t_i, t_{i+1}) (the last one covers [t_n-1, t_n-1 + 1)). Steps:
  1. decide:   d_i = score_i >= threshold
  2. smooth:   k-of-n majority in a centred window (default 2 of 3, window clipped at the edges)
  3. runs:     maximal runs of kept samples -> [start, end) seconds
  4. merge:    join runs separated by gaps <= merge_gap
  5. drop:     remove runs shorter than min_len
  6. pad:      extend by pad_before / pad_after, clamp to [0, duration], merge overlaps
  7. snap:     start -> latest sample time <= start + SNAP_TOL, end -> earliest sample end >= end - SNAP_TOL
               (samples are keyframes, so the kept duration equals the lossless output duration)
Budget mode finds the threshold whose kept duration is closest to (not above, if possible) a target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PostParams:
    smooth_window: int = 3
    smooth_need: int = 2
    merge_gap: float = 5.0
    min_len: float = 4.0
    pad_before: float = 2.0
    pad_after: float = 2.0


def sample_ends(t: np.ndarray) -> np.ndarray:
    end = np.empty_like(t)
    end[:-1] = t[1:]
    end[-1] = t[-1] + 1.0
    return end


def smooth(dec: np.ndarray, window: int = 3, need: int = 2) -> np.ndarray:
    if window <= 1:
        return dec.astype(bool)
    n = len(dec)
    half = window // 2
    c = np.concatenate([[0], np.cumsum(dec.astype(np.int32))])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    counts = c[hi] - c[lo]
    need_eff = np.minimum(need, hi - lo)  # edges: window shrinks, so does the requirement
    return counts >= need_eff


def runs_to_intervals(mask: np.ndarray, t: np.ndarray) -> list[tuple[float, float]]:
    ends = sample_ends(t)
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((float(t[i]), float(ends[j])))
            i = j + 1
        else:
            i += 1
    return out


def postprocess(intervals: list[tuple[float, float]], duration: float, p: PostParams) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for s, e in intervals:
        if merged and s - merged[-1][1] <= p.merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    kept = [(s, e) for s, e in merged if e - s >= p.min_len]
    padded: list[list[float]] = []
    for s, e in kept:
        s2, e2 = max(0.0, s - p.pad_before), min(duration, e + p.pad_after)
        if padded and s2 <= padded[-1][1]:
            padded[-1][1] = max(padded[-1][1], e2)
        else:
            padded.append([s2, e2])
    return [(s, e) for s, e in padded]


SNAP_TOL = 0.05  # seconds; keyframe timestamps jitter by a few ms around the nominal grid


def snap_to_samples(intervals: list[tuple[float, float]], t: np.ndarray) -> list[tuple[float, float]]:
    ends = sample_ends(t)
    out: list[list[float]] = []
    for s, e in intervals:
        i = int(np.searchsorted(t, s + SNAP_TOL, side="right")) - 1
        j = int(np.searchsorted(ends, e - SNAP_TOL, side="left"))
        s2 = float(t[max(0, i)])
        e2 = float(ends[min(len(ends) - 1, j)])
        if out and s2 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e2)
        else:
            out.append([s2, e2])
    return [(s, e) for s, e in out]


def select(score: np.ndarray, t: np.ndarray, threshold: float, p: PostParams = PostParams(),
           duration: float | None = None) -> list[tuple[float, float]]:
    dur = float(sample_ends(t)[-1]) if duration is None else duration
    dec = smooth(score >= threshold, p.smooth_window, p.smooth_need)
    return snap_to_samples(postprocess(runs_to_intervals(dec, t), dur, p), t)


def kept_seconds(intervals: list[tuple[float, float]]) -> float:
    return float(sum(e - s for s, e in intervals))


def threshold_for_budget(score: np.ndarray, t: np.ndarray, target_s: float, p: PostParams = PostParams(),
                         duration: float | None = None) -> tuple[float, list[tuple[float, float]]]:
    """Lowest candidate threshold whose kept duration does not exceed target_s (kept duration is
    non-increasing in the threshold). Falls back to the highest threshold if none fits."""
    cands = np.unique(score)
    lo, hi = 0, len(cands) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        iv = select(score, t, float(cands[mid]), p, duration)
        if kept_seconds(iv) <= target_s:
            best = (float(cands[mid]), iv)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        thr = float(np.nextafter(cands[-1], np.inf))
        best = (thr, select(score, t, thr, p, duration))
    return best
