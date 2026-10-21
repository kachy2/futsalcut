"""Light scorers (plan §3.3/§3.4, docs/algorithms.md §5) in a streaming-friendly two-pass form.

Pass 1 (per sample, needs only a sliding window of samples):
  - background: lower median of samples [c-60, c+60] step 4 around the centre c of each 10-sample block
  - fg      = majority3(|I - B| > FG_T)                        (full frame, for blobs)
  - fg_zone = majority3((|I - B| > FG_T) & zone)               -> per-row counts
  - fg_gain = majority3((|I - clip(B * g)| > FG_T) & zone)     -> per-row counts, g = median(I[zone]) / max(1, median(B[zone]))
  - diff    = |I - I_prev| > DIFF_T, fraction over the zone    (no filtering)
  - blobs   = 8-connected components of fg with area >= 6: x, y, w, h, area, foot (bottom centre), in_zone
Pass 2 (after the perspective model h(y) = max(4, a*y + c) is known):
  - fg_zone_w_gain = sum_y row_fg_gain[y] / (0.35 h(y)^2)       ("person-equivalents" present)
  - pe_foot        = sum over in-zone blobs of min(3, area / (0.3 h(foot)^2))
  - count_foot     = number of in-zone blobs passing the size test (wide blobs count as several) + min(0.99, pe/10)
  - framediff_zone = diff fraction
Every operation is integer or float64 arithmetic with a fixed order, mirrored by web/src/core/features.ts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

FG_T = 22
DIFF_T = 25
BG_HALF, BG_STEP, BG_EVERY = 60, 4, 10
MIN_BLOB_AREA = 6


def majority3(mask: np.ndarray) -> np.ndarray:
    """Pixel stays on when >= 5 of its 3x3 neighbourhood are on; outside the image counts as off."""
    s = cv2.boxFilter(mask.astype(np.uint8), cv2.CV_16U, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    return s >= 5


def block_window(n: int, block: int) -> list[int]:
    c = min(n - 1, block * BG_EVERY + BG_EVERY // 2)
    return list(range(max(0, c - BG_HALF), min(n, c + BG_HALF + 1), BG_STEP))


def lower_median_stack(stack: np.ndarray) -> np.ndarray:
    k = (len(stack) - 1) // 2
    return np.partition(stack, k, axis=0)[k]


def median_f64(v: np.ndarray) -> float:
    """numpy median: mean of the two middle values for even counts."""
    return float(np.median(v.astype(np.float64)))


@dataclass
class Blob:
    x: int
    y: int
    w: int
    h: int
    a: int
    foot_y: int
    in_zone: bool


@dataclass
class Pass1:
    row_fg: np.ndarray       # uint32[H] zone-clipped foreground per row
    row_fg_gain: np.ndarray  # uint32[H]
    diff_zone: float
    blobs: list[Blob] = field(default_factory=list)


def blobs_of(fg: np.ndarray, zone: np.ndarray) -> list[Blob]:
    H, W = fg.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    out = []
    for j in range(1, n):
        x, y, w, h, a = (int(v) for v in st[j])
        if a < MIN_BLOB_AREA:
            continue
        fy, fx = min(H - 1, y + h - 1), min(W - 1, x + w // 2)
        out.append(Blob(x, y, w, h, a, fy, bool(zone[fy, fx])))
    out.sort(key=lambda b: (b.y, b.x, b.h, b.w))  # canonical order (sums are order-sensitive)
    return out


def pass1_sample(img: np.ndarray, bg: np.ndarray, prev: np.ndarray | None, zone: np.ndarray) -> Pass1:
    i16 = img.astype(np.int16)
    d = np.abs(i16 - bg.astype(np.int16)) > FG_T
    fg = majority3(d)
    fg_zone = majority3(d & zone)
    g = median_f64(img[zone]) / max(1.0, median_f64(bg[zone]))
    dg = np.abs(i16.astype(np.float64) - np.clip(bg.astype(np.float64) * g, 0.0, 255.0)) > FG_T
    fg_gain = majority3(dg & zone)
    if prev is None:
        diff = 0.0
    else:
        dd = np.abs(i16 - prev.astype(np.int16)) > DIFF_T
        diff = float(dd[zone].sum()) / float(zone.sum())
    return Pass1(row_fg=fg_zone.sum(axis=1).astype(np.uint32), row_fg_gain=fg_gain.sum(axis=1).astype(np.uint32),
                 diff_zone=diff, blobs=blobs_of(fg, zone))


def run_pass1(cache: np.ndarray, zone: np.ndarray) -> list[Pass1]:
    """Reference implementation over a whole cache (the web app streams the same computation)."""
    n = len(cache)
    out: list[Pass1] = []
    bg_block, bg = -1, None
    for i in range(n):
        b = i // BG_EVERY
        if b != bg_block:
            bg = lower_median_stack(np.asarray(cache[block_window(n, b)]))
            bg_block = b
        out.append(pass1_sample(np.asarray(cache[i]), bg, np.asarray(cache[i - 1]) if i else None, zone))
    return out


@dataclass
class Perspective:
    a: float
    c: float
    source: str = ""

    def h(self, y):
        return np.maximum(4.0, self.a * np.asarray(y, np.float64) + self.c)


def theil_sen(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Median of all pairwise slopes (pairs with |dy| > 10 px), then median intercept."""
    slopes = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[j][0] - points[i][0]
            if abs(dx) > 10:
                slopes.append((points[j][1] - points[i][1]) / dx)
    a = float(np.median(slopes)) if slopes else 0.5
    c = float(np.median([hv - a * yv for yv, hv in points]))
    return a, c


def clamp_horizon(a: float, c: float, zone_top_px: float) -> tuple[float, float]:
    if a > 0 and -c / a > zone_top_px - 8:  # people at the top of the zone must still be > 0 px tall
        c = -a * (zone_top_px - 8)
    return a, c


def perspective_from_blobs(p1: list[Pass1], zone_top_px: float) -> Perspective | None:
    ys, hs = [], []
    for s in p1:
        for b in s.blobs:
            if b.in_zone and b.h >= 8 and b.h >= 1.3 * b.w and b.a >= 0.3 * b.h * b.w:
                ys.append(b.foot_y)
                hs.append(b.h)
    if len(ys) < 12:
        return None
    ys_a, hs_a = np.array(ys, np.float64), np.array(hs, np.float64)
    pts = []
    lo = float(np.floor(ys_a.min() / 16) * 16)
    while lo <= ys_a.max():
        sel = (ys_a >= lo) & (ys_a < lo + 16)
        if sel.sum() >= 6:
            pts.append((lo + 8, float(np.percentile(hs_a[sel], 75))))
        lo += 16
    if len(pts) < 2:
        return None
    a, c = clamp_horizon(*theil_sen(pts), zone_top_px)
    return Perspective(a, c, "blobs")


def perspective_from_boxes(boxes_per_sample: list[list[list[float]]], H: int, zone_top_px: float,
                           min_score: float = 0.6, max_points: int = 200) -> Perspective | None:
    """Detector boxes (normalized x1,y1,x2,y2,score) -> (foot_y, height) points in cache pixels."""
    pts = []
    for boxes in boxes_per_sample:
        for x1, y1, x2, y2, s in boxes:
            if s >= min_score and y2 > y1 and y2 < 0.995:
                pts.append((y2 * H, (y2 - y1) * H))
    if len(pts) < 8:
        return None
    if len(pts) > max_points:  # deterministic thinning
        step = len(pts) / max_points
        pts = [pts[int(k * step)] for k in range(max_points)]
    a, c = clamp_horizon(*theil_sen(pts), zone_top_px)
    return Perspective(a, c, "detector")


def pass2(p1: list[Pass1], persp: Perspective, H: int) -> dict[str, np.ndarray]:
    hy = persp.h(np.arange(H))
    w_row = 1.0 / (0.35 * hy * hy)
    n = len(p1)
    out = {k: np.zeros(n) for k in ("fg_zone_w", "fg_zone_w_gain", "pe_foot", "count_foot", "framediff_zone")}
    for i, s in enumerate(p1):
        # sequential sums (np.cumsum is a left-to-right prefix sum) to match the JS loop bit for bit
        out["fg_zone_w"][i] = float(np.cumsum(s.row_fg.astype(np.float64) * w_row)[-1])
        out["fg_zone_w_gain"][i] = float(np.cumsum(s.row_fg_gain.astype(np.float64) * w_row)[-1])
        out["framediff_zone"][i] = s.diff_zone
        pe, cnt = 0.0, 0
        for b in s.blobs:
            if not b.in_zone:
                continue
            he = float(persp.h(b.foot_y))
            pe += min(3.0, b.a / (0.3 * he * he))
            if b.h < 0.5 * he or b.a < 0.12 * he * he:
                continue
            cnt += min(4, max(1, int(np.floor(b.w / (0.42 * he) + 0.5)))) if b.w > 0.8 * he else 1
        out["pe_foot"][i] = pe
        out["count_foot"][i] = cnt + min(0.99, pe / 10)
    return out
