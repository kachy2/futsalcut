"""Exploration: foreground blobs (median background) in the counting zone vs GT labels, and a
perspective fit h(y) from blob heights. Prototype only (not the parity-safe implementation).

python eval/explore_blobs.py /data/cache/<stem>.luma640x360.u8 eval/labels/<stem>.roi.json eval/labels/<stem>.labels.json
"""
from __future__ import annotations

import collections
import json
import sys

import cv2
import numpy as np

from futsalcut.roi import Roi
from futsalcut.sampler import load_cache


def median_background(cache: np.ndarray, i: int, half: int = 60, step: int = 4) -> np.ndarray:
    n = len(cache)
    idx = list(range(max(0, i - half), min(n, i + half + 1), step))
    stack = np.asarray(cache[idx])
    k = (len(idx) - 1) // 2
    return np.partition(stack, k, axis=0)[k]


def main() -> int:
    cache_path, roi_path, labels_path = sys.argv[1:4]
    cache, meta = load_cache(cache_path)
    H, W = meta["height"], meta["width"]
    roi = Roi.load(roi_path)
    zone = roi.mask(W, H, strict=True)
    labels = json.load(open(labels_path, encoding="utf-8"))["labels"]
    print("zone pixels:", int(zone.sum()), "of", W * H)
    bg_cache: dict[int, np.ndarray] = {}
    rows = []
    for i in range(len(cache)):
        key = i // 10
        if key not in bg_cache:
            bg_cache = {key: median_background(cache, key * 10 + 5)}
        bg = bg_cache[key]
        img = cache[i].astype(np.int16)
        d = np.abs(img - bg.astype(np.int16))
        fg = ((d > 22) & zone).astype(np.uint8)
        # 3x3 majority filter (zero padding)
        s = cv2.boxFilter(fg, cv2.CV_16U, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        fg = (s >= 5).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
        blobs = []
        for j in range(1, n):
            x, y, w, h, a = stats[j]
            if a < 8:
                continue
            blobs.append((int(y + h), int(h), int(w), int(a)))
        rows.append((i, labels[i], blobs, float(fg.sum()) / zone.sum()))
    # perspective: heights of reasonably isolated blobs by foot y (tallest per 20px band)
    by_band = collections.defaultdict(list)
    for i, lb, blobs, _ in rows:
        for fy, h, w, a in blobs:
            if h >= 6 and a >= 0.25 * h * w and w <= 0.8 * h:
                by_band[fy // 20 * 20].append(h)
    print("foot_y band: n, p50, p75, p90 of blob height")
    pts = []
    for b in sorted(by_band):
        hs = np.array(by_band[b])
        if len(hs) >= 5:
            p75 = np.percentile(hs, 75)
            pts.append((b + 10, p75))
            print(f"  {b:3d}-{b + 19:3d}: n={len(hs):4d} p50={np.median(hs):5.1f} p75={p75:5.1f} p90={np.percentile(hs, 90):5.1f}")
    if len(pts) >= 2:
        ys, hs = np.array(pts).T
        a, c = np.polyfit(ys, hs, 1)
        print(f"fit p75: h(y) = {a:.3f} * y + {c:.1f}  -> horizon y_h = {-c / a:.1f}")
    # label vs largest blob / fg ratio summary
    print("label: n, median fg_ratio, share with >=2 blobs of area>=60")
    for lb in "2M b?":
        sel = [r for r in rows if r[1] == lb]
        if not sel:
            continue
        fgr = np.median([r[3] for r in sel])
        two = np.mean([sum(1 for b in r[2] if b[3] >= 60) >= 2 for r in sel])
        print(f"  {lb!r}: n={len(sel):3d} fg={fgr:.4f} >=2 big blobs={two:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
