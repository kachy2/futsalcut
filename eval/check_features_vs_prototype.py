"""Check that the product module (futsalcut.features) reproduces the tournament prototype features.

python eval/check_features_vs_prototype.py --cache /data/cache/<stem>.luma640x360.u8 --roi eval/labels/<stem>.roi.json \
    --features /data/outputs/tournament/<stem>/features_dp.npz --boxes eval/reports/raw/dfine_<stem>.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from futsalcut.features import Perspective, pass2, perspective_from_blobs, perspective_from_boxes, run_pass1
from futsalcut.roi import Roi
from futsalcut.sampler import load_cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--boxes", required=True)
    args = ap.parse_args()
    cache, meta = load_cache(args.cache)
    H, W = meta["height"], meta["width"]
    roi = Roi.load(args.roi)
    zone = roi.mask(W, H, strict=True)
    zone_top = float(min(y for _, y in roi.zone(strict=True))) * H
    t0 = time.perf_counter()
    p1 = run_pass1(cache, zone)
    t1 = time.perf_counter()
    boxes = json.load(open(args.boxes))["boxes"]
    pd = perspective_from_boxes(boxes[::12], H, zone_top)
    pb = perspective_from_blobs(p1, zone_top)
    ref = dict(np.load(args.features))
    a_ref, c_ref = ref["_perspective"]
    print(f"pass1 {t1 - t0:.1f}s ({(t1 - t0) / len(cache) * 1000:.1f} ms/sample)")
    print(f"perspective detector: {pd.a:.4f}*y + {pd.c:.2f} | prototype (random-pair Theil-Sen): {a_ref:.4f}*y + {c_ref:.2f}")
    print(f"perspective blobs:    {pb.a:.4f}*y + {pb.c:.2f}")
    # same perspective as the prototype -> features must agree
    f = pass2(p1, Perspective(float(a_ref), float(c_ref)), H)
    for k in ("fg_zone_w", "fg_zone_w_gain", "pe_foot", "count_foot"):
        r = ref[k] if k != "fg_zone_w" else ref.get("fg_zone_w", None)
        if r is None:
            continue
        rel = np.abs(f[k] - r) / np.maximum(1e-9, np.abs(r))
        print(f"{k:15s} max rel diff vs prototype: {rel.max():.2e}  (samples differing > 1e-9: {(rel > 1e-9).sum()})")
    proto_diff = dict(np.load(args.features.replace("features_dp", "features")))["framediff_zone"]
    print(f"framediff_zone  max abs diff: {np.abs(f['framediff_zone'] - proto_diff).max():.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
