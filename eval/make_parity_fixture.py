"""Golden fixture for the Python <-> TypeScript parity tests (plan §3.1).

Input: a luma cache made by the sampler (synthetic video for the committed fixture; real videos only
for local, uncommitted checks). Output dir:
  luma.u8        N x H x W uint8 samples
  fixture.json   meta, zone polygon, pass-1 results, perspectives, pass-2 scores, model probabilities,
                 detector counts for synthetic boxes, and selected intervals

python eval/make_parity_fixture.py --cache /data/cache/syn_h264_gop1.luma640x360.u8 --zone "0.05,0.30 0.95,0.30 1,1 0,1" \
    --model python/src/futsalcut/models/ensemble_v1.json --out web/tests/fixtures/synthetic
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from futsalcut.ensemble import Calibrator, Ensemble
from futsalcut.features import Perspective, pass2, perspective_from_blobs, perspective_from_boxes, run_pass1
from futsalcut.roi import polygon_mask
from futsalcut.sampler import load_cache
from futsalcut.select import PostParams, select


def load_model(path: str, name: str) -> Ensemble:
    m = json.load(open(path))[name]
    cal = {k: Calibrator(**v) for k, v in m["calibrators"].items()}
    return Ensemble(names=m["names"], calibrators=cal, weights=m["weights"], bias=m["bias"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--zone", default=None, help='"x,y x,y ..." normalized polygon')
    ap.add_argument("--roi", default=None, help="ROI json (strict zone) instead of --zone")
    ap.add_argument("--boxes", default=None, help="real detector boxes json instead of synthetic boxes")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache, meta = load_cache(args.cache)
    H, W = meta["height"], meta["width"]
    if args.roi:
        from futsalcut.roi import Roi
        zone_poly = [tuple(p) for p in Roi.load(args.roi).zone(strict=True)]
    else:
        zone_poly = [tuple(float(v) for v in p.split(",")) for p in args.zone.split()]
    zone = polygon_mask(zone_poly, W, H)
    zone_top = min(y for _, y in zone_poly) * H
    shutil.copyfile(args.cache, os.path.join(args.out, "luma.u8"))
    p1 = run_pass1(cache, zone)
    pb = perspective_from_blobs(p1, zone_top)
    # deterministic synthetic detector boxes (people-shaped, growing with y) for the detector paths
    rng = np.random.default_rng(3)
    boxes = []
    if args.boxes:
        boxes = json.load(open(args.boxes))["boxes"]
    else:
        for i in range(len(cache)):
            bs = []
            for _ in range(int(rng.integers(0, 4))):
                y2 = float(rng.uniform(0.35, 0.98))
                hgt = 0.05 + 0.25 * y2
                x1 = float(rng.uniform(0.0, 0.85))
                bs.append([x1, y2 - hgt, x1 + 0.35 * hgt, y2, float(rng.uniform(0.2, 0.99))])
            boxes.append(bs)
    pd = perspective_from_boxes(boxes, H, zone_top)
    fixed = Perspective(0.6, 5.0, "fixed")
    scores = pass2(p1, fixed, H)
    from futsalcut.roi import point_in_polygon
    det = np.zeros(len(cache))
    for i, bs in enumerate(boxes):
        inz = [b for b in bs if point_in_polygon(zone_poly, (b[0] + b[2]) / 2, min(0.9999, b[3]))]
        det[i] = sum(1 for b in inz if b[4] >= 0.4) + min(0.99, sum(b[4] for b in inz) / 10)
    light = load_model(args.model, "light").prob(scores)
    full_scores = dict(scores) | {"dfine_count": det}
    full = load_model(args.model, "full").prob(full_scores)
    model = json.load(open(args.model))
    post = PostParams(**model["post"])
    t = np.asarray(meta["t"], float)
    fx = {
        "meta": {"n": len(cache), "W": W, "H": H, "t": meta["t"]},
        "zone": zone_poly,
        "zone_pixels": int(zone.sum()),
        "pass1": [{"rowFg": s.row_fg.tolist(), "rowFgGain": s.row_fg_gain.tolist(), "diffZone": s.diff_zone,
                   "blobs": [[b.x, b.y, b.w, b.h, b.a, b.foot_y, int(b.in_zone)] for b in s.blobs]} for s in p1],
        "perspective_blobs": [pb.a, pb.c] if pb else None,
        "boxes": boxes,
        "perspective_boxes": [pd.a, pd.c] if pd else None,
        "fixed_perspective": [fixed.a, fixed.c],
        "scores": {k: v.tolist() for k, v in scores.items()},
        "detector": det.tolist(),
        "light": light.tolist(),
        "full": full.tolist(),
        "threshold": model["threshold"],
        "intervals_light": select(light, t, model["threshold"], post),
    }
    with open(os.path.join(args.out, "fixture.json"), "w") as f:
        json.dump(fx, f)
    print(f"fixture: {len(cache)} samples {W}x{H}, zone px {int(zone.sum())}, blobs/sample avg "
          f"{np.mean([len(s.blobs) for s in p1]):.1f}, persp blobs {pb}, persp boxes {pd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
