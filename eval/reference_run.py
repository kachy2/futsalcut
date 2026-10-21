"""Python reference of the web app's auto mode on one recording (for browser-vs-Python comparison).

Same steps as web/src/worker/analyze.ts: pass 1 -> perspective from detector boxes on every 12th sample
(fallback: blobs) -> pass 2 -> light ensemble -> detector score where the light model is unsure -> select.
Detector boxes come from eval/reports/raw/dfine_<stem>.json (thumbnail-based, so the cascade may differ
slightly from the browser, which feeds the detector from the full-resolution frame).

python eval/reference_run.py --stem <stem> [--zone strict]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from futsalcut.ensemble import Calibrator, Ensemble
from futsalcut.features import pass2, perspective_from_blobs, perspective_from_boxes, run_pass1
from futsalcut.roi import Roi, point_in_polygon
from futsalcut.sampler import load_cache
from futsalcut.select import PostParams, kept_seconds, select


def load(model: dict, name: str) -> Ensemble:
    m = model[name]
    return Ensemble(m["names"], {k: Calibrator(**v) for k, v in m["calibrators"].items()}, m["weights"], m["bias"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--model", default="python/src/futsalcut/models/ensemble_v1.json")
    ap.add_argument("--pad-before", type=float, default=None, help="override the model's lead-in (s)")
    ap.add_argument("--pad-after", type=float, default=None, help="override the model's tail (s)")
    args = ap.parse_args()
    model = json.load(open(args.model))
    cache, meta = load_cache(f"/data/cache/{args.stem}.luma640x360.u8")
    H, W = meta["height"], meta["width"]
    zone_poly = Roi.load(f"eval/labels/{args.stem}.roi.json").zone(strict=True)
    from futsalcut.roi import polygon_mask
    zone = polygon_mask(zone_poly, W, H)
    zone_top = min(y for _, y in zone_poly) * H
    boxes = json.load(open(f"eval/reports/raw/dfine_{args.stem}.json"))["boxes"]
    stride = model["detector"]["perspective_stride"]
    p1 = run_pass1(cache, zone)
    persp = perspective_from_boxes([boxes[i] for i in range(0, len(boxes), stride)], H, zone_top) or perspective_from_blobs(p1, zone_top)
    scores = pass2(p1, persp, H)
    light = load(model, "light").prob(scores)
    lo, hi = model["detector"]["cascade"]

    def det(i: int) -> float:
        inz = [b for b in boxes[i] if point_in_polygon(zone_poly, (b[0] + b[2]) / 2, min(0.9999, b[3]))]
        return sum(1 for b in inz if b[4] >= model["detector"]["min_score"]) + min(0.99, sum(b[4] for b in inz) / 10)

    unsure = [i for i in range(len(light)) if lo < light[i] < hi]
    prob = light.copy()
    if unsure:
        sub = {k: np.asarray(v)[unsure] for k, v in scores.items()} | {"dfine_count": np.array([det(i) for i in unsure])}
        prob[unsure] = load(model, "full").prob(sub)
    t = np.asarray(meta["t"], float)
    post = PostParams(**(model["post"] | {k: v for k, v in (("pad_before", args.pad_before),
                                                              ("pad_after", args.pad_after)) if v is not None}))
    iv = select(prob, t, model["threshold"], post)
    print(json.dumps({"perspective": [round(persp.a, 4), round(persp.c, 2), persp.source], "unsure": len(unsure),
                      "intervals": [[round(a, 3), round(b, 3)] for a, b in iv], "kept_s": round(kept_seconds(iv), 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
