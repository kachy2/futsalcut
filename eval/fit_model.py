"""Fit the shipped ensemble model (light + full) from labeled videos using the product feature module,
report out-of-fold metrics at the default operating point, and write models/ensemble_v1.json.

python eval/fit_model.py --video-set eval/labels/<stem>  [--video-set ...] --out python/src/futsalcut/models/ensemble_v1.json
Each video set needs: /data/cache/<stem>.luma640x360.u8, eval/labels/<stem>.{roi,labels}.json, eval/reports/raw/dfine_<stem>.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import binary_gt, block_bootstrap, classifier_prf, edit_metrics  # noqa: E402

from futsalcut.ensemble import Ensemble  # noqa: E402
from futsalcut.features import pass2, perspective_from_blobs, perspective_from_boxes, run_pass1  # noqa: E402
from futsalcut.roi import Roi, point_in_polygon  # noqa: E402
from futsalcut.sampler import load_cache  # noqa: E402
from futsalcut.select import PostParams, select, smooth  # noqa: E402

LIGHT = ["fg_zone_w_gain", "pe_foot", "count_foot", "framediff_zone"]
FULL = LIGHT + ["dfine_count"]
DETECTOR_STRIDE = 12      # perspective calibration: detector on every 12th sample
DET_MIN_SCORE = 0.4
THRESHOLD = 0.3
POST = PostParams(min_len=3.0)
CASCADE_LO, CASCADE_HI = 0.05, 0.95


def dfine_count(boxes_per_sample, zone_poly) -> np.ndarray:
    out = np.zeros(len(boxes_per_sample))
    for i, boxes in enumerate(boxes_per_sample):
        inz = [b for b in boxes if point_in_polygon(zone_poly, (b[0] + b[2]) / 2, min(0.9999, b[3]))]
        hard = sum(1 for b in inz if b[4] >= DET_MIN_SCORE)
        out[i] = hard + min(0.99, sum(b[4] for b in inz) / 10)
    return out


def load_set(stem: str, cache_dir: str, raw_dir: str) -> dict:
    cache, meta = load_cache(os.path.join(cache_dir, f"{stem}.luma640x360.u8"))
    H, W = meta["height"], meta["width"]
    roi = Roi.load(f"eval/labels/{stem}.roi.json")
    zone = roi.mask(W, H, strict=True)
    zone_poly = roi.zone(strict=True)
    zone_top = float(min(y for _, y in zone_poly)) * H
    lab = json.load(open(f"eval/labels/{stem}.labels.json", encoding="utf-8"))
    boxes = json.load(open(os.path.join(raw_dir, f"dfine_{stem}.json")))["boxes"]
    p1 = run_pass1(cache, zone)
    persp = perspective_from_boxes(boxes[::DETECTOR_STRIDE], H, zone_top) or perspective_from_blobs(p1, zone_top)
    f = pass2(p1, persp, H)
    f["dfine_count"] = dfine_count(boxes, zone_poly)
    return {"stem": stem, "scores": f, "gt": binary_gt(lab["labels"], "strict"), "t": np.asarray(lab["t"], float),
            "perspective": asdict(persp)}


def summarize(s: dict, prob: np.ndarray) -> dict:
    pred = smooth(prob >= THRESHOLD, POST.smooth_window, POST.smooth_need)
    iv = select(prob, s["t"], THRESHOLD, POST)
    em = edit_metrics(s["gt"], iv, s["t"], POST.pad_before, POST.pad_after)
    return {"cv": asdict(classifier_prf(s["gt"], pred)), "cv_f1_ci95": block_bootstrap(s["gt"], pred), "edit": asdict(em)}


OOF_PROBS: dict[tuple[str, str], np.ndarray] = {}


def oof_eval(sets: list[dict], members: list[str], blocks: int = 6, tag: str = "") -> dict:
    res = {}
    for s in sets:
        n = len(s["t"])
        edges = np.linspace(0, n, blocks + 1).astype(int)
        prob = np.zeros(n)
        for b in range(blocks):
            test = np.zeros(n, bool)
            test[edges[b]:edges[b + 1]] = True
            # train on the other blocks of this video plus all other videos
            X = {k: np.concatenate([s["scores"][k][~test & (s["gt"] >= 0)]] + [o["scores"][k][o["gt"] >= 0] for o in sets if o is not s])
                 for k in members}
            y = np.concatenate([(s["gt"][~test & (s["gt"] >= 0)] == 1)] + [(o["gt"][o["gt"] >= 0] == 1) for o in sets if o is not s]).astype(float)
            prob[test] = Ensemble.fit(X, y, members).prob({k: s["scores"][k][test] for k in members})
        OOF_PROBS[(tag, s["stem"])] = prob
        res[s["name"]] = summarize(s, prob)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", action="append", required=True)
    ap.add_argument("--name", action="append", default=None,
                    help="public name per --stem written into the model (default video_1, video_2, ...)")
    ap.add_argument("--cache-dir", default="/data/cache")
    ap.add_argument("--raw-dir", default="eval/reports/raw")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets = [load_set(s, args.cache_dir, args.raw_dir) for s in args.stem]
    names = args.name or [f"video_{i + 1}" for i in range(len(sets))]
    assert len(names) == len(sets), "give one --name per --stem"
    for s, name in zip(sets, names):
        s["name"] = name  # recordings are private: only this name goes into the published model
    report = {}
    for name, members in (("light", LIGHT), ("full", FULL)):
        report[name] = oof_eval(sets, members, tag=name)
    # cascade: the detector score is used only where the light ensemble is unsure
    report["cascade"] = {}
    for s in sets:
        pl, pf = OOF_PROBS[("light", s["stem"])], OOF_PROBS[("full", s["stem"])]
        unsure = (pl > CASCADE_LO) & (pl < CASCADE_HI)
        report["cascade"][s["name"]] = summarize(s, np.where(unsure, pf, pl)) | {
            "detector_share": round(float(unsure.mean()), 4)}
    for name in ("light", "full", "cascade"):
        for stem, r in report[name].items():
            e = r["edit"]
            print(f"[{name}] {stem}: CV P {r['cv']['precision']:.2f} R {r['cv']['recall']:.2f} F1 {r['cv']['f1']:.2f} "
                  f"{r['cv_f1_ci95']} | edit P {e['prf_band']['precision']:.2f} R {e['prf_band']['recall']:.2f} "
                  f"eventR3 {e['event_recall_3s']:.2f} false {e['false_scene_rate']:.2f} kept {e['kept_ratio']:.1%}"
                  + (f" detector share {r['detector_share']:.1%}" if "detector_share" in r else ""))
    model = {"version": 1, "created": dt.date.today().isoformat(), "trained_on": names,
             "gt": "strict (2+ players in the counting zone)", "threshold": THRESHOLD, "post": asdict(POST),
             "detector": {"model": "dfine_n_coco.onnx (ustc-community/dfine-nano-coco@066438d, Apache-2.0)",
                          "min_score": DET_MIN_SCORE, "perspective_stride": DETECTOR_STRIDE,
                          "cascade": [CASCADE_LO, CASCADE_HI]},
             "perspectives": {s["name"]: s["perspective"] for s in sets}, "oof": report}
    for name, members in (("light", LIGHT), ("full", FULL)):
        X = {k: np.concatenate([s["scores"][k][s["gt"] >= 0] for s in sets]) for k in members}
        y = np.concatenate([(s["gt"][s["gt"] >= 0] == 1) for s in sets]).astype(float)
        e = Ensemble.fit(X, y, members)
        model[name] = {"names": e.names, "weights": e.weights, "bias": e.bias,
                       "calibrators": {k: asdict(c) for k, c in e.calibrators.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
