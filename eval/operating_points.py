"""Operating-point sweep for the auto mode: thresholds x post-processing on out-of-fold probabilities.

python eval/operating_points.py --auto /data/outputs/tournament/<stem>/auto.json \
    --labels eval/labels/<stem>.labels.json [--names ensemble single:pe_foot_dp] [--out eval/reports/operating_points.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import binary_gt, classifier_prf, edit_metrics, runs  # noqa: E402

from futsalcut.select import PostParams, kept_seconds, select, smooth  # noqa: E402

VARIANTS = {
    "default (2/3, min4)": PostParams(),
    "2/3, min3": PostParams(min_len=3.0),
    "no smooth, min3": PostParams(smooth_window=1, smooth_need=1, min_len=3.0),
    "1/3, min4": PostParams(smooth_need=1),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--names", nargs="+", default=["ensemble", "ensemble_light"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    auto = json.load(open(args.auto))
    oof = auto["results"]["_oof"]
    lab = json.load(open(args.labels, encoding="utf-8"))
    gt = binary_gt(lab["labels"], "strict")
    t = np.asarray(lab["t"], float)
    eps = runs(gt == 1)
    lines = ["| scorer | post | thr | CV P | CV R | edit P(band) | edit R | event R (all) | event R (>=3 s) | false scenes | kept | missed episodes |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in args.names:
        p = np.asarray(oof[name])
        for vname, pp in VARIANTS.items():
            for thr in (0.2, 0.3, 0.4, 0.5):
                pred = smooth(p >= thr, pp.smooth_window, pp.smooth_need)
                c = classifier_prf(gt, pred)
                iv = select(p, t, thr, pp)
                e = edit_metrics(gt, iv, t, pp.pad_before, pp.pad_after)
                mid = t + 0.5
                kept = np.zeros(len(t), bool)
                for a, b in iv:
                    kept |= (mid >= a) & (mid < b)
                missed = [f"{int(t[s]) // 60:02d}:{int(t[s]) % 60:02d}({e_ - s})" for s, e_ in eps if kept[s:e_].mean() < 0.5]
                lines.append(f"| {name} | {vname} | {thr} | {c.precision:.2f} | {c.recall:.2f} | {e.prf_band.precision:.2f} | "
                             f"{e.prf_band.recall:.2f} | {e.event_recall:.2f} | {e.event_recall_3s:.2f} | {e.false_scene_rate:.2f} | "
                             f"{e.kept_ratio:.1%} | {' '.join(missed)} |")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# Operating points (out-of-fold probabilities, strict GT)\n\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
