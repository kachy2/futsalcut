"""Error analysis for one tournament candidate: per-episode coverage, false-positive samples, and a
contact sheet of the misclassified samples.

python eval/diagnose.py --out /data/outputs/tournament/<stem> --labels eval/labels/<stem>.labels.json \
    --cache /data/cache/<stem>.luma640x360.u8 --candidate fg_zone [--sheet /data/outputs/diag_fg_zone.jpg]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import binary_gt, runs  # noqa: E402

from futsalcut.select import PostParams, smooth  # noqa: E402


def mmss(s: float) -> str:
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--sheet", default=None)
    args = ap.parse_args()
    feats = dict(np.load(os.path.join(args.out, "features.npz")))
    res = {r["name"]: r for r in json.load(open(os.path.join(args.out, "results.json")))["results"]}
    r = res[args.candidate]
    score = feats[args.candidate]
    lab = json.load(open(args.labels, encoding="utf-8"))
    labels, t = lab["labels"], np.asarray(lab["t"])
    gt = binary_gt(labels, "strict")
    thr = r["edit_cvthr"]["threshold"]
    p = PostParams()
    pred = smooth(score >= thr, p.smooth_window, p.smooth_need)
    print(f"{args.candidate}: threshold {thr}")
    print("episodes (start, len, covered, scores):")
    for s, e in runs(gt == 1):
        cov = pred[s:e].mean()
        print(f"  {mmss(t[s])} len={e - s:2d} covered={cov:.2f} scores={np.round(score[s:e], 4).tolist()}")
    fp = np.where(pred & (gt == 0))[0]
    print(f"false-positive samples ({len(fp)}):", [(int(i), mmss(t[i]), labels[i], round(float(score[i]), 4)) for i in fp])
    if args.sheet:
        stem = os.path.basename(args.cache).split(".luma")[0]
        tdir = os.path.join(os.path.dirname(args.cache), f"{stem}.thumbs")
        fn = [i for s, e in runs(gt == 1) for i in range(s, e) if not pred[i]]
        items = [(i, "FP") for i in fp] + [(i, "FN") for i in fn]
        tiles = []
        for i, kind in items[:24]:
            img = cv2.imread(os.path.join(tdir, f"{i:06d}.jpg"))
            img = cv2.resize(img, (400, 225), interpolation=cv2.INTER_AREA)
            txt = f"{kind} {i} {mmss(t[i])} {labels[i]} s={score[i]:.4f}"
            cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(img)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        if tiles:
            grid = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
            cv2.imwrite(args.sheet, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print("sheet:", args.sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
