"""Contact sheets for ground-truth labeling: 4x4 tiles of 1-second keyframe thumbnails with the
near-half polygon drawn on top and "index  mm:ss" in the corner.

python eval/runners/make_sheets.py --cache /data/cache/<stem>.luma640x360.u8 \
    --roi eval/labels/<stem>.roi.json --out /data/outputs/sheets/<stem> [--crop-top 0.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="luma cache .u8 path (thumbs dir is derived)")
    ap.add_argument("--roi", default=None, help="json with {'near': [[x,y],...]} normalized polygon")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--tile-width", type=int, default=400)
    ap.add_argument("--indices", default=None, help="comma list of sample indices (default: all)")
    ap.add_argument("--shuffle-seed", type=int, default=None, help="shuffle order and hide times (blind relabel)")
    args = ap.parse_args()

    meta_path = args.cache[:-3] + ".json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    stem = os.path.basename(args.cache).split(".luma")[0]
    thumb_dir = os.path.join(os.path.dirname(args.cache), f"{stem}.thumbs")
    times = meta["t"]
    poly = None
    band = None
    if args.roi:
        with open(args.roi, encoding="utf-8") as f:
            roi = json.load(f)
        poly = np.array(roi["near"], np.float64)
        band = roi.get("band_y")

    idx = list(range(len(times))) if not args.indices else [int(x) for x in args.indices.split(",")]
    blind = args.shuffle_seed is not None
    if blind:
        rng = np.random.default_rng(args.shuffle_seed)
        idx = [int(i) for i in rng.permutation(idx)]
    os.makedirs(args.out, exist_ok=True)
    per = args.cols * args.rows
    manifest = []
    for s in range(0, len(idx), per):
        tiles = []
        for k, i in enumerate(idx[s:s + per]):
            img = cv2.imread(os.path.join(thumb_dir, f"{i:06d}.jpg"))
            h, w = img.shape[:2]
            th = int(round(h * args.tile_width / w))
            img = cv2.resize(img, (args.tile_width, th), interpolation=cv2.INTER_AREA)
            if poly is not None:
                pts = np.round(poly * [args.tile_width, th]).astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
            if band is not None:
                cv2.line(img, (0, int(band * th)), (args.tile_width, int(band * th)), (255, 255, 0), 1)
            label = f"#{s + k}" if blind else f"{i} {int(times[i]) // 60:02d}:{int(times[i]) % 60:02d}"
            cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        while len(tiles) < per:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[r * args.cols:(r + 1) * args.cols]) for r in range(args.rows)])
        name = f"sheet_{s // per:03d}.jpg"
        cv2.imwrite(os.path.join(args.out, name), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        manifest.append({"sheet": name, "indices": idx[s:s + per]})
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"blind": blind, "sheets": manifest}, f)
    print(f"{len(manifest)} sheets -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
