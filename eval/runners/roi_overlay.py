"""Draw an ROI file's near-half polygon (yellow) and band line (cyan) on frames of a recording,
in display orientation, to check the polygon by eye.

python eval/runners/roi_overlay.py <video> <roi.json> <out.jpg> --times 100 335 --width 1280
"""
from __future__ import annotations

import argparse
import json
import sys

import av
import cv2
import numpy as np

from futsalcut.sampler import display_rotation, rotate_cw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("roi")
    ap.add_argument("out")
    ap.add_argument("--times", type=float, nargs="+", default=[0.0])
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()
    roi = json.load(open(args.roi, encoding="utf-8"))
    poly = np.array(roi["near"], np.float64)
    band = roi.get("band_y")
    rot = display_rotation(args.video)
    tiles = []
    with av.open(args.video) as c:
        vs = c.streams.video[0]
        for t in args.times:
            c.seek(int(t / vs.time_base), stream=vs, backward=True, any_frame=False)
            fr = next(c.decode(vs))
            w0, h0 = fr.width, fr.height
            dw, dh = (h0, w0) if rot in (90, 270) else (w0, h0)
            W = args.width
            H = int(round(dh * W / dw))
            cw, ch = (H, W) if rot in (90, 270) else (W, H)
            img = rotate_cw(fr.to_ndarray(format="bgr24", width=cw, height=ch), rot).copy()
            pts = np.round(poly * [W, H]).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
            if band is not None:
                cv2.line(img, (0, int(band * H)), (W, int(band * H)), (255, 255, 0), 1, cv2.LINE_AA)
            for i, (x, y) in enumerate(pts):
                cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.putText(img, str(i), (int(x) + 5, int(y) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, f"t={t:.0f}s", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(img)
    cv2.imwrite(args.out, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
