"""Baseline A: DVR-Scan 1.8 (MOG2 motion events) restricted to the counting zone, scan-only.

python eval/runners/dvrscan_baseline.py /src/<video> eval/labels/<stem>.roi.json \
    eval/reports/raw/dvrscan_<stem>_t<thr>.json --threshold 0.15 [--zone strict|half]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import cv2

from futsalcut.roi import Roi


def tc_to_s(tc: str) -> float:
    h, m, s = tc.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("roi")
    ap.add_argument("out")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--zone", choices=["strict", "half"], default="strict")
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--frame-skip", type=int, default=1)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)  # display orientation (auto-rotated), like DVR-Scan sees it
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    poly = Roi.load(args.roi).zone(strict=args.zone == "strict")
    pts = []
    for x, y in poly:
        pts += [str(min(W - 1, max(0, round(x * W)))), str(min(H - 1, max(0, round(y * H))))]
    cmd = ["dvr-scan", "-i", args.video, "-so", "-q", "--ignore-user-config",
           "-t", str(args.threshold), "-df", str(args.downscale), "-fs", str(args.frame_skip),
           "-a", *pts]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        return r.returncode
    line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    tcs = [x for x in line.split(",") if x]
    intervals = [(tc_to_s(a), tc_to_s(b)) for a, b in zip(tcs[0::2], tcs[1::2])]
    res = {"tool": "dvr-scan 1.8.2.1", "cmd": cmd, "wall_s": round(wall, 1), "frame_size": [W, H],
           "intervals": intervals, "kept_s": round(sum(b - a for a, b in intervals), 1)}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("wall_s", "kept_s")} | {"events": len(intervals)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
