"""G1 report assets: the actual output video for the recommended auto setting (lossless 4K + 720p
preview), a timeline image (GT vs candidates), and an example sheet for the strict-vs-half decision.

python eval/make_g1_assets.py --video /src/<video> --auto /data/outputs/tournament/<stem>/auto.json \
    --labels eval/labels/<stem>.labels.json --cache /data/cache/<stem>.luma640x360.u8 --out /data/outputs/g1 \
    --scorer ensemble_light --threshold 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import binary_gt, edit_metrics  # noqa: E402

from futsalcut.remux import Segment, remux  # noqa: E402
from futsalcut.select import PostParams, kept_seconds, select  # noqa: E402
from futsalcut.verify import check_file  # noqa: E402

COLORS = {"2": (60, 180, 60), "M": (40, 200, 230), "b": (170, 170, 170), "?": (255, 255, 255),
          "1": (60, 60, 220), "F": (200, 120, 60), "0": (90, 90, 90)}


def timeline(labels, t, rows: list[tuple[str, np.ndarray | list]], width: int = 1600) -> np.ndarray:
    """rows: (name, per-sample probability array) or (name, list of kept intervals)."""
    dur = t[-1] + 1
    row_h, label_w = 26, 210
    H = row_h * (len(rows) + 2) + 30
    img = np.full((H, width, 3), 255, np.uint8)
    x = lambda s: int(label_w + (width - label_w - 10) * s / dur)  # noqa: E731
    cv2.putText(img, "GT label", (6, 18 + row_h), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    for i, lb in enumerate(labels):
        cv2.rectangle(img, (x(t[i]), row_h + 4), (max(x(t[i]) + 1, x(t[i] + 1)), 2 * row_h - 4), COLORS[lb], -1)
    for r, (name, data) in enumerate(rows):
        y0 = (r + 2) * row_h + 4
        cv2.putText(img, name[:26], (6, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        if isinstance(data, np.ndarray):
            for i, p in enumerate(data):
                v = int(255 * (1 - float(p)))
                cv2.rectangle(img, (x(t[i]), y0), (max(x(t[i]) + 1, x(t[i] + 1)), y0 + row_h - 8), (v, v, 255), -1)
        else:
            for a, b in data:
                cv2.rectangle(img, (x(a), y0), (x(b), y0 + row_h - 8), (40, 40, 40), -1)
    for m in range(0, int(dur) + 1, 60):
        cv2.line(img, (x(m), row_h), (x(m), H - 24), (220, 220, 220), 1)
        cv2.putText(img, f"{m // 60}:00", (x(m) - 12, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    cv2.putText(img, "GT: green=keep(2+ near)  yellow=2+ near halfway(M)  grey=far/uncertain  | rows: dark=kept, red=probability",
                (label_w, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return img


def example_sheet(cache_path: str, labels: list[str], t, n_each: int = 6) -> np.ndarray:
    stem = os.path.basename(cache_path).split(".luma")[0]
    tdir = os.path.join(os.path.dirname(cache_path), f"{stem}.thumbs")
    rows = []
    for lb, title in (("2", "STRICT keep: 2+ players in the near zone"), ("M", "HALF only: 2+ players near the halfway line")):
        idx = [i for i, x in enumerate(labels) if x == lb]
        pick = [idx[int(k)] for k in np.linspace(0, len(idx) - 1, n_each)]
        tiles = []
        for i in pick:
            im = cv2.resize(cv2.imread(os.path.join(tdir, f"{i:06d}.jpg")), (320, 180), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{int(t[i]) // 60:02d}:{int(t[i]) % 60:02d}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(im)
        bar = np.full((24, 320 * n_each, 3), 255, np.uint8)
        cv2.putText(bar, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        rows += [bar, np.hstack(tiles)]
    return np.vstack(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--auto", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scorer", default="ensemble_light")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--min-len", type=float, default=3.0)
    ap.add_argument("--extra-intervals", nargs="*", default=[], help="name=path.json with {'intervals': [...]}")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    oof = json.load(open(args.auto))["results"]["_oof"]
    lab = json.load(open(args.labels, encoding="utf-8"))
    labels, t = lab["labels"], np.asarray(lab["t"], float)
    gt = binary_gt(labels, "strict")
    pp = PostParams(min_len=args.min_len)
    prob = np.asarray(oof[args.scorer])
    iv = select(prob, t, args.threshold, pp)
    em = edit_metrics(gt, iv, t, pp.pad_before, pp.pad_after)
    summary = {"scorer": args.scorer, "threshold": args.threshold, "post": pp.__dict__, "intervals": iv,
               "kept_s": kept_seconds(iv), "scenes": len(iv), "edit": em.__dict__ | {"prf_band": em.prf_band.__dict__}}
    # lossless output from the original recording
    out4k = os.path.join(args.out, "result_4k_lossless.mp4")
    rep = remux([args.video], [Segment(0, a, b) for a, b in iv], out4k)
    fc = check_file(out4k)
    summary["remux"] = {k: v for k, v in rep.__dict__.items() if k != "per_segment"}
    summary["output_check"] = {"streams": [s.__dict__ for s in fc.streams], "decode_errors": fc.decode_errors[:5],
                               "bytes": os.path.getsize(out4k)}
    prev = os.path.join(args.out, "result_preview_720p.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out4k, "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "28", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", prev], check=True)
    summary["preview_bytes"] = os.path.getsize(prev)
    rows = [(f"{args.scorer} p", prob), (f"{args.scorer} kept (thr {args.threshold})", iv)]
    for k in ("ensemble", "dfine_count", "fg_zone_w_gain_dp", "pe_foot_dp"):
        if k in oof and k != args.scorer:
            rows.append((k, np.asarray(oof[k])))
    for spec in args.extra_intervals:
        name, path = spec.split("=", 1)
        rows.append((name, json.load(open(path))["intervals"]))
    cv2.imwrite(os.path.join(args.out, "timeline.png"), timeline(labels, t, rows))
    cv2.imwrite(os.path.join(args.out, "strict_vs_half.jpg"), example_sheet(args.cache, labels, t), [cv2.IMWRITE_JPEG_QUALITY, 88])
    with open(os.path.join(args.out, "g1_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: summary[k] for k in ("kept_s", "scenes")} | {"edit_P": em.prf_band.precision, "edit_R": em.prf_band.recall,
                     "event_R_3s": em.event_recall_3s, "remux_wall_s": round(rep.wall_s, 1), "decode_errors": len(fc.decode_errors),
                     "max_cut_av_err_ms": round(rep.max_cut_av_error_s * 1000, 1), "out_MB": round(os.path.getsize(out4k) / 1e6),
                     "preview_MB": round(summary["preview_bytes"] / 1e6, 1)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
