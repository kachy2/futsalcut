"""Combine track A (as-is runs) and the tournament (idea-level quality) into eval/reports/baseline.md.

python eval/report_baseline.py --track-a /data/outputs/track_a/track_a_results.json \
    --tournament /data/outputs/tournament/<stem>/results.json --out eval/reports/baseline.md
"""
from __future__ import annotations

import argparse
import json
import os

PROPOSALS = [
    # name, idea-level tournament candidate(s), method summary
    ("claude_full", ["hog_full"], "2 s chunks, YOLOv4→MobileNet→HOG fallback, OpenCV re-encode"),
    ("claude_simple", ["mog2_blobs_full", "hog_full"], "2 s chunks, MOG2 + HOG, OpenCV re-encode"),
    ("gemini", ["framediff_full"], "3 s chunks, frame differencing every frame, top-N, OpenCV re-encode"),
    ("chatgpt", ["dfine_full_count"], "2 s chunks, YOLOv8n person count (no ROI), ffmpeg concat list"),
    ("grok", ["mog2_blobs_full", "framediff_full"], "MOG2 blobs + differencing every frame, ffmpeg copy"),
    ("perplexity", ["hog_full", "framediff_full"], "2 fps samples, HOG + differencing, 8 s chunks, ffmpeg copy"),
    ("qwen", ["dfine_full_count"], "YOLOv5 on every frame, OpenCV re-encode"),
    ("deepseek", [], "MobileNet-SSD (Caffe prototxt + TF .pb mismatch)"),
    ("liner", [], "concept only, no code"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-a", required=True)
    ap.add_argument("--tournament", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ta = json.load(open(args.track_a)) if os.path.exists(args.track_a) else {}
    tour = {r["name"]: r for r in json.load(open(args.tournament))["results"]}
    lines = ["# Baseline comparison: the 8 AI proposals, DVR-Scan, and our candidates", "",
             "Track A runs each script as-is (paths + headless fix only) on a 60-s 4K excerpt (05:00-06:00) on the test",
             "host (2 vCPU, no AVX). Track B re-implements each script's scoring idea on the shared 1 Hz cache with",
             "identical post-processing, so quality differences come from the idea, not from incidental bugs.", "",
             "| proposal | method | track A: status | wall (60 s) | est. 2 h video | output usable? | idea-level CV F1 (track B) |",
             "|---|---|---|---|---|---|---|"]
    for name, cands, method in PROPOSALS:
        r = ta.get(name)
        if name in ("deepseek", "liner"):
            status, wall, est, usable = "not runnable", "-", "-", "-"
        elif r is None:
            status, wall, est, usable = "not run", "-", "-", "-"
        else:
            if r["returncode"] is None:
                status = "not runnable as-is"
            else:
                status = "timeout" if r["timed_out"] else ("ok" if r["returncode"] == 0 else f"error rc={r['returncode']}")
            wall = f"{r['wall_s']:.0f} s" if r.get("wall_s") is not None else "-"
            est = f"{r['est_2h_h']:.1f} h" if r.get("est_2h_h") is not None else "-"
            if r["returncode"] is None:
                est = f"~{est} (est.)"
            o = r["output"]
            if not o.get("exists"):
                usable = "no output" + (f" ({o['note']})" if o.get("note") else "")
            elif "has_audio" in o:
                usable = f"{o.get('duration_s', 0):.0f} s, audio {'yes' if o['has_audio'] else 'NO'}" + \
                    (f"; {o['note']}" if o.get("note") else "") + (", **empty file**" if o.get("bytes", 1000) < 1000 else "")
            else:
                usable = "cut list only"
        f1 = ", ".join(f"{c} {tour[c]['cv']['f1']:.2f}" for c in cands if c in tour) or "-"
        lines.append(f"| {name} | {method} | {status} | {wall} | {est} | {usable} | {f1} |")
    lines += ["", "Reference rows (track B):", "", "| candidate | AP | CV F1 [95% CI] | kept |", "|---|---|---|---|"]
    for k in ("dvrscan_t0.15", "dvrscan_t0.5", "fg_zone_w_gain", "pe_foot", "count_foot", "dfine_count",
              "framediff_zone", "anchor_keep_all"):
        if k in tour:
            r = tour[k]
            lines.append(f"| {k} | {r['ap']:.3f} | {r['cv']['f1']:.2f} [{r['cv_f1_ci95'][0]:.2f}, {r['cv_f1_ci95'][1]:.2f}] | "
                         f"{r['edit_cvthr']['kept_ratio']:.1%} |")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
