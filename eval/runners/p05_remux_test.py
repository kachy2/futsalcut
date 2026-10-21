"""P0.5(c): cut 5 segments out of each synthetic file with the PyAV remuxer and verify the result.

Run inside futsalcut-py:  python eval/runners/p05_remux_test.py --syn /data/synthetic --out /data/outputs/p05
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

from futsalcut.remux import Segment, remux
from futsalcut.verify import av_offsets, check_file

CASES = {
    # name: (sources, [(file_index, start, end)])  -- boundaries on the 1 s keyframe grid
    "h264_gop1": (["syn_h264_gop1.mp4"], [(0, 3, 9), (0, 20, 27), (0, 40, 52), (0, 70, 75), (0, 100, 112)]),
    "h264_bframes": (["syn_h264_bframes.mp4"], [(0, 2, 6), (0, 9, 13), (0, 18, 22), (0, 27, 31), (0, 35, 40)]),
    "hevc_opengop": (["syn_hevc_opengop.mp4"], [(0, 2, 6), (0, 9, 13), (0, 18, 22), (0, 27, 31), (0, 35, 40)]),
    "rot90": (["syn_rot90.mp4"], [(0, 2, 6), (0, 9, 13), (0, 18, 22), (0, 27, 31), (0, 35, 40)]),
    "vfr": (["syn_vfr.mp4"], [(0, 2, 6), (0, 9, 13), (0, 18, 22), (0, 27, 31), (0, 35, 40)]),
    "split_pair": (["syn_split_a.mp4", "syn_split_b.mp4"],
                   [(0, 10, 20), (0, 50, 60), (1, 0, 8), (1, 30, 36), (1, 50, 60)]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", default="/data/synthetic")
    ap.add_argument("--out", default="/data/outputs/p05")
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--verify-prefix", default=None,
                    help="only verify existing outputs named <prefix><case>.mp4 (e.g. mb_remux_)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = {}
    prefix = args.verify_prefix or "remux_"
    for name in args.cases:
        srcs, segs = CASES[name]
        sources = [os.path.join(args.syn, s) for s in srcs]
        out = os.path.join(args.out, f"{prefix}{name}.mp4")
        if args.verify_prefix:
            if not os.path.exists(out):
                results[name] = {"missing": out}
                print(f"== {name}: missing {out}")
                continue
            rep = None
        else:
            rep = remux(sources, [Segment(*s) for s in segs], out)
        expected = sum(e - s for _, s, e in segs)
        fc = check_file(out)
        offs = av_offsets(out)
        src_offs = av_offsets(sources[0])
        results[name] = {
            "expected_video_s": expected,
            "report": {k: v for k, v in rep.__dict__.items() if k != "per_segment"} if rep else None,
            "per_segment": rep.per_segment if rep else None,
            "streams": [s.__dict__ for s in fc.streams],
            "decode_errors": fc.decode_errors[:5],
            "source_av_offset_ms": round(statistics.median(src_offs) * 1000, 2) if src_offs else None,
            "av_offsets_ms": [round(o * 1000, 2) for o in offs],
            "av_offset_spread_ms": round((max(offs) - min(offs)) * 1000, 2) if offs else None,
        }
        vdur = next((s.last_end_s - (s.first_pts_s or 0) for s in fc.streams if s.kind == "video"), None)
        adur = next((s.last_end_s - (s.first_pts_s or 0) for s in fc.streams if s.kind == "audio"), None)
        print(f"== {name}: video {vdur}s (expected {expected}), audio {adur}s, "
              f"decode errors {len(fc.decode_errors)}, A/V offsets ms {results[name]['av_offsets_ms']}")
    with open(os.path.join(args.out, f"p05_{prefix}results.json"), "w") as f:
        json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
