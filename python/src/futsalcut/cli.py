"""Command line entry point: python -m futsalcut.cli <command> ..."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _cmd_probe(args: argparse.Namespace) -> int:
    from .probe import probe

    for path in args.files:
        print(json.dumps(probe(path).to_dict(), ensure_ascii=False, indent=2))
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    from .sampler import sample_file

    for path in args.files:
        ss, timing, data_path = sample_file(path, args.out, args.long_side, args.strategy,
                                            args.threads, args.limit_seconds, args.thumbs)
        span = (ss.t[-1] - ss.t[0] + 1.0) if ss.count else 0.0
        per = timing.wall_s / ss.count * 1000 if ss.count else 0.0
        print(json.dumps({
            "file": path,
            "cache": data_path,
            "samples": ss.count,
            "size": f"{ss.width}x{ss.height}",
            "strategy": args.strategy,
            "threads": args.threads,
            "video_span_s": round(span, 1),
            "wall_s": round(timing.wall_s, 2),
            "ms_per_sample": round(per, 1),
            "x_realtime": round(span / timing.wall_s, 2) if timing.wall_s else None,
            "decode_s": round(timing.decode_s, 2),
            "luma_s": round(timing.luma_s, 2),
            "downscale_s": round(timing.downscale_s, 2),
            "write_s": round(timing.write_s, 2),
            "packets_total": timing.packets_total,
            "packets_decoded": timing.packets_decoded,
        }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="futsalcut", description=f"futsalcut {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("probe", help="describe recordings (index only)")
    sp.add_argument("files", nargs="+")
    sp.set_defaults(func=_cmd_probe)

    sp = sub.add_parser("sample", help="decode the 1-second keyframe grid into a luma cache")
    sp.add_argument("files", nargs="+")
    sp.add_argument("--out", default="/data/cache")
    sp.add_argument("--long-side", type=int, default=640)
    sp.add_argument("--strategy", choices=["packets", "skipframe", "all"], default="packets")
    sp.add_argument("--threads", type=int, default=2)
    sp.add_argument("--limit-seconds", type=float, default=None)
    sp.add_argument("--thumbs", action="store_true", help="also write colour JPEG thumbnails")
    sp.set_defaults(func=_cmd_sample)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
