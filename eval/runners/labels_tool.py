"""Assemble per-sheet label strings into a labels JSON and print the label distribution.

python eval/runners/labels_tool.py eval/labels/<stem>.sheets.txt /data/cache/<stem>.luma640x360.json \
    eval/labels/<stem>.labels.json
"""
from __future__ import annotations

import collections
import json
import sys

VALID = set("2M1F0b?")


def parse_sheets(path: str) -> list[str]:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, labels = (s.strip() for s in line.split(":", 1))
            bad = set(labels) - VALID
            if bad:
                raise ValueError(f"{name}: invalid labels {bad}")
            rows[int(name.split("_")[1])] = labels
    out = []
    for i in sorted(rows):
        out.extend(rows[i])
    return out


def main() -> int:
    sheets, meta_path, out_path = sys.argv[1:4]
    labels = parse_sheets(sheets)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    t = meta["t"]
    if len(labels) != len(t):
        raise SystemExit(f"label count {len(labels)} != sample count {len(t)}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "rules": "docs/labeling.md", "source": meta["source"],
                   "labels": labels, "t": t}, f)
    c = collections.Counter(labels)
    n = len(labels)
    print("counts:", dict(sorted(c.items())))
    print(f"'?' share: {c['?'] / n:.1%}   strict keep ('2'): {c['2'] / (n - c['?']):.1%}   "
          f"half keep ('2','M'): {(c['2'] + c['M']) / (n - c['?'] - c['b']):.1%} (of {n - c['?'] - c['b']} scored)")
    # episodes (runs of '2')
    runs, cur = [], None
    for i, lb in enumerate(labels):
        if lb == "2" and cur is None:
            cur = i
        if lb != "2" and cur is not None:
            runs.append((cur, i))
            cur = None
    if cur is not None:
        runs.append((cur, n))
    print("strict episodes:", [(f"{int(t[a]) // 60:02d}:{int(t[a]) % 60:02d}", b - a) for a, b in runs])
    return 0


if __name__ == "__main__":
    sys.exit(main())
