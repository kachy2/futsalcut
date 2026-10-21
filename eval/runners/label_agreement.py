"""Agreement between the first-pass labels and a blind re-label (Cohen's kappa, raw and binary).

python eval/runners/label_agreement.py eval/labels/<stem>.labels.json eval/labels/<stem>.blind.txt \
    /data/outputs/sheets_blind/<stem>/manifest.json
"""
from __future__ import annotations

import collections
import json
import sys


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> int:
    labels_path, blind_path, manifest_path = sys.argv[1:4]
    first = json.load(open(labels_path, encoding="utf-8"))["labels"]
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    order = [i for sh in manifest["sheets"] for i in sh["indices"]]
    blind = []
    for line in open(blind_path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            blind.extend(line.split(":", 1)[1].strip())
    if len(blind) != len(order):
        raise SystemExit(f"blind labels {len(blind)} != shown samples {len(order)}")
    a = [first[i] for i in order]
    b = blind
    agree = sum(x == y for x, y in zip(a, b))
    print(f"samples {len(a)}  exact agreement {agree / len(a):.1%}  kappa(raw) {kappa(a, b):.3f}")
    for mode, keep, ignore in (("strict", {"2"}, {"?"}), ("half", {"2", "M"}, {"?", "b"})):
        pairs = [(x in keep, y in keep) for x, y in zip(a, b) if x not in ignore and y not in ignore]
        ba = ["1" if p else "0" for p, _ in pairs]
        bb = ["1" if q else "0" for _, q in pairs]
        print(f"  {mode:6s} binary: n={len(pairs)} agreement {sum(x == y for x, y in zip(ba, bb)) / len(pairs):.1%} "
              f"kappa {kappa(ba, bb):.3f}")
    print("disagreements (index, first, blind):", [(i, x, y) for i, x, y in zip(order, a, b) if x != y])
    return 0


if __name__ == "__main__":
    sys.exit(main())
