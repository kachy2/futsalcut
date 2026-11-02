"""Cascade budget: how few person-detector calls can the auto mode make without losing quality?

The auto mode sends the samples whose light probability is "unsure" to the detector (web/src/core/model.ts unsureSamples) and
uses the full ensemble there. For several zones (the labelled strict zone, the near half, the app's default suggestion) and several
rules for choosing those samples, this runs the same steps as the web app (see eval/reference_run.py) and reports, against the
labels: edit precision / recall / F1 and the recall of episodes >= 3 s, the share of samples sent to the detector, and how much of
the kept time agrees with the shipped rule. Thresholds other than the default are reported too, because the sensitivity slider
moves the threshold after the analysis.

python eval/cascade_budget.py --stem <stem> --name video_1 [--report eval/reports/cascade_budget.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import binary_gt, edit_metrics, intervals_to_mask  # noqa: E402

from futsalcut.ensemble import Calibrator, Ensemble  # noqa: E402
from futsalcut.features import pass2, perspective_from_blobs, perspective_from_boxes, run_pass1  # noqa: E402
from futsalcut.roi import Roi, point_in_polygon, polygon_mask  # noqa: E402
from futsalcut.sampler import load_cache  # noqa: E402
from futsalcut.select import PostParams, kept_seconds, select  # noqa: E402

APP_DEFAULT = [(0.12, 0.25), (0.88, 0.25), (1.0, 1.0), (0.0, 1.0)]  # the "default" button of web/src/ui/RoiEditor.svelte


def ensemble(model: dict, name: str) -> Ensemble:
    m = model[name]
    return Ensemble(m["names"], {k: Calibrator(**v) for k, v in m["calibrators"].items()}, m["weights"], m["bias"])


def prepare(cache, meta, model: dict, zone_poly, boxes) -> tuple[np.ndarray, np.ndarray]:
    """Light probability of every sample, and the full-ensemble probability it would get with its detector count."""
    H, W = meta["height"], meta["width"]
    zone_top = min(y for _, y in zone_poly) * H
    p1 = run_pass1(cache, polygon_mask(zone_poly, W, H))
    stride = model["detector"]["perspective_stride"]
    persp = perspective_from_boxes([boxes[i] for i in range(0, len(boxes), stride)], H, zone_top) or perspective_from_blobs(p1, zone_top)
    scores = pass2(p1, persp, H)
    light = ensemble(model, "light").prob(scores)

    def count(i: int) -> float:
        inz = [b for b in boxes[i] if point_in_polygon(zone_poly, (b[0] + b[2]) / 2, min(0.9999, b[3]))]
        return sum(1 for b in inz if b[4] >= model["detector"]["min_score"]) + min(0.99, sum(b[4] for b in inz) / 10)

    det = np.array([count(i) for i in range(len(light))])
    full = ensemble(model, "full").prob({k: np.asarray(v) for k, v in scores.items()} | {"dfine_count": det})
    return np.asarray(light), np.asarray(full)


def chosen(rule: tuple, light: np.ndarray, thr: float) -> np.ndarray:
    """Indices sent to the detector under a rule."""
    kind = rule[0]
    if kind == "band":
        return np.where((light > rule[1]) & (light < rule[2]))[0]
    if kind == "around":  # a band around the decision threshold
        return np.where(np.abs(light - thr) < rule[1])[0]
    if kind == "cap":  # inside the shipped band, at most this share of all samples, closest to the threshold first
        inside = np.where((light > 0.05) & (light < 0.95))[0]
        k = int(round(rule[1] * len(light)))
        return np.sort(inside[np.argsort(np.abs(light[inside] - thr), kind="stable")[:k]])
    raise ValueError(rule)


def label(rule: tuple, shipped: tuple) -> str:
    if rule[0] == "band":
        tags = [t for t, ok in (("비교 기준", rule == RULES[0]), ("모델", tuple(rule[1:]) == shipped)) if ok]
        return f"구간 {rule[1]}~{rule[2]}" + (f" ({', '.join(tags)})" if tags else "")
    if rule[0] == "around":
        return f"임계값 ±{rule[1]}"
    return f"상한 {int(rule[1] * 100)}%"


RULES = [("band", 0.05, 0.95), ("band", 0.1, 0.9), ("band", 0.15, 0.85), ("band", 0.2, 0.8), ("band", 0.3, 0.7),
         ("around", 0.25), ("around", 0.2), ("around", 0.15), ("around", 0.1), ("cap", 0.3), ("cap", 0.2), ("cap", 0.1)]
THRESHOLDS = (0.3, 0.15, 0.5)  # the auto mode's threshold first


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--name", default="video_1", help="name used in the report (the recording's file name is not published)")
    ap.add_argument("--model", default="python/src/futsalcut/models/ensemble_v1.json")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    model = json.load(open(args.model))
    cache, meta = load_cache(f"/data/cache/{args.stem}.luma640x360.u8")
    roi = Roi.load(f"eval/labels/{args.stem}.roi.json")
    labels = json.load(open(f"eval/labels/{args.stem}.labels.json"))["labels"]
    boxes = json.load(open(f"eval/reports/raw/dfine_{args.stem}.json"))["boxes"]
    t = np.asarray(meta["t"], float)
    post = PostParams(**model["post"])
    zones = [("엄격 구역(라벨 기준)", roi.zone(strict=True), "strict"), ("카메라 쪽 진영", roi.zone(strict=False), "half"),
             ("앱 기본 구역", APP_DEFAULT, "half")]
    out = {"name": args.name, "samples": len(t), "zones": []}
    lines = [f"# 검출 대상 줄이기 평가 ({args.name}, 표본 {len(t)}개)", "",
             "자동 모드는 가벼운 판정이 애매한 표본만 사람 검출기로 다시 본다. 표본을 고르는 규칙별로, 웹앱과 같은 계산으로 편집 결과를 만들어",
             f"라벨과 비교했다. 편집 지표는 장면 앞뒤 여유({post.pad_before}초/{post.pad_after}초)를 허용 구간으로 본다. "
             "'비교 기준과 일치'는 남긴 시간이 비교 기준(0.05~0.95, 2026-11-02 전 기본값)의 결과와 겹치는 비율(IoU)이다.", ""]
    for zname, poly, mode in zones:
        gt = binary_gt(labels, mode)
        light, full = prepare(cache, meta, model, poly, boxes)
        base_mask = {}
        rows = []
        for rule in RULES:
            row = {"rule": label(rule, tuple(model["detector"]["cascade"]))}
            for thr in THRESHOLDS:
                idx = chosen(rule, light, thr)
                prob = light.copy()
                prob[idx] = full[idx]
                iv = select(prob, t, thr, post)
                m = edit_metrics(gt, iv, t, post.pad_before, post.pad_after)
                mask = intervals_to_mask(iv, t)
                if rule == RULES[0]:
                    base_mask[thr] = mask
                inter, union = (mask & base_mask[thr]).sum(), (mask | base_mask[thr]).sum()
                row[thr] = {"share": round(len(idx) / len(light), 4), "p": m.prf_band.precision, "r": m.prf_band.recall,
                            "f1": m.prf_band.f1, "event_r3": m.event_recall_3s, "kept_s": round(kept_seconds(iv), 1),
                            "agree": round(float(inter / union), 4) if union else 1.0}
            rows.append(row)
        out["zones"].append({"zone": zname, "gt": mode, "rows": rows,
                             "light_hist": np.histogram(light, bins=10, range=(0, 1))[0].tolist()})
        b = rows[0][0.3]
        lines += [f"## {zname} (정답 기준: {'엄격' if mode == 'strict' else '진영'})", "",
                  f"가벼운 판정 확률 분포(0.1 간격): {out['zones'][-1]['light_hist']}", "",
                  "| 규칙 | 검출 비율 | 정밀도 | 재현율 | F1 | 3초 이상 에피소드 | 남긴 시간 | 비교 기준과 일치 | F1 @0.15 / @0.5 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            r = row[0.3]
            ok = (r["f1"] >= b["f1"] - 0.01 and r["r"] >= b["r"] - 0.01)
            lines.append(f"| {row['rule']}{'' if ok else ' ✗'} | {100 * r['share']:.1f}% | {r['p']:.3f} | {r['r']:.3f} | {r['f1']:.3f} | "
                         f"{r['event_r3']:.2f} | {r['kept_s']:.0f}초 | {r['agree']:.3f} | {row[0.15]['f1']:.3f} / {row[0.5]['f1']:.3f} |")
        lines.append("")
    lines += ["✗: 이 구역에서 임계값 0.3의 F1이나 재현율이 비교 기준보다 0.01 넘게 낮다(채택 기준 미달).", "",
              "결정(2026-11-02): 라벨이 있는 엄격 구역에서 임계값 0.15·0.3·0.5 모두 정밀도·재현율이 비교 기준 이상인 안 가운데 "
              "검출 호출이 가장 적은 0.2~0.8을 모델의 구간으로 채택했다. 임계값 주변 구간은 0.3에서는 같지만 0.5에서 재현율이 "
              "0.981 → 0.885로 떨어져 감도 슬라이더를 움직이는 사용자에게 불리하다. 영상 1개의 결과이며 다른 영상은 task-0005에서 본다.", ""]
    text = "\n".join(lines)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        with open(os.path.splitext(args.report)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
