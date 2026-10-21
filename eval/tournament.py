"""Algorithm tournament on the shared 1 Hz luma cache (plan P2-B / P3).

Every candidate produces one score per sample; all share the same selection/post-processing and
metrics. Features are cached in <out>/features.npz; external scores (YOLO, DVR-Scan) are read from
eval/reports/raw/ when present.

python eval/tournament.py --cache /data/cache/<stem>.luma640x360.u8 --roi eval/labels/<stem>.roi.json \
    --labels eval/labels/<stem>.labels.json --out /data/outputs/tournament/<stem> [--report eval/reports/tournament.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import binary_gt, block_bootstrap, classifier_prf, edit_metrics  # noqa: E402

from futsalcut.roi import Roi  # noqa: E402
from futsalcut.sampler import load_cache  # noqa: E402
from futsalcut.select import PostParams, kept_seconds, select, smooth, threshold_for_budget  # noqa: E402

FG_T = 22          # foreground threshold (gray levels) vs median background
DIFF_T = 25        # frame-difference threshold (proposal idea)
BG_HALF, BG_STEP, BG_EVERY = 60, 4, 10


# ----------------------------------------------------------------------------- shared helpers
def majority3(mask: np.ndarray) -> np.ndarray:
    s = cv2.boxFilter(mask.astype(np.uint8), cv2.CV_16U, (3, 3), normalize=False,
                      borderType=cv2.BORDER_CONSTANT)
    return s >= 5


def backgrounds(cache: np.ndarray) -> list[np.ndarray]:
    """Lower-median background per block of BG_EVERY samples (window centred on the block)."""
    n = len(cache)
    out = []
    for b in range(0, n, BG_EVERY):
        c = min(n - 1, b + BG_EVERY // 2)
        idx = list(range(max(0, c - BG_HALF), min(n, c + BG_HALF + 1), BG_STEP))
        stack = np.asarray(cache[idx])
        k = (len(idx) - 1) // 2
        out.append(np.partition(stack, k, axis=0)[k])
    return out


class Perspective:
    """Expected person height h(y) = a*y + c (pixels at the cache resolution), floor 4 px."""

    def __init__(self, a: float, c: float):
        self.a, self.c = a, c

    def h(self, y):
        return np.maximum(4.0, self.a * np.asarray(y, float) + self.c)

    @classmethod
    def fit(cls, blobs_per_sample, zone_top: float) -> "Perspective":
        """Robust line through (foot_y, height) of upright, person-shaped blobs whose foot is in the
        zone: 75th percentile of height per 16-px band, Theil-Sen slope, horizon kept above the zone."""
        ys, hs = [], []
        for blobs in blobs_per_sample:
            for b in blobs:
                if b["in_zone"] and b["h"] >= 8 and b["h"] >= 1.3 * b["w"] and b["a"] >= 0.3 * b["h"] * b["w"]:
                    ys.append(b["foot_y"])
                    hs.append(b["h"])
        ys, hs = np.array(ys, float), np.array(hs, float)
        pts = []
        for b0 in np.arange(ys.min() // 16 * 16, ys.max() + 1, 16):
            sel = (ys >= b0) & (ys < b0 + 16)
            if sel.sum() >= 6:
                pts.append((b0 + 8, np.percentile(hs[sel], 75)))
        pts = np.array(pts)
        slopes = [(pts[j, 1] - pts[i, 1]) / (pts[j, 0] - pts[i, 0])
                  for i in range(len(pts)) for j in range(i + 1, len(pts))]
        a = float(np.median(slopes)) if slopes else 0.5
        c = float(np.median(pts[:, 1] - a * pts[:, 0]))
        y_h = -c / a if a > 0 else -1e9
        if y_h > zone_top - 8:  # people at the top of the zone must still be > 0 px tall
            y_h = zone_top - 8
            c = -a * y_h
        return cls(a, c)


def perspective_from_boxes(boxes_per_sample: list, H: int, every: int = 12, min_score: float = 0.6,
                           zone_top: float | None = None) -> Perspective | None:
    """h(y) from detector boxes on a sparse subset of samples (every `every`-th): Theil-Sen line through
    (foot_y, box height) in pixels of the cache resolution."""
    ys, hs = [], []
    for i in range(0, len(boxes_per_sample), every):
        for x1, y1, x2, y2, s in boxes_per_sample[i]:
            if s >= min_score and (y2 - y1) > 0 and y2 < 0.995:
                ys.append(y2 * H)
                hs.append((y2 - y1) * H)
    if len(ys) < 8:
        return None
    ys, hs = np.array(ys), np.array(hs)
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(ys), size=(4000, 2))
    dy = ys[pairs[:, 1]] - ys[pairs[:, 0]]
    ok = np.abs(dy) > 10
    a = float(np.median((hs[pairs[ok, 1]] - hs[pairs[ok, 0]]) / dy[ok]))
    c = float(np.median(hs - a * ys))
    if zone_top is not None and a > 0 and -c / a > zone_top - 8:
        c = -a * (zone_top - 8)
    return Perspective(a, c)


def blobs_of(fg: np.ndarray, zone: np.ndarray) -> list[dict]:
    """Connected components of the full-frame foreground; membership by the foot point (bottom centre)."""
    H, W = fg.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    out = []
    for j in range(1, n):
        x, y, w, h, a = (int(v) for v in st[j])
        if a < 6:
            continue
        fy, fx = min(H - 1, y + h - 1), min(W - 1, x + w // 2)
        out.append({"x": x, "y": y, "w": w, "h": h, "a": a, "foot_y": fy, "in_zone": bool(zone[fy, fx])})
    return out


# ----------------------------------------------------------------------------- feature extraction
def compute_features(cache, meta, roi: Roi, persp_override: Perspective | None = None) -> dict[str, np.ndarray]:
    H, W = meta["height"], meta["width"]
    zone = roi.mask(W, H, strict=True)
    zone_top = float(min(y for _, y in roi.zone(strict=True))) * H
    n = len(cache)
    bgs = backgrounds(cache)
    # pass 1: full-frame foreground blobs (not clipped by the zone) for the perspective fit
    fg_full = []
    all_blobs = []
    for i in range(n):
        fgm = majority3(np.abs(cache[i].astype(np.int16) - bgs[i // BG_EVERY]) > FG_T)
        fg_full.append(fgm)
        all_blobs.append(blobs_of(fgm, zone))
    persp = persp_override or Perspective.fit(all_blobs, zone_top)
    ycoords = np.arange(H)[:, None].repeat(W, axis=1)
    h_exp = persp.h(ycoords)
    w_pix = 1.0 / (0.35 * h_exp ** 2)  # person-equivalents per foreground pixel
    f = {k: np.zeros(n) for k in ("framediff_full", "framediff_zone", "fg_zone", "fg_zone_w",
                                  "fg_zone_w_gain", "count_zone", "motion_zone_w",
                                  "mog2_blobs_full", "mog2_blobs_zone", "count_foot", "pe_foot")}
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    prev = None
    for i in range(n):
        img = cache[i]
        bg = bgs[i // BG_EVERY]
        i16 = img.astype(np.int16)
        # proposal idea: frame differencing (1 s apart at 1 Hz)
        if prev is not None:
            d = np.abs(i16 - prev.astype(np.int16)) > DIFF_T
            f["framediff_full"][i] = d.mean()
            f["framediff_zone"][i] = d[zone].mean()
            mv = majority3(d & zone)
            f["motion_zone_w"][i] = float((mv * w_pix).sum())
        prev = img
        # candidate B: perspective-weighted foreground vs median background
        fg = majority3((np.abs(i16 - bg) > FG_T) & zone)
        f["fg_zone"][i] = fg[zone].mean()
        f["fg_zone_w"][i] = float((fg * w_pix).sum())
        g = np.median(img[zone].astype(float)) / max(1.0, np.median(bg[zone].astype(float)))
        fg_g = majority3((np.abs(i16 - np.clip(bg * g, 0, 255)) > FG_T) & zone)
        f["fg_zone_w_gain"][i] = float((fg_g * w_pix).sum())
        # candidate A: person count from blobs with a perspective-consistent size
        nlab, _, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        count = 0.0
        for j in range(1, nlab):
            x, y, w, h, a = st[j]
            foot = y + h - 1
            he = float(persp.h(foot))
            if h < 0.35 * he or a < 0.08 * he * he:
                continue
            count += min(4, max(1, int(round(w / (0.45 * he))))) if w > 0.75 * he else 1
        f["count_zone"][i] = count + min(0.99, f["fg_zone_w"][i] / 100)  # tie-break for ranking
        # A2 / B2: blobs from the unclipped foreground, counted when the foot is in the zone
        cnt, pe = 0.0, 0.0
        for b in all_blobs[i]:
            if not b["in_zone"]:
                continue
            he = float(persp.h(b["foot_y"]))
            pe += min(3.0, b["a"] / (0.3 * he * he))
            if b["h"] < 0.5 * he or b["a"] < 0.12 * he * he:
                continue
            cnt += min(4, max(1, int(round(b["w"] / (0.42 * he))))) if b["w"] > 0.8 * he else 1
        f["count_foot"][i] = cnt + min(0.99, pe / 10)  # tie-break by person-equivalents
        f["pe_foot"][i] = pe
        # proposal idea (grok): MOG2 contours
        m = mog2.apply(img)
        m = cv2.dilate(cv2.erode((m > 200).astype(np.uint8), np.ones((3, 3), np.uint8)), np.ones((3, 3), np.uint8), iterations=2)
        nlab, _, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        big = [j for j in range(1, nlab) if st[j][4] > 22]
        f["mog2_blobs_full"][i] = len(big)
        f["mog2_blobs_zone"][i] = sum(1 for j in big if zone[min(H - 1, int(cen[j][1])), min(W - 1, int(cen[j][0]))])
    f["_perspective"] = np.array([persp.a, persp.c])
    return f


def hog_counts(cache, meta, roi: Roi) -> tuple[np.ndarray, np.ndarray]:
    """Proposal idea (perplexity / claude-simple): OpenCV HOG people detector on the 640x360 frame."""
    H, W = meta["height"], meta["width"]
    zone = roi.mask(W, H, strict=True)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    full, inz = np.zeros(len(cache)), np.zeros(len(cache))
    for i in range(len(cache)):
        rects, weights = hog.detectMultiScale(np.ascontiguousarray(cache[i]), winStride=(8, 8), padding=(8, 8), scale=1.05)
        boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h), s in zip(rects, np.ravel(weights)) if s > 0.5]
        idx = cv2.dnn.NMSBoxes(boxes, [1.0] * len(boxes), 0.0, 0.3) if boxes else []
        keep = [boxes[k] for k in np.ravel(idx)] if len(idx) else []
        full[i] = len(keep)
        inz[i] = sum(1 for x, y, w, h in keep if zone[min(H - 1, y + h - 1), min(W - 1, x + w // 2)])
    return full, inz


# ----------------------------------------------------------------------------- evaluation
def pr_auc(gt: np.ndarray, score: np.ndarray) -> float:
    m = gt >= 0
    g, s = gt[m] == 1, score[m]
    order = np.argsort(-s, kind="stable")
    g = g[order]
    tp = np.cumsum(g)
    prec = tp / np.arange(1, len(g) + 1)
    return float((prec * g).sum() / max(1, g.sum()))  # average precision


def best_threshold(score, gt, sel, p: PostParams) -> float:
    cands = np.unique(score[sel])
    best, best_f = float(cands[-1]) + 1, -1.0
    for c in cands:
        pred = smooth(score >= c, p.smooth_window, p.smooth_need)
        f1 = classifier_prf(gt[sel], pred[sel]).f1
        if f1 > best_f:
            best, best_f = float(c), f1
    return best


def evaluate(name: str, score: np.ndarray, gt: np.ndarray, t: np.ndarray, p: PostParams,
             blocks: int = 6, budget_ratio: float | None = None) -> dict:
    n = len(score)
    edges = np.linspace(0, n, blocks + 1).astype(int)
    cv_pred = np.zeros(n, bool)
    thrs = []
    for b in range(blocks):
        test = np.zeros(n, bool)
        test[edges[b]:edges[b + 1]] = True
        thr = best_threshold(score, gt, ~test, p)
        thrs.append(thr)
        pred = smooth(score >= thr, p.smooth_window, p.smooth_need)
        cv_pred[test] = pred[test]
    cv = classifier_prf(gt, cv_pred)
    res = {"name": name, "ap": round(pr_auc(gt, score), 4), "cv": asdict(cv),
           "cv_f1_ci95": block_bootstrap(gt, cv_pred), "cv_thresholds": [round(x, 4) for x in thrs]}
    # final edit at the median CV threshold (no labels from the scored sample influence it much)
    thr = float(np.median(thrs))
    iv = select(score, t, thr, p)
    res["edit_cvthr"] = asdict(edit_metrics(gt, iv, t, p.pad_before, p.pad_after)) | {"threshold": round(thr, 4)}
    if budget_ratio is not None:
        target = budget_ratio * (t[-1] + 1)
        bthr, biv = threshold_for_budget(score, t, target, p)
        res["edit_budget"] = asdict(edit_metrics(gt, biv, t, p.pad_before, p.pad_after)) | {
            "target_ratio": budget_ratio, "kept_s": round(kept_seconds(biv), 1)}
    # false positives in the second half (sunset, almost no positives)
    half = t >= (t[-1] + 1) / 2
    res["fp_second_half"] = int((cv_pred & half & (gt == 0)).sum())
    return res


def evaluate_auto(scores: dict[str, np.ndarray], gt: np.ndarray, t: np.ndarray, p: PostParams,
                  members: list[str], heavy: str | None, blocks: int = 6) -> dict:
    """Auto mode on held-out blocks: (1) ensemble fitted on the other blocks, (2) cascade that runs the
    heavy scorer only where the light ensemble is uncertain, (3) quick calibration with k simulated
    answers (questions picked from scores only; answers from GT)."""
    from futsalcut.ensemble import Ensemble, pick_questions, quick_calibrate

    n = len(t)
    edges = np.linspace(0, n, blocks + 1).astype(int)
    light = [m for m in members if m != heavy]
    oof = {"ensemble": np.zeros(n), "ensemble_light": np.zeros(n)}
    oof_single = {m: np.zeros(n) for m in members}
    for b in range(blocks):
        test = np.zeros(n, bool)
        test[edges[b]:edges[b + 1]] = True
        train = ~test & (gt >= 0)
        y = (gt[train] == 1).astype(float)
        tr = {k: v[train] for k, v in scores.items()}
        te = {k: v[test] for k, v in scores.items()}
        oof["ensemble"][test] = Ensemble.fit(tr, y, members).prob(te)
        oof["ensemble_light"][test] = Ensemble.fit(tr, y, light).prob(te)
        for m in members:
            oof_single[m][test] = Ensemble.fit(tr, y, [m]).prob(te)  # = Platt-calibrated single scorer
    out: dict = {}

    def summarize(prob: np.ndarray, thr: float = 0.5) -> dict:
        pred = smooth(prob >= thr, p.smooth_window, p.smooth_need)
        iv = select(prob, t, thr, p)
        return {"cv": asdict(classifier_prf(gt, pred)), "cv_f1_ci95": block_bootstrap(gt, pred),
                "edit": asdict(edit_metrics(gt, iv, t, p.pad_before, p.pad_after))}

    for k, v in list(oof.items()) + [(f"single:{m}", v) for m, v in oof_single.items()]:
        out[k] = summarize(v)
    # cascade: heavy scorer only where the light ensemble is uncertain
    if heavy is not None:
        for lo, hi in ((0.05, 0.95), (0.1, 0.9), (0.2, 0.8)):
            unsure = (oof["ensemble_light"] > lo) & (oof["ensemble_light"] < hi)
            prob = np.where(unsure, oof["ensemble"], oof["ensemble_light"])
            out[f"cascade_{lo}_{hi}"] = summarize(prob) | {"heavy_fraction": round(float(unsure.mean()), 4)}
    out["_oof"] = {k: v.round(6).tolist() for k, v in {**oof, **oof_single}.items()}
    # quick calibration on the whole video (probabilities are out-of-fold)
    probs = {"ensemble": oof["ensemble"], **oof_single}
    for k in (8, 12, 20):
        q = pick_questions(probs, t, k=k, min_gap_s=20.0, exclude=gt < 0)
        answers = {i: bool(gt[i] == 1) for i in q}
        name, thr = quick_calibrate(probs, answers, default="ensemble")
        rest = gt.copy()
        rest[q] = -1  # do not score the answered samples
        pred = smooth(probs[name] >= thr, p.smooth_window, p.smooth_need)
        out[f"quick_{k}"] = {"chosen": name, "threshold": round(thr, 4), "positives_asked": int(sum(answers.values())),
                             "cv": asdict(classifier_prf(rest, pred))}
    return out


def intervals_to_score(intervals, t) -> np.ndarray:
    mid = t + 0.5
    s = np.zeros(len(t))
    for a, b in intervals:
        s[(mid >= a) & (mid < b)] = 1.0
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="eval/reports/raw")
    ap.add_argument("--report", default=None)
    ap.add_argument("--skip-hog", action="store_true")
    ap.add_argument("--name", default=None, help="name shown in the report (default: the cache stem)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache, meta = load_cache(args.cache)
    roi = Roi.load(args.roi)
    lab = json.load(open(args.labels, encoding="utf-8"))
    gt = binary_gt(lab["labels"], "strict")
    t = np.asarray(meta["t"], float)
    stem = os.path.basename(args.cache).split(".luma")[0]

    feat_path = os.path.join(args.out, "features.npz")
    if os.path.exists(feat_path):
        feats = dict(np.load(feat_path))
        if not args.skip_hog and "hog_full" not in feats:
            t0 = time.perf_counter()
            feats["hog_full"], feats["hog_zone"] = hog_counts(cache, meta, roi)
            feats["_hog_wall_s"] = np.array([time.perf_counter() - t0])
            np.savez(feat_path, **feats)
    else:
        t0 = time.perf_counter()
        feats = compute_features(cache, meta, roi)
        feats["_feature_wall_s"] = np.array([time.perf_counter() - t0])
        if not args.skip_hog:
            t0 = time.perf_counter()
            feats["hog_full"], feats["hog_zone"] = hog_counts(cache, meta, roi)
            feats["_hog_wall_s"] = np.array([time.perf_counter() - t0])
        np.savez(feat_path, **feats)
    print("perspective (blobs) h(y) = %.3f*y + %.1f" % tuple(feats["_perspective"]))
    # variant: perspective fitted from sparse detector boxes (every 12th sample) -> *_dp features
    dfine_path = os.path.join(args.raw, f"dfine_{stem}.json")
    feat_dp_path = os.path.join(args.out, "features_dp.npz")
    if os.path.exists(dfine_path):
        if os.path.exists(feat_dp_path):
            fdp = dict(np.load(feat_dp_path))
        else:
            zone_top = float(min(y for _, y in roi.zone(strict=True))) * meta["height"]
            pdp = perspective_from_boxes(json.load(open(dfine_path))["boxes"], meta["height"], zone_top=zone_top)
            fdp = compute_features(cache, meta, roi, persp_override=pdp) if pdp else {}
            np.savez(feat_dp_path, **fdp)
        if "_perspective" in fdp:
            print("perspective (detector boxes) h(y) = %.3f*y + %.1f" % tuple(fdp["_perspective"]))
        for k in ("fg_zone_w", "fg_zone_w_gain", "pe_foot", "count_foot", "motion_zone_w"):
            if k in fdp:
                feats[f"{k}_dp"] = fdp[k]

    scores = {k: v for k, v in feats.items() if not k.startswith("_")}
    # external scores
    yolo_path = os.path.join(args.raw, f"yolo_{stem}.json")
    if os.path.exists(yolo_path):
        y = json.load(open(yolo_path))
        zone = roi.zone(strict=True)
        from futsalcut.roi import point_in_polygon
        full, inz = np.zeros(len(t)), np.zeros(len(t))
        for i, boxes in enumerate(y["boxes"]):
            ok = [b for b in boxes if b[4] >= 0.3]
            full[i] = len(ok)
            inz[i] = sum(1 for b in ok if point_in_polygon(zone, (b[0] + b[2]) / 2, b[3]))
        scores["yolo_full"], scores["yolo_zone"] = full, inz
    from futsalcut.roi import point_in_polygon
    zone_poly = roi.zone(strict=True)
    for fn in sorted(os.listdir(args.raw)) if os.path.isdir(args.raw) else []:
        if fn.startswith("dfine") and fn.endswith(f"_{stem}.json"):
            d = json.load(open(os.path.join(args.raw, fn)))
            tag = fn[:-5].replace(f"_{stem}", "")
            hard, soft = np.zeros(len(t)), np.zeros(len(t))
            for i, boxes in enumerate(d["boxes"]):
                inz = [b for b in boxes if point_in_polygon(zone_poly, (b[0] + b[2]) / 2, min(0.9999, b[3]))]
                hard[i] = sum(1 for b in inz if b[4] >= 0.4)
                soft[i] = sum(b[4] for b in inz)
            scores[f"{tag}_count"] = hard + np.minimum(0.99, soft / 10)
            scores[f"{tag}_soft"] = soft
            # proposal idea (chatgpt): person count over the whole frame, no region of interest
            scores[f"{tag}_full_count"] = np.array([sum(1 for b in boxes if b[4] >= 0.4) for boxes in d["boxes"]], float)
            scores[f"_{tag}_wall_ms"] = np.array([d.get("ms_per_sample", 0)])
        if fn.startswith(f"dvrscan_{stem}") and fn.endswith(".json"):
            d = json.load(open(os.path.join(args.raw, fn)))
            scores[fn[:-5].replace(f"_{stem}", "")] = intervals_to_score(d["intervals"], t)
    scores = {k: v for k, v in scores.items() if not k.startswith("_")}
    rng = np.random.default_rng(0)
    scores["anchor_random"] = rng.random(len(t))
    scores["anchor_keep_all"] = np.ones(len(t))

    p = PostParams()
    gt_ratio = float((gt == 1).sum() / (gt >= 0).sum())
    results = [evaluate(k, v, gt, t, p, budget_ratio=round(1.5 * gt_ratio, 4)) for k, v in sorted(scores.items())]
    results.sort(key=lambda r: -r["cv"]["f1"])
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"gt_ratio": gt_ratio, "post": asdict(p), "results": results}, f, indent=1)
    lines = [f"# Tournament: {args.name or stem}", "",
             f"GT (strict) positives {int((gt == 1).sum())} / scored {int((gt >= 0).sum())} "
             f"(ratio {gt_ratio:.1%}); post-processing {asdict(p)}; budget = 1.5 x GT ratio.", "",
             "| candidate | AP | CV P | CV R | CV F1 [95% CI] | FP 2nd half | edit P(band) | edit R | event R | false scenes | kept | budget edit F1 | budget event R |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        e, b = r["edit_cvthr"], r.get("edit_budget", {})
        lines.append(
            f"| {r['name']} | {r['ap']:.3f} | {r['cv']['precision']:.2f} | {r['cv']['recall']:.2f} | "
            f"{r['cv']['f1']:.2f} [{r['cv_f1_ci95'][0]:.2f}, {r['cv_f1_ci95'][1]:.2f}] | {r['fp_second_half']} | "
            f"{e['prf_band']['precision']:.2f} | {e['prf_band']['recall']:.2f} | {e['event_recall']:.2f} | "
            f"{e['false_scene_rate']:.2f} | {e['kept_ratio']:.1%} | "
            f"{b.get('prf_band', {}).get('f1', 0):.2f} | {b.get('event_recall', 0):.2f} |")
    # auto mode (multi-algorithm): ensemble / cascade / quick calibration
    pool = ("fg_zone_w_gain_dp", "pe_foot_dp", "count_foot_dp", "framediff_zone", "dfine_count") \
        if "pe_foot_dp" in scores else ("fg_zone_w_gain", "pe_foot", "count_foot", "framediff_zone", "dfine_count")
    members = [m for m in pool if m in scores]
    heavy = "dfine_count" if "dfine_count" in scores else None
    auto = evaluate_auto(scores, gt, t, p, members, heavy)
    with open(os.path.join(args.out, "auto.json"), "w") as f:
        json.dump({"members": members, "heavy": heavy, "results": auto}, f, indent=1)
    lines += ["", f"## Auto mode (members: {', '.join(members)}; heavy: {heavy})", "",
              "| strategy | CV P | CV R | CV F1 [95% CI] | edit P(band) | edit R | event R | false scenes | kept | heavy share |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in auto.items():
        if k.startswith("quick_") or k.startswith("_"):
            continue
        e = v["edit"]
        lines.append(f"| {k} | {v['cv']['precision']:.2f} | {v['cv']['recall']:.2f} | {v['cv']['f1']:.2f} "
                     f"[{v['cv_f1_ci95'][0]:.2f}, {v['cv_f1_ci95'][1]:.2f}] | {e['prf_band']['precision']:.2f} | "
                     f"{e['prf_band']['recall']:.2f} | {e['event_recall']:.2f} | {e['false_scene_rate']:.2f} | "
                     f"{e['kept_ratio']:.1%} | {v.get('heavy_fraction', '')} |")
    lines += ["", "| quick calibration | chosen | threshold | positives asked | P | R | F1 |", "|---|---|---|---|---|---|---|"]
    for k, v in auto.items():
        if k.startswith("quick_"):
            lines.append(f"| {k} | {v['chosen']} | {v['threshold']} | {v['positives_asked']} | "
                         f"{v['cv']['precision']:.2f} | {v['cv']['recall']:.2f} | {v['cv']['f1']:.2f} |")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
