"""Candidate C: D-FINE-N (Apache-2.0) person detections per 1 Hz sample, via onnxruntime (CPU).

Modes:
  thumbs - the 640x360 colour thumbnails of the sampler cache, resized to 640x640 (as the HF processor does)
  tiles  - keyframes decoded again at full resolution; the counting zone's bounding box is cut into
           square-ish tiles (overlap 15%) so distant players are larger in the 640x640 input
Output JSON: {"boxes": [[[x1, y1, x2, y2, score], ...] per sample]} in normalized display coordinates.

python eval/runners/detect_dfine.py --model /data/models/dfine_n_coco.onnx --cache /data/cache/<stem>.luma640x360.u8 \
    --out eval/reports/raw/dfine_<stem>.json [--mode tiles --video /src/<video> --roi eval/labels/<stem>.roi.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

PERSON = 0
MIN_SCORE = 0.15


def preprocess(rgb: np.ndarray) -> np.ndarray:
    x = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def run(sess, rgb: np.ndarray) -> list[list[float]]:
    logits, boxes = sess.run(None, {"pixel_values": preprocess(rgb)})
    prob = 1.0 / (1.0 + np.exp(-logits[0, :, PERSON]))
    out = []
    for q in np.where(prob >= MIN_SCORE)[0]:
        cx, cy, w, h = boxes[0, q]
        out.append([float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2), float(prob[q])])
    return out


def nms(boxes: list[list[float]], iou: float = 0.6) -> list[list[float]]:
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep: list[list[float]] = []
    for b in boxes:
        ok = True
        for k in keep:
            ix = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
            inter = ix * iy
            union = (b[2] - b[0]) * (b[3] - b[1]) + (k[2] - k[0]) * (k[3] - k[1]) - inter
            if union > 0 and inter / union > iou:
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


def tiles_for(bbox: tuple[float, float, float, float], W: int, H: int, tile_px: int, overlap: float = 0.15):
    x0, y0, x1, y1 = int(bbox[0] * W), int(bbox[1] * H), int(np.ceil(bbox[2] * W)), int(np.ceil(bbox[3] * H))
    step = int(tile_px * (1 - overlap))
    out = []
    for ty in range(y0, max(y0 + 1, y1 - int(tile_px * overlap)), step):
        for tx in range(x0, max(x0 + 1, x1 - int(tile_px * overlap)), step):
            out.append((tx, ty, min(W, tx + tile_px), min(H, ty + tile_px)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["thumbs", "tiles"], default="thumbs")
    ap.add_argument("--video", default=None)
    ap.add_argument("--roi", default=None)
    ap.add_argument("--tile-px", type=int, default=1280, help="tile size at source resolution")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    with open(args.cache[:-3] + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    n = len(meta["t"])
    stem = os.path.basename(args.cache).split(".luma")[0]
    t0 = time.perf_counter()
    result = []
    if args.mode == "thumbs":
        tdir = os.path.join(os.path.dirname(args.cache), f"{stem}.thumbs")
        for i in range(n):
            bgr = cv2.imread(os.path.join(tdir, f"{i:06d}.jpg"))
            result.append(run(sess, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    else:
        import av
        from futsalcut.roi import Roi
        from futsalcut.sampler import display_rotation, iter_grid_frames, rotate_cw
        zone = Roi.load(args.roi).zone(strict=True)
        xs, ys = [p[0] for p in zone], [p[1] for p in zone]
        # include the band above the zone top so players standing on the boundary keep their heads
        bbox = (max(0.0, min(xs)), max(0.0, min(ys) - 0.08), min(1.0, max(xs)), min(1.0, max(ys)))
        rot = display_rotation(args.video)
        for frame, _ in iter_grid_frames(args.video, "packets", 2):
            rgb = rotate_cw(frame.to_ndarray(format="rgb24"), rot)
            H, W = rgb.shape[:2]
            dets = []
            for (tx0, ty0, tx1, ty1) in tiles_for(bbox, W, H, args.tile_px):
                for x1, y1, x2, y2, s in run(sess, rgb[ty0:ty1, tx0:tx1]):
                    tw, th = tx1 - tx0, ty1 - ty0
                    dets.append([(tx0 + x1 * tw) / W, (ty0 + y1 * th) / H, (tx0 + x2 * tw) / W, (ty0 + y2 * th) / H, s])
            result.append(nms(dets))
            if len(result) % 50 == 0:
                print(f"{len(result)}/{n} samples, {time.perf_counter() - t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": os.path.basename(args.model), "mode": args.mode, "wall_s": round(wall, 1),
                   "ms_per_sample": round(wall / max(1, len(result)) * 1000, 1), "boxes": result}, f)
    print(json.dumps({"samples": len(result), "wall_s": round(wall, 1), "ms_per_sample": round(wall / max(1, len(result)) * 1000, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
