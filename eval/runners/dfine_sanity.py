"""Draw D-FINE person boxes (score >= 0.4) on a few thumbnails to sanity-check the ONNX export.

python eval/runners/dfine_sanity.py /data/models/dfine_n_coco.onnx /data/cache/<stem>.thumbs out.jpg 320 337 100
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_dfine import run  # noqa: E402


def main() -> int:
    model, tdir, out = sys.argv[1:4]
    idx = [int(x) for x in sys.argv[4:]]
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    tiles = []
    for i in idx:
        bgr = cv2.imread(os.path.join(tdir, f"{i:06d}.jpg"))
        t0 = time.perf_counter()
        boxes = run(sess, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ms = (time.perf_counter() - t0) * 1000
        h, w = bgr.shape[:2]
        n = 0
        for x1, y1, x2, y2, s in boxes:
            if s < 0.4:
                continue
            n += 1
            cv2.rectangle(bgr, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 255), 1)
            cv2.putText(bgr, f"{s:.2f}", (int(x1 * w), int(y1 * h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        cv2.putText(bgr, f"#{i} persons>=0.4: {n}  {ms:.0f} ms", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(bgr)
        print(i, "persons>=0.4:", n, "all>=0.15:", len(boxes), f"{ms:.0f} ms")
    cv2.imwrite(out, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return 0


if __name__ == "__main__":
    sys.exit(main())
