"""Export D-FINE-N (Apache-2.0, ustc-community/dfine-nano-coco, weights fixed 2025-08-20) to ONNX and
check it against the PyTorch model. Runs in a throwaway container with torch + transformers:

  docker run --rm -v ~/futsalcut:/work -v ~/futsalcut_data:/data -w /work futsalcut-py bash -c \
    "pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 && \
     pip install -q transformers==5.18.0 onnx==1.23.1 onnxruntime==1.29.0 && \
     python eval/runners/export_dfine.py /data/models"
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = "ustc-community/dfine-nano-coco"
REVISION = "066438d3d8f0da137a37b38fdf3368fd4afceced"  # "Fix weights (#1)", 2025-08-20


def main() -> int:
    import torch
    from transformers import AutoModelForObjectDetection

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/models"
    os.makedirs(out_dir, exist_ok=True)
    model = AutoModelForObjectDetection.from_pretrained(REPO, revision=REVISION).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            o = self.m(pixel_values=pixel_values)
            return o.logits, o.pred_boxes

    x = torch.rand(1, 3, 640, 640)
    path = os.path.join(out_dir, "dfine_n_coco.onnx")
    torch.onnx.export(Wrap(model), (x,), path, input_names=["pixel_values"], output_names=["logits", "pred_boxes"],
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    lo, bo = sess.run(None, {"pixel_values": x.numpy()})
    with torch.no_grad():
        lt, bt = Wrap(model)(x)
    meta = {"repo": REPO, "revision": REVISION, "license": "apache-2.0", "opset": 17,
            "input": "pixel_values [1,3,640,640] RGB/255, no mean/std (RTDetrImageProcessor do_normalize=false)",
            "outputs": "logits [1,300,80] (sigmoid), pred_boxes [1,300,4] cx,cy,w,h normalized",
            "max_abs_diff_logits": float(np.abs(lo - lt.numpy()).max()),
            "max_abs_diff_boxes": float(np.abs(bo - bt.numpy()).max()),
            "bytes": os.path.getsize(path)}
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
