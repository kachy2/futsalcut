"""Make the D-FINE ONNX model loadable by onnxruntime-web: constant-fold the input-size-dependent
subgraphs (position embeddings use double-precision Casts that the web runtime does not implement).
Uses only standard ONNX ops (ORT_ENABLE_BASIC). Verifies the folded model against the original.

python eval/runners/fold_dfine.py /data/models/dfine_n_coco.onnx /data/models/dfine_n_coco_web.onnx
"""
from __future__ import annotations

import sys

import numpy as np
import onnxruntime as ort


def main() -> int:
    src, dst = sys.argv[1:3]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = dst
    ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
    a = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, 640, 640), dtype=np.float32)
    la, ba = a.run(None, {"pixel_values": x})
    lb, bb = b.run(None, {"pixel_values": x})
    print("max |logits diff|", float(np.abs(la - lb).max()), " max |boxes diff|", float(np.abs(ba - bb).max()))
    # list remaining Cast nodes and double tensors
    try:
        import onnx
        m = onnx.load(dst)
        casts = [(n.name, [a.i for a in n.attribute if a.name == "to"]) for n in m.graph.node if n.op_type == "Cast"]
        doubles = [t.name for t in m.graph.initializer if t.data_type == onnx.TensorProto.DOUBLE]
        print("Cast nodes:", len(casts), "to-double casts:", [c for c in casts if 11 in c[1]][:5], "double initializers:", doubles[:5])
        ops = sorted({n.op_type for n in m.graph.node})
        print("ops:", ops)
    except ImportError:
        print("(onnx package not installed; skipping graph inspection)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
