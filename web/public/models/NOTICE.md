# Models

## dfine_n_coco.onnx
- Source: [ustc-community/dfine-nano-coco](https://huggingface.co/ustc-community/dfine-nano-coco) at revision
  `066438d3d8f0da137a37b38fdf3368fd4afceced` ("Fix weights", 2025-08-20), COCO-trained D-FINE-N.
- License: Apache-2.0 (D-FINE: https://github.com/Peterande/D-FINE).
- Exported to ONNX (opset 17) by `eval/runners/export_dfine.py`, then constant-folded by
  `eval/runners/fold_dfine.py` (onnxruntime-web lacks the double-precision Casts of the position embedding). Input `pixel_values [1,3,640,640]` (RGB/255),
  outputs `logits [1,300,80]`, `pred_boxes [1,300,4]` (cx, cy, w, h normalized). Only class 0 (person) is used.

## ensemble_v1.json
- Calibration and ensemble weights fitted by `eval/fit_model.py` on the project's labeled recordings.
