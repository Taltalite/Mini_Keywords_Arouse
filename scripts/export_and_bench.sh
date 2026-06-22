#!/usr/bin/env bash
set -euo pipefail

CKPT="${CKPT:-outputs/best.pt}"
ONNX_FP32="${ONNX_FP32:-outputs/model.onnx}"
ONNX_INT8="${ONNX_INT8:-outputs/model_int8.onnx}"

python -m src.export_onnx \
  --ckpt "$CKPT" \
  --out "$ONNX_FP32"

python -m src.quantize_onnx \
  --input "$ONNX_FP32" \
  --output "$ONNX_INT8"

python -m src.benchmark_latency \
  --ckpt "$CKPT" \
  --onnx "$ONNX_FP32" \
  --onnx-int8 "$ONNX_INT8"
