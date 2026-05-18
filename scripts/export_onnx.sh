#!/usr/bin/env bash
set -euo pipefail

python -m src.export_onnx \
  --ckpt outputs/best.pt \
  --config configs/mka_smallcnn.yaml \
  --out outputs/model.onnx
