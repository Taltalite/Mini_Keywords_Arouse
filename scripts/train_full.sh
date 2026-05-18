#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="outputs/train_full_$(date +%Y%m%d_%H%M%S)"
EVAL_DIR="$RUN_DIR/eval_validation"

python -m src.train --config configs/mka_smallcnn.yaml --limit 2000 --epochs 2 --output "$RUN_DIR"
python -m src.eval --ckpt "$RUN_DIR/best.pt" --subset validation --limit 1000 --output "$EVAL_DIR"
