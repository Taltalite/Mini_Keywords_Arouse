#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-outputs/eval_smoke}"
VALIDATION_DIR="$RUN_DIR/eval_validation"
TESTING_DIR="$RUN_DIR/eval_testing"

if [[ ! -f "$RUN_DIR/best.pt" ]]; then
  python -m src.train --config configs/mka_smallcnn.yaml --limit 200 --epochs 1 --output "$RUN_DIR"
fi

python -m src.eval --ckpt "$RUN_DIR/best.pt" --subset validation --limit 100 --output "$VALIDATION_DIR"
python -m src.eval --ckpt "$RUN_DIR/best.pt" --subset testing --limit 100 --output "$TESTING_DIR"
