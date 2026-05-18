#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-outputs/eval_smoke}"

if [[ ! -f "$RUN_DIR/best.pt" ]]; then
  python -m src.train --config configs/mka_smallcnn.yaml --limit 200 --epochs 1 --output "$RUN_DIR"
fi

python -m src.eval --ckpt "$RUN_DIR/best.pt" --subset validation --limit 100
python -m src.eval --ckpt "$RUN_DIR/best.pt" --subset testing --limit 100
