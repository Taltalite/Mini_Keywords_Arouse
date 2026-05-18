#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="outputs/train_smoke_$(date +%Y%m%d_%H%M%S)"

python scripts/check_data.py --limit 100
python -m src.train --config configs/mka_smallcnn.yaml --limit 200 --epochs 1 --output "$RUN_DIR"
python -m src.eval --ckpt "$RUN_DIR/best.pt" --subset validation --limit 100
