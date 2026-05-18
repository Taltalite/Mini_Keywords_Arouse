#!/usr/bin/env bash
set -euo pipefail

python scripts/check_data.py --limit 100
python -m src.train --config configs/mka_smallcnn.yaml --limit 200 --epochs 1
python -m src.eval --ckpt outputs/best.pt --subset validation --limit 100
