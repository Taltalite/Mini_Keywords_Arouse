#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/mka_smallcnn.yaml --limit 2000 --epochs 2
python -m src.eval --ckpt outputs/best.pt --subset validation --limit 1000
