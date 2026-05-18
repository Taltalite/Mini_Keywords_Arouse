#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f outputs/best.pt ]]; then
  python -m src.train --config configs/mka_smallcnn.yaml --limit 200 --epochs 1
fi

python -m src.eval --ckpt outputs/best.pt --subset validation --limit 100
python -m src.eval --ckpt outputs/best.pt --subset testing --limit 100
