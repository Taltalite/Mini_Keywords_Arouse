#!/usr/bin/env bash
set -euo pipefail

python scripts/check_data.py --config configs/mka_smallcnn.yaml --limit 1000
