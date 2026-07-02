#!/usr/bin/env bash
# Ketchup right-hand SPIDER pipeline: setup → workspace → E2E + video.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Step 0: SPIDER + LFS ==="
bash scripts/setup_spider.sh

echo ""
echo "=== Step 2: Build right-hand workspace ==="
python3 scripts/build_spider_ketchup_right.py

echo ""
echo "=== Step 3: Replay + 2s extend (mimic last 1s, wrist +10cm) ==="
python3 scripts/run_spider_e2e.py \
  --ketchup-right \
  --extend 2 \
  --mimic-last 1 \
  --lift 0.10

echo ""
echo "Done. Outputs under data/spider_e2e/ and data/spider_ketchup_right/"
