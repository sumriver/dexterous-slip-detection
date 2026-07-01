#!/usr/bin/env bash
# Fetch SPIDER + example dataset (XHAND demos). Requires git-lfs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPIDER="$ROOT/third_party/spider"

if ! command -v git-lfs >/dev/null; then
  echo "Installing git-lfs..."
  sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs
fi
git lfs install

if [ ! -d "$SPIDER/.git" ]; then
  echo "Cloning SPIDER..."
  git clone --depth 1 https://github.com/facebookresearch/spider "$SPIDER"
fi

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing SPIDER deps (uv sync)..."
(cd "$SPIDER" && uv sync)

DATA="$SPIDER/example_datasets"
if [ ! -d "$DATA/.git" ]; then
  echo "Cloning example_datasets (metadata first)..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/datasets/retarget/retarget_example "$DATA"
fi

echo "Pulling LFS assets for XHAND p36-tea demo..."
(cd "$DATA" && git lfs pull -I "processed/gigahand/xhand/bimanual/p36-tea/0/*")
(cd "$DATA" && git lfs pull -I "processed/gigahand/assets/robots/xhand/assets/*.STL")
(cd "$DATA" && git lfs pull -I "processed/gigahand/assets/objects/p36-tea/**")

echo "SPIDER ready. Run: python scripts/run_spider_xhand_demo.py"
