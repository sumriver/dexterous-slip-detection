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

echo "Pulling LFS for oakinkv2 XHAND pick_spoon_bowl (E2E default)..."
(cd "$DATA" && git lfs fetch --include="processed/oakinkv2/xhand/right/pick_spoon_bowl/**" --include="processed/oakinkv2/assets/objects/O02_0030_00002/**")
(cd "$DATA" && git lfs checkout processed/oakinkv2/xhand/right/pick_spoon_bowl/0/trajectory_mjwp_fast.npz)
(cd "$DATA" && git lfs checkout processed/oakinkv2/assets/objects/O02_0030_00002/**)

# oakinkv2 xhand STLs often stay as LFS pointers; reuse gigahand meshes (same robot)
OAK_XHAND="$DATA/processed/oakinkv2/assets/robots/xhand/assets"
GIGA_XHAND="$DATA/processed/gigahand/assets/robots/xhand/assets"
if [ -d "$GIGA_XHAND" ] && [ ! -L "$OAK_XHAND" ]; then
  mkdir -p "$(dirname "$OAK_XHAND")"
  rm -rf "$OAK_XHAND"
  ln -s "$GIGA_XHAND" "$OAK_XHAND"
  echo "Linked oakinkv2 xhand meshes -> gigahand assets"
fi

echo "Pulling LFS for arcticv2 XHAND ketchup (s01-ketchup_use_01)..."
(cd "$DATA" && git lfs fetch --include="processed/arcticv2/xhand/bimanual/s01-ketchup_use_01/**" --include="processed/arcticv2/assets/objects/ketchup_bottom/**")
(cd "$DATA" && git lfs checkout processed/arcticv2/xhand/bimanual/s01-ketchup_use_01/0/trajectory_mjwp_fast.npz)
(cd "$DATA" && git lfs checkout processed/arcticv2/xhand/bimanual/s01-ketchup_use_01/0/visualization_mjwp_fast.mp4)
(cd "$DATA" && git lfs checkout processed/arcticv2/assets/objects/ketchup_bottom/**)

echo "SPIDER ready."
echo "  Full ketchup pipeline:  bash scripts/run_ketchup_pipeline.sh"
echo "  E2E (pick_spoon_bowl):  python3 scripts/run_spider_e2e.py --copy-official-video"
echo "  E2E (ketchup bimanual): python3 scripts/run_spider_e2e.py --dataset arcticv2 --task s01-ketchup_use_01 --embodiment bimanual"
echo "  E2E (ketchup right):    python3 scripts/run_spider_e2e.py --ketchup-right --extend 2 --lift 0.10"
echo "  Tea demo replay:        python3 scripts/run_spider_xhand_demo.py"
