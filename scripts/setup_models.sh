#!/usr/bin/env bash
# Fetch Shadow Hand model from MuJoCo Menagerie (sparse checkout).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/mujoco_menagerie"

if [ -d "$DEST/shadow_hand/right_hand.xml" ]; then
  echo "Shadow Hand model already present at $DEST/shadow_hand"
  exit 0
fi

echo "Cloning MuJoCo Menagerie (shadow_hand only)..."
mkdir -p "$ROOT/third_party"
cd "$ROOT/third_party"

if [ ! -d mujoco_menagerie ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git
fi

cd mujoco_menagerie
git sparse-checkout set shadow_hand
git checkout

echo "Done. Model at: $DEST/shadow_hand/right_hand.xml"
echo "Free-base hand: $ROOT/models/shadow_hand/right_hand_free.xml"
