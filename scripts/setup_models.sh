#!/usr/bin/env bash
# Fetch simulation assets: Shadow Hand (Menagerie) + XHAND1 URDF (worldstring).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MENAGERIE="$ROOT/third_party/mujoco_menagerie"
WORLDSTRING="$ROOT/third_party/worldstring"
XHAND_MESHES="$ROOT/models/xhand/meshes"
XHAND_URDF_SRC="$WORLDSTRING/assets/xhand_right/urdf/xhand_right.urdf"

mkdir -p "$ROOT/third_party" "$ROOT/models/xhand"

# --- Shadow Hand (MuJoCo Menagerie) ---
if [ ! -d "$MENAGERIE/shadow_hand/right_hand.xml" ]; then
  echo "Cloning MuJoCo Menagerie (shadow_hand only)..."
  cd "$ROOT/third_party"
  if [ ! -d mujoco_menagerie ]; then
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/google-deepmind/mujoco_menagerie.git
  fi
  cd mujoco_menagerie
  git sparse-checkout set shadow_hand
  git checkout
  echo "Shadow Hand ready at $MENAGERIE/shadow_hand"
else
  echo "Shadow Hand already present"
fi

# --- XHAND1 URDF + meshes (worldstring) ---
if [ ! -f "$XHAND_URDF_SRC" ]; then
  echo "Cloning worldstring (xhand_right assets only)..."
  cd "$ROOT/third_party"
  if [ ! -d worldstring ]; then
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/MaureenZOU/worldstring.git
  fi
  cd worldstring
  git sparse-checkout set assets/xhand_right
  git checkout
else
  echo "worldstring xhand assets already present"
fi

if [ ! -e "$XHAND_MESHES" ]; then
  ln -sfn "$WORLDSTRING/assets/xhand_right/meshes" "$XHAND_MESHES"
fi

URDF_DST="$ROOT/models/xhand/xhand_right.urdf"
if [ -f "$XHAND_URDF_SRC" ]; then
  sed 's|package://xhand_right/meshes/|meshes/|g' "$XHAND_URDF_SRC" > "$URDF_DST"
fi

# Build sim MJCF from URDF if base xml missing
if [ -f "$URDF_DST" ] && [ ! -f "$ROOT/models/xhand/xhand_right.xml" ]; then
  echo "Converting XHAND URDF → MJCF (urdf2mjcf)..."
  ~/.local/bin/urdf2mjcf "$URDF_DST" --output "$ROOT/models/xhand/xhand_right.xml" --copy-meshes 2>/dev/null \
    || python3 -m urdf2mjcf "$URDF_DST" --output "$ROOT/models/xhand/xhand_right.xml" --copy-meshes
fi

echo "Building XHAND sim MJCF..."
python3 "$ROOT/scripts/build_xhand_mjcf.py"

echo "Done."
echo "  Shadow Hand : $MENAGERIE/shadow_hand"
echo "  XHAND URDF  : $URDF_DST"
echo "  XHAND MJCF  : $ROOT/models/xhand/xhand_right_sim.xml"
echo "  Scene       : $ROOT/models/scenes/xhand_grasp_scene.xml"
