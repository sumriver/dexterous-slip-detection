"""MuJoCo model loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_scene(scene_rel: str = "models/scenes/bottle_grasp_scene.xml") -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load MJCF scene, resolving Shadow Hand mesh paths correctly."""
    root = project_root()
    scene_path = (root / scene_rel).resolve()
    hand_dir = root / "third_party/mujoco_menagerie/shadow_hand"

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene not found: {scene_path}")
    if not (hand_dir / "right_hand.xml").exists():
        raise FileNotFoundError(
            f"Shadow Hand model missing at {hand_dir}. Run: bash scripts/setup_models.sh"
        )
    if not (root / "models/shadow_hand/right_hand_free.xml").exists():
        raise FileNotFoundError("Missing models/shadow_hand/right_hand_free.xml")

    cwd = os.getcwd()
    try:
        os.chdir(root)
        model = mujoco.MjModel.from_xml_path(str(scene_path.relative_to(root)))
    finally:
        os.chdir(cwd)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data
