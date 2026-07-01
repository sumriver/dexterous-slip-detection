"""MuJoCo model loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_scene(scene_rel: str = "models/scenes/bottle_grasp_scene.xml") -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load MJCF scene from project root."""
    root = project_root()
    scene_path = (root / scene_rel).resolve()

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    if "xhand" in scene_rel:
        hand_xml = root / "models/xhand/xhand_right_sim.xml"
        mesh_dir = root / "models/xhand/meshes"
        if not hand_xml.exists():
            raise FileNotFoundError(
                f"XHAND model missing at {hand_xml}. Run: bash scripts/setup_models.sh"
            )
        if not mesh_dir.exists():
            raise FileNotFoundError(f"XHAND meshes missing at {mesh_dir}. Run: bash scripts/setup_models.sh")
    else:
        hand_dir = root / "third_party/mujoco_menagerie/shadow_hand"
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


def load_xhand_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    return load_scene("models/scenes/xhand_grasp_scene.xml")
