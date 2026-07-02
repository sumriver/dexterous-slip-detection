"""Derive right-hand-only ketchup workspace from SPIDER bimanual Arctic scene."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPIDER = ROOT / "third_party" / "spider"
BI_SCENE = (
    SPIDER
    / "example_datasets/processed/arcticv2/xhand/bimanual/s01-ketchup_use_01/scene.xml"
)
BI_TRAJ = (
    SPIDER
    / "example_datasets/processed/arcticv2/xhand/bimanual/s01-ketchup_use_01/0/trajectory_mjwp_fast.npz"
)
ARCTIC_ASSETS = SPIDER / "example_datasets/processed/arcticv2/assets"
DEFAULT_WORKSPACE = ROOT / "data/spider_ketchup_right"

# Bimanual SPIDER layout: right arm+hand, left arm+hand, right object, left object.
_BI_QPOS_RIGHT = slice(0, 18)
_BI_QPOS_OBJECT = slice(36, 43)
_BI_QVEL_RIGHT = slice(0, 18)
_BI_QVEL_OBJECT = slice(36, 42)
_BI_CTRL_RIGHT = slice(0, 18)

_LEFT_BODY_NAMES = {
    "L_forearm_ty_link",
    "left_object",
}
_LEFT_REF_PREFIXES = (
    "ref_object_left_",
    "ref_hand_left_",
)


def _skip_body_block(lines: list[str], start: int) -> int:
    """Return index after closing tag for body starting at start."""
    depth = 0
    i = start
    while i < len(lines):
        line = lines[i]
        if "<body " in line or line.strip() == "<body>":
            depth += 1
        if "</body>" in line:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(lines)


def strip_bimanual_scene_to_right(scene_xml: str) -> str:
    """Remove left hand, left object, and left-related pairs from bimanual scene."""
    lines = scene_xml.splitlines()
    out: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("<mesh ") and "left_hand" in stripped:
            i += 1
            continue
        if stripped.startswith("<texture ") and "left_groundplane" in stripped:
            i += 1
            continue
        if stripped.startswith("<material ") and "left_groundplane" in stripped:
            i += 1
            continue

        body_m = re.search(r'<body name="([^"]+)"', stripped)
        if body_m:
            name = body_m.group(1)
            if name in _LEFT_BODY_NAMES or any(
                name.startswith(p) for p in _LEFT_REF_PREFIXES
            ):
                i = _skip_body_block(lines, i)
                continue

        if stripped.startswith("<site ") and "track_object_left_" in stripped:
            i += 1
            continue

        if stripped.startswith("<pair ") and (
            "collision_hand_left_" in stripped
            or "geom2=\"left_" in stripped
            or "geom1=\"collision_hand_left_" in stripped
            or "_collision_hand_left_" in stripped
            or "collision_hand_right_" in stripped and "collision_hand_left_" in stripped
        ):
            i += 1
            continue

        if stripped.startswith("<general ") and (
            stripped.startswith('<general name="L_')
            or 'joint="left_hand' in stripped
            or 'joint="L_' in stripped
        ):
            i += 1
            continue

        out.append(line)
        i += 1

    text = "\n".join(out)
    text = text.replace('<mujoco model="bimanual">', '<mujoco model="xhand_right_ketchup">')
    text = re.sub(
        r'\s*<material name="left_groundplane"[^/]*/>\n',
        "\n",
        text,
    )
    return text + "\n"


def slice_bimanual_trajectory(
    traj_path: Path,
    data_type: str = "mjwp_fast",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract right-hand + ketchup object DOFs from bimanual trajectory."""
    raw = np.load(traj_path)
    if data_type == "mjwp_fast":
        attempt = 0
        if "rew_mean" in raw:
            attempt = int(np.argmax(raw["rew_mean"].sum(axis=1)))
        elif "succeeded" in raw:
            succ = raw["succeeded"].reshape(-1)
            attempt = int(np.argmax(succ)) if succ.any() else 0
        qpos = raw["qpos"][attempt]
        qvel = raw["qvel"][attempt]
        ctrl = raw["ctrl"][attempt]
    else:
        qpos = raw["qpos"]
        qvel = raw["qvel"]
        ctrl = raw["ctrl"]

    qpos_r = np.concatenate([qpos[:, _BI_QPOS_RIGHT], qpos[:, _BI_QPOS_OBJECT]], axis=1)
    qvel_r = np.concatenate([qvel[:, _BI_QVEL_RIGHT], qvel[:, _BI_QVEL_OBJECT]], axis=1)
    ctrl_r = ctrl[:, _BI_CTRL_RIGHT]
    return qpos_r, qvel_r, ctrl_r


def save_right_trajectory(
    src_traj: Path,
    dst_traj: Path,
    data_type: str = "mjwp_fast",
) -> Path:
    """Write right-only trajectory npz (mjwp_fast schema)."""
    qpos, qvel, ctrl = slice_bimanual_trajectory(src_traj, data_type)
    dst_traj.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst_traj,
        qpos=qpos[None, ...],
        qvel=qvel[None, ...],
        ctrl=ctrl[None, ...],
        rew_mean=np.ones((1, qpos.shape[0])),
        succeeded=np.array([True]),
    )
    return dst_traj


def build_right_hand_workspace(
    out_dir: Path | None = None,
    *,
    src_scene: Path = BI_SCENE,
    src_traj: Path = BI_TRAJ,
    meshdir: Path = ARCTIC_ASSETS,
) -> Path:
    """Copy ketchup pick into our workspace as right-hand-only scene + trajectory."""
    out_dir = out_dir or DEFAULT_WORKSPACE
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_text = strip_bimanual_scene_to_right(src_scene.read_text())
    scene_text = re.sub(
        r'<compiler([^>]*)\smeshdir="[^"]*"',
        f'<compiler\\1 meshdir="{meshdir.as_posix()}/"',
        scene_text,
        count=1,
    )
    (out_dir / "scene.xml").write_text(scene_text)
    save_right_trajectory(src_traj, out_dir / "trajectory_mjwp_fast.npz")
    return out_dir
