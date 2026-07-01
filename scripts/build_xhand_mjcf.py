#!/usr/bin/env python3
"""Post-process urdf2mjcf output into a MuJoCo sim-ready XHAND model."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models/xhand/xhand_right.xml"
DST = ROOT / "models/xhand/xhand_right_sim.xml"

JOINT_RANGES: dict[str, tuple[float, float]] = {
    "right_hand_thumb_bend_joint": (0.0, 1.832),
    "right_hand_thumb_rota_joint1": (-0.698, 1.57),
    "right_hand_thumb_rota_joint2": (0.0, 1.57),
    "right_hand_index_bend_joint": (-0.174, 0.174),
    "right_hand_index_joint1": (0.0, 1.919),
    "right_hand_index_joint2": (0.0, 1.919),
    "right_hand_mid_joint1": (0.0, 1.919),
    "right_hand_mid_joint2": (0.0, 1.919),
    "right_hand_ring_joint1": (0.0, 1.919),
    "right_hand_ring_joint2": (0.0, 1.919),
    "right_hand_pinky_joint1": (0.0, 1.919),
    "right_hand_pinky_joint2": (0.0, 1.919),
}

FINGER_OPEN = [0.0] * 12
FINGER_PRE_GRASP = [0.35, 0.25, 0.45, 0.05, 0.35, 0.25, 0.35, 0.25, 0.35, 0.25, 0.30, 0.20]
FINGER_GRASP = [0.95, 0.55, 1.05, 0.08, 1.15, 0.95, 1.10, 0.90, 1.05, 0.85, 0.95, 0.75]


def _position_actuators() -> str:
  lines = [
    '  <actuator>',
    '    <position name="xh_A_thumb_bend" joint="right_hand_thumb_bend_joint" kp="2.5" forcerange="-8 8"/>',
    '    <position name="xh_A_thumb_rota1" joint="right_hand_thumb_rota_joint1" kp="2.0" forcerange="-6 6"/>',
    '    <position name="xh_A_thumb_rota2" joint="right_hand_thumb_rota_joint2" kp="2.0" forcerange="-6 6"/>',
    '    <position name="xh_A_index_bend" joint="right_hand_index_bend_joint" kp="1.5" forcerange="-4 4"/>',
    '    <position name="xh_A_index_j1" joint="right_hand_index_joint1" kp="1.8" forcerange="-5 5"/>',
    '    <position name="xh_A_index_j2" joint="right_hand_index_joint2" kp="1.5" forcerange="-4 4"/>',
    '    <position name="xh_A_mid_j1" joint="right_hand_mid_joint1" kp="1.8" forcerange="-5 5"/>',
    '    <position name="xh_A_mid_j2" joint="right_hand_mid_joint2" kp="1.5" forcerange="-4 4"/>',
    '    <position name="xh_A_ring_j1" joint="right_hand_ring_joint1" kp="1.8" forcerange="-5 5"/>',
    '    <position name="xh_A_ring_j2" joint="right_hand_ring_joint2" kp="1.5" forcerange="-4 4"/>',
    '    <position name="xh_A_pinky_j1" joint="right_hand_pinky_joint1" kp="1.5" forcerange="-4 4"/>',
    '    <position name="xh_A_pinky_j2" joint="right_hand_pinky_joint2" kp="1.2" forcerange="-3 3"/>',
    '  </actuator>',
  ]
  return "\n".join(lines)


def _keyframes() -> str:
    def row(ctrl: list[float]) -> str:
        return " ".join(f"{v:.4f}" for v in ctrl)

    return f"""  <keyframe>
    <key name="open hand" qpos="0 0 0.95 1 0 0 0 {' '.join('0' for _ in range(12))}" ctrl="{row(FINGER_OPEN)}"/>
    <key name="pre grasp" qpos="0 0 0.95 1 0 0 0 {' '.join(f'{v:.4f}' for v in FINGER_PRE_GRASP)}" ctrl="{row(FINGER_PRE_GRASP)}"/>
    <key name="grasp soft" qpos="0 0 0.95 1 0 0 0 {' '.join(f'{v:.4f}' for v in FINGER_GRASP)}" ctrl="{row(FINGER_GRASP)}"/>
  </keyframe>"""


def build() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source MJCF: {SRC}. Run scripts/setup_models.sh first.")

    text = SRC.read_text()
    text = text.replace('name="floating_base"', 'name="hand_free"')
    text = text.replace('file="meshes/', 'file="')
    text = text.replace(
        '<geom material="collision_material" condim="3" contype="0" conaffinity="1" priority="1" group="1" solref="0.005 1" solimp="0.99 0.999 1e-05" friction="1 0.01 0.01" />',
        '<geom material="collision_material" condim="4" contype="1" conaffinity="1" priority="1" group="1" solref="0.02 1" solimp="0.9 0.95 0.001" friction="1.2 0.02 0.002" />',
    )
    # Decorative / backing links cause explosive mesh self-contact — disable their collision.
    text = re.sub(
        r'(<geom name="right_hand_(?:\w*back\w*|ee_link)_collision"[^>]*)(/>)',
        r'\1 contype="0" conaffinity="0"\2',
        text,
    )
    # Palm mesh pushes the bottle sideways — disable; rely on finger links + tip spheres.
    text = re.sub(
        r'(<geom name="right_hand_link_collision"[^>]*)(/>)',
        r'\1 contype="0" conaffinity="0"\2',
        text,
        count=1,
    )
    # Fingertip pads: use smooth spheres instead of mesh collision for stabler contacts.
    for finger in ("thumb_rota", "index_rota", "mid", "ring", "pinky"):
        text = re.sub(
            rf'<geom name="right_hand_{finger}_tip_collision"[^/]*/>',
            f'<geom name="right_hand_{finger}_tip_collision" type="sphere" size="0.012" '
            f'class="collision" friction="1.2 0.02 0.002"/>',
            text,
            count=1,
        )
    text = re.sub(
        r"  <actuator>.*?</actuator>",
        _position_actuators(),
        text,
        count=1,
        flags=re.S,
    )
    if "<keyframe>" not in text:
        text = text.replace("</mujoco>", f"{_keyframes()}\n</mujoco>")

    # tactile sites on fingertips
    tip_sites = """
            <site name="xh_tip_thumb" pos="0 0.05 0" size="0.004" rgba="1 0 0 0.4" group="4"/>
"""
    text = text.replace(
        '<body name="right_hand_thumb_rota_tip"',
        tip_sites + '            <body name="right_hand_thumb_rota_tip"',
        1,
    )
    for finger, offset in [
        ("index", "0 0 0.04"),
        ("mid", "0 0 0.04"),
        ("ring", "0 0 0.04"),
        ("pinky", "0 0 0.04"),
    ]:
        snippet = f"""
            <site name="xh_tip_{finger}" pos="{offset}" size="0.004" rgba="1 0 0 0.4" group="4"/>
"""
        text = text.replace(
            f'<body name="right_hand_{finger}_tip"',
            snippet + f'            <body name="right_hand_{finger}_tip"',
            1,
        )

    DST.write_text(text)
    print(f"Wrote {DST}")


if __name__ == "__main__":
    build()
