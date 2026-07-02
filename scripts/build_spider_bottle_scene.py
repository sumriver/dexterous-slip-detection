#!/usr/bin/env python3
"""Build SPIDER-style XHAND + horizontal bottle scene (explicit contact pairs)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAND_XML = ROOT / "models/xhand_spider/right_hand.xml"
OUT = ROOT / "models/xhand_spider/bottle_scene.xml"

# Lateral pinch finger targets (from build_xhand_mjcf FINGER_GRASP_LATERAL)
FINGER_GRASP_LATERAL = [
    1.00, 0.40, 0.90, 0.0, 0.50, 0.35, 0.50, 0.35, 0.50, 0.35, 0.45, 0.30,
]

HAND_COLLISION_GEOMS = [
    "collision_hand_right_palm_0",
    "collision_hand_right_thumb_2",
    "collision_hand_right_thumb_1",
    "collision_hand_right_thumb_0",
    "collision_hand_right_index_1",
    "collision_hand_right_index_0",
    "collision_hand_right_middle_1",
    "collision_hand_right_middle_0",
    "collision_hand_right_ring_1",
    "collision_hand_right_ring_0",
    "collision_hand_right_pinky_1",
    "collision_hand_right_pinky_0",
]


def _self_pairs() -> str:
    """Hand-internal pairs (from SPIDER scene pattern)."""
    lines = []
    palm = "collision_hand_right_palm_0"
    fingers = [g for g in HAND_COLLISION_GEOMS if g != palm]
    for g in fingers:
        lines.append(
            f'    <pair geom1="{palm}" geom2="{g}" '
            f'friction="1 1 0.1 0 0" condim="4"/>'
        )
    thumb = [g for g in HAND_COLLISION_GEOMS if "thumb" in g]
    others = [g for g in fingers if "thumb" not in g]
    for t in thumb:
        for o in others:
            lines.append(
                f'    <pair geom1="{t}" geom2="{o}" '
                f'friction="1 1 0.1 0 0" condim="4"/>'
            )
    return "\n".join(lines)


def _bottle_pairs() -> str:
    lines = [
        '    <pair geom1="floor" geom2="bottle_geom" friction="1 1 0.1 0 0" condim="4"/>',
    ]
    for g in HAND_COLLISION_GEOMS:
        lines.append(
            f'    <pair geom1="{g}" geom2="bottle_geom" '
            f'friction="1.2 1.2 0.1 0 0" condim="4"/>'
        )
    return "\n".join(lines)


def build() -> None:
    if not HAND_XML.exists():
        raise FileNotFoundError(f"Missing {HAND_XML}. Run: bash scripts/setup_spider.sh")
    mesh_dir = ROOT / "models/xhand_spider/meshes"
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Missing meshes at {mesh_dir}. Run: bash scripts/setup_spider.sh")

    key_ctrl = " ".join(f"{v:.4f}" for v in FINGER_GRASP_LATERAL)
    # Arm at grid-search seed: tx ty tz roll pitch yaw + fingers open-ish
    key_qpos_arm = "0.55 -0.08 0.06 0.0 -1.2 0.0"
    key_finger_zeros = " ".join("0" for _ in range(12))

    xml = f"""<mujoco model="xhand_spider_bottle">
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <default>
    <geom contype="0" conaffinity="0" condim="4"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.1"/>
    <material name="bottle_mat" rgba="0.15 0.55 0.85 0.95" specular="0.4" shininess="0.3"/>
  </asset>

  <include file="right_hand.xml"/>

  <worldbody>
    <light pos="0.55 0 2.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"
          contype="1" conaffinity="1" friction="1.0 0.02 0.002"/>
    <!-- Horizontal bottle: cylinder axis along +X -->
    <body name="bottle" pos="0.55 0 0.022" quat="0.707107 0 0.707107 0">
      <freejoint name="bottle_free"/>
      <inertial pos="0 0 0" mass="0.15" diaginertia="0.00015 0.00015 0.00002"/>
      <geom name="bottle_geom" type="cylinder" size="0.022 0.14" material="bottle_mat"
            contype="1" conaffinity="1" friction="1.2 0.03 0.003"
            solimp="0.9 0.95 0.001" solref="0.02 1"/>
    </body>
    <camera name="grasp_cam" pos="0.25 -0.65 0.35" xyaxes="0.85 0.5 0 -0.15 0.3 0.94"/>
  </worldbody>

  <contact>
{_self_pairs()}
{_bottle_pairs()}
  </contact>

  <keyframe>
    <key name="home" qpos="{key_qpos_arm} {key_finger_zeros} 0.55 0 0.022 0.707107 0 0.707107 0"
         ctrl="{key_qpos_arm} {key_finger_zeros}"/>
    <key name="grasp_lateral" qpos="{key_qpos_arm} {key_ctrl} 0.55 0 0.022 0.707107 0 0.707107 0"
         ctrl="{key_qpos_arm} {key_ctrl}"/>
  </keyframe>
</mujoco>
"""
    OUT.write_text(xml)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
