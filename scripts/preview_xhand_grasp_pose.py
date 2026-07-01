#!/usr/bin/env python3
"""Render and validate XHAND lateral grasp pose (no penetration before close)."""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mujoco_utils import get_hand_geom_ids
from scene_loader import load_xhand_scene
from sim.xhand_grasp_controller import XHandGraspController

OUT = ROOT / "data" / "xhand_grasp" / "pose_preview.png"


def min_bottle_gap(model, data) -> float:
    bottle_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_geom")
    return float(
        min(
            mujoco.mj_geomDistance(model, data, gi, bottle_geom, 10.0, np.zeros(6))
            for gi in get_hand_geom_ids(model)
        )
    )


def main() -> None:
    model, data = load_xhand_scene()
    ctrl = XHandGraspController(model, model.opt.timestep)
    ctrl.reset_hand_pose(data)
    mujoco.mj_forward(model, data)

    # Pre-grasp beside bottle
    adr = ctrl.hand_free_adr
    data.qpos[adr : adr + 3] = ctrl.GRASP_POS
    data.qpos[adr + 3 : adr + 7] = ctrl.BASE_QUAT / np.linalg.norm(ctrl.BASE_QUAT)
    open_ctrl = ctrl._finger_ctrl_from_key("open hand")
    data.ctrl[:] = open_ctrl
    mujoco.mj_forward(model, data)
    gap_open = min_bottle_gap(model, data)

    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = np.array([0.55, 0.0, 0.05])
    cam.distance = 0.55
    cam.azimuth = 135
    cam.elevation = -5
    renderer.update_scene(data, camera=cam)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUT, renderer.render())
    renderer.close()

    thumb = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_rota_tip")]
    fingers = np.mean(
        [
            data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)][1]
            for n in (
                "right_hand_index_rota_tip",
                "right_hand_mid_tip",
                "right_hand_ring_tip",
                "right_hand_pinky_tip",
            )
        ]
    )
    print(f"Preview: {OUT}")
    print(f"Open-hand clearance to bottle: {gap_open*1000:.1f} mm")
    print(f"Thumb Y={thumb[1]:.3f}, fingers mean Y={fingers:.3f} (bottle center Y=0)")
    if gap_open < 0.003:
        print("WARNING: hand too close / penetrating before grasp close")
        sys.exit(1)
    if thumb[1] * fingers > 0:
        print("WARNING: thumb and fingers on same side of bottle")
        sys.exit(1)
    print("Pose OK: lateral tripod geometry")


if __name__ == "__main__":
    main()
