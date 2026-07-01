#!/usr/bin/env python3
"""Minimal MuJoCo simulation: load scene, step physics, log contact energy."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_flow import SlipDetector, compute_applied_power, compute_mass_estimate
from energy_flow.state import compute_retained_power
from mujoco_utils import extract_hand_contacts, get_hand_geom_ids

SCENE = ROOT / "models" / "scenes" / "block_grasp_scene.xml"
STEPS = 500


def main() -> None:
    if not SCENE.exists():
        print(f"Scene not found: {SCENE}")
        print("Run: bash scripts/setup_models.sh")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    hand_geom_ids = get_hand_geom_ids(model)

    detector = SlipDetector(window_size=30, threshold=0.15)
    slip_events = 0

    print(f"Loaded: {SCENE.name}  (nbody={model.nbody}, ngeom={model.ngeom})")
    print(f"Hand geoms detected: {len(hand_geom_ids)}")
    print("-" * 60)

    for step in range(STEPS):
        mujoco.mj_step(model, data)

        forces, positions, velocities = extract_hand_contacts(model, data, hand_geom_ids)
        if len(forces) == 0:
            continue

        applied = compute_applied_power(forces, velocities)
        total_force = np.sum(forces, axis=0)
        # Use block center velocity as grasp center proxy
        block_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grasp_block")
        block_vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, block_body_id, block_vel, 0)
        retained = compute_retained_power(total_force, block_vel[:3])
        mass_est = compute_mass_estimate(applied, retained)

        if detector.update(mass_est):
            slip_events += 1
            if slip_events <= 5:
                print(f"  step {step:4d}  SLIP  m̃={mass_est:.4f}  median={detector.median:.4f}  contacts={len(forces)}")

    print("-" * 60)
    print(f"Simulation complete: {STEPS} steps, {slip_events} slip events detected")
    print(f"Block height: {data.xpos[block_body_id][2]:.4f} m")


if __name__ == "__main__":
    main()
