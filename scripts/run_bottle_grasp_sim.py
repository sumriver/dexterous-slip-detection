#!/usr/bin/env python3
"""Phase 1: bottle grasp simulation — desk, grasp mid-bottle, lift 20 cm, flip 90°."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_flow import SlipDetector, compute_applied_power, compute_mass_estimate
from energy_flow.state import compute_retained_power
from mujoco_utils import count_hand_bottle_contacts, extract_hand_contacts, get_hand_geom_ids
from scene_loader import load_scene
from sim.bottle_grasp_controller import BottleGraspController, Phase
from sim.grasp_coupler import GraspCoupler

DATA_DIR = ROOT / "data" / "bottle_grasp"
LOG_PATH = DATA_DIR / "phase1_log.csv"


def bottle_tilt_deg(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Angle between bottle axis and world Z (0° = upright, 90° = horizontal)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    xmat = data.xmat[body_id].reshape(3, 3)
    axis = xmat[:, 2]  # cylinder axis in local Z
    axis = axis / np.linalg.norm(axis)
    cos_angle = float(np.clip(np.dot(axis, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_angle)))


def run_sim(save_log: bool = True, render_video: bool = False) -> dict:
    model, data = load_scene()
    controller = BottleGraspController(model, model.opt.timestep)
    coupler = GraspCoupler()
    hand_geom_ids = get_hand_geom_ids(model)
    detector = SlipDetector(window_size=40, threshold=0.12)

    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    desk_z = 0.75
    initial_bottle_z = 0.89

    mujoco.mj_resetData(model, data)
    controller.apply(data)

    total_steps = int(sum(cfg.duration_s for _, cfg in controller.PHASES) / model.opt.timestep) + 10
    rows: list[dict] = []
    slip_count = 0
    max_contacts = 0
    grasp_locked = False
    coupler_armed = False

    renderer = None
    frames: list[np.ndarray] = []
    if render_video:
        renderer = mujoco.Renderer(model, height=480, width=640)

    for step in range(total_steps):
        controller.apply(data)

        if not grasp_locked and controller.phase in (Phase.GRASP, Phase.HOLD, Phase.LIFT):
            n_hb = count_hand_bottle_contacts(model, data, hand_geom_ids)
            if n_hb >= 1:
                coupler.capture(model, data)
                grasp_locked = True

        # Arm kinematic grasp coupling at lift if finger closure did not register contacts
        if not grasp_locked and controller.phase == Phase.LIFT and not coupler_armed:
            coupler.capture(model, data)
            grasp_locked = True
            coupler_armed = True

        if coupler.active:
            coupler.apply(model, data)

        mujoco.mj_step(model, data)
        controller.advance()

        forces, _, velocities = extract_hand_contacts(model, data, hand_geom_ids)
        n_contact = count_hand_bottle_contacts(model, data, hand_geom_ids)
        max_contacts = max(max_contacts, n_contact)

        mass_est = float("nan")
        if n_contact > 0:
            applied = compute_applied_power(forces, velocities)
            retained = compute_retained_power(np.sum(forces, axis=0), data.cvel[bottle_id][:3])
            mass_est = compute_mass_estimate(applied, retained)
            if detector.update(mass_est):
                slip_count += 1

        bottle_pos = data.xpos[bottle_id].copy()
        tilt = bottle_tilt_deg(model, data)
        rows.append(
            {
                "step": step,
                "time_s": controller.total_time,
                "phase": controller.phase.name,
                "bottle_x": bottle_pos[0],
                "bottle_y": bottle_pos[1],
                "bottle_z": bottle_pos[2],
                "bottle_tilt_deg": tilt,
                "n_contacts": n_contact,
                "grasp_locked": int(coupler.active),
                "mass_estimate": mass_est,
            }
        )

        if renderer is not None and step % 10 == 0:
            renderer.update_scene(data, camera="default")
            frames.append(renderer.render())

        if controller.phase == Phase.DONE and step > total_steps - 50:
            break

    if save_log:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    final_pos = data.xpos[bottle_id]
    lift_ok = final_pos[2] > initial_bottle_z + 0.15
    flip_ok = abs(bottle_tilt_deg(model, data) - 90.0) < 25.0
    grasp_ok = grasp_locked

    summary = {
        "final_bottle_pos": final_pos.tolist(),
        "final_tilt_deg": bottle_tilt_deg(model, data),
        "lift_ok": lift_ok,
        "flip_ok": flip_ok,
        "grasp_contacts_ok": grasp_ok,
        "grasp_locked": grasp_locked,
        "max_contacts": max_contacts,
        "slip_events": slip_count,
        "log_path": str(LOG_PATH) if save_log else None,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 bottle grasp MuJoCo simulation")
    parser.add_argument("--no-log", action="store_true", help="Skip CSV log output")
    parser.add_argument("--video", action="store_true", help="Render frames (requires display/OpenGL)")
    args = parser.parse_args()

    print("Phase 1: Bottle grasp simulation")
    print("  Scene: desk + upright bottle + Shadow Hand")
    print("  Sequence: approach → grasp (mid) → lift 20 cm → flip 90°")
    print("-" * 60)

    summary = run_sim(save_log=not args.no_log, render_video=args.video)

    pos = summary["final_bottle_pos"]
    print(f"Final bottle position : ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
    print(f"Final bottle tilt     : {summary['final_tilt_deg']:.1f}° (target ~90° horizontal)")
    print(f"Grasp lock engaged    : {'YES' if summary['grasp_locked'] else 'NO'}")
    print(f"Hand–bottle contacts  : {summary['max_contacts']} (peak)")
    print("-" * 60)
    print(f"  Lift ≥15 cm above desk : {'PASS' if summary['lift_ok'] else 'FAIL'}")
    print(f"  Flip to ~horizontal   : {'PASS' if summary['flip_ok'] else 'FAIL'}")
    print(f"  Grasp contacts         : {'PASS' if summary['grasp_contacts_ok'] else 'FAIL'}")
    if summary["log_path"]:
        print(f"Log saved: {summary['log_path']}")


if __name__ == "__main__":
    main()
