#!/usr/bin/env python3
"""XHAND1: horizontal bottle → mid grasp → lift 20 cm → stand upright → hold."""

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
from mujoco_utils import extract_hand_contacts, get_hand_geom_ids
from scene_loader import load_xhand_scene
from sim.bottle_grasp_controller import Phase
from sim.grasp_physics import measure_bottle_grasp, required_support_force
from sim.video_recorder import VideoRecorder
from sim.xhand_grasp_controller import XHandGraspController
from sim.xhand_tactile_sim import XHandTactileSimulator

DATA_DIR = ROOT / "data" / "xhand_grasp"
LOG_PATH = DATA_DIR / "xhand_phase1_log.csv"
VIDEO_PATH = DATA_DIR / "xhand_bottle_grasp.mp4"
KEYFRAME_DIR = DATA_DIR / "keyframes"


def bottle_tilt_deg(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Angle between bottle long axis and world +Z (0° = upright)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    xmat = data.xmat[body_id].reshape(3, 3)
    axis = xmat[:, 2] / np.linalg.norm(xmat[:, 2])
    cos_angle = float(np.clip(np.dot(axis, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_angle)))


def save_keyframe(renderer: mujoco.Renderer, data: mujoco.MjData, camera, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer.update_scene(data, camera=camera)
    import imageio.v3 as iio

    iio.imwrite(path, renderer.render())


def run_sim(
    save_log: bool = True,
    render_video: bool = False,
    save_keyframes: bool = True,
    video_path: Path | None = None,
    video_fps: int = 30,
) -> dict:
    model, data = load_xhand_scene()
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    hand_geom_ids = get_hand_geom_ids(model)
    tactile = XHandTactileSimulator(model, hand_geom_ids)
    detector = SlipDetector(window_size=40, threshold=0.12)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    initial_bottle_z = float(data.xpos[bottle_id][2])
    initial_tilt = bottle_tilt_deg(model, data)

    controller = XHandGraspController(model, model.opt.timestep)
    controller.reset_hand_pose(data)
    mujoco.mj_forward(model, data)

    mg = required_support_force(model)
    total_steps = int(sum(cfg.duration_s for _, cfg in controller.PHASES) / model.opt.timestep) + 10
    rows: list[dict] = []
    slip_count = 0

    lift_metrics: list[float] = []
    stand_metrics: list[float] = []
    lift_contact_steps = 0
    stand_contact_steps = 0
    hold_contact_steps = 0
    upright_hold_steps = 0
    max_contacts = 0
    max_tactile_fn = 0.0

    recorder: VideoRecorder | None = None
    if render_video:
        recorder = VideoRecorder(model, video_path or VIDEO_PATH, fps=video_fps, timestep=model.opt.timestep)

    keyframe_renderer = None
    keyframe_camera = None
    saved_kf: dict[str, str] = {}
    if save_keyframes:
        keyframe_renderer = mujoco.Renderer(model, height=720, width=1280)
        from sim.video_recorder import make_scene_camera

        keyframe_camera = make_scene_camera(model)

    last_phase = None

    for step in range(total_steps):
        controller.apply(data)
        mujoco.mj_step(model, data)
        controller.advance()

        metrics = measure_bottle_grasp(model, data, hand_geom_ids)
        tactile_frame = tactile.sample(data)
        max_contacts = max(max_contacts, metrics.n_contacts)
        max_tactile_fn = max(max_tactile_fn, tactile_frame.total_normal_force)

        phase = controller.phase
        tilt = bottle_tilt_deg(model, data)

        if phase == Phase.HOLD and metrics.n_contacts >= 2:
            hold_contact_steps += 1
        if phase == Phase.LIFT:
            lift_metrics.append(metrics.support_force_z)
            if metrics.n_contacts >= 2:
                lift_contact_steps += 1
        elif phase == Phase.FLIP:
            stand_metrics.append(metrics.support_force_z)
            if metrics.n_contacts >= 2:
                stand_contact_steps += 1
        elif phase == Phase.DONE:
            if metrics.n_contacts >= 2 and tilt < 25.0:
                upright_hold_steps += 1

        forces, _, velocities = extract_hand_contacts(model, data, hand_geom_ids)
        mass_est = float("nan")
        if metrics.n_contacts > 0:
            applied = compute_applied_power(forces, velocities)
            retained = compute_retained_power(metrics.force_on_bottle, data.cvel[bottle_id][:3])
            mass_est = compute_mass_estimate(applied, retained)
            if detector.update(mass_est):
                slip_count += 1

        bottle_pos = data.xpos[bottle_id].copy()
        rows.append(
            {
                "step": step,
                "time_s": controller.total_time,
                "phase": phase.name,
                "bottle_x": bottle_pos[0],
                "bottle_y": bottle_pos[1],
                "bottle_z": bottle_pos[2],
                "bottle_tilt_deg": tilt,
                "n_contacts": metrics.n_contacts,
                "support_force_z": metrics.support_force_z,
                "tactile_fn": tactile_frame.total_normal_force,
                "tactile_taxels": tactile_frame.contact_count,
                "mass_estimate": mass_est,
            }
        )

        if recorder is not None:
            recorder.maybe_capture(data, step)

        if phase == Phase.HOLD and controller._phase_time > 2.0:
            controller.try_enable_lift(metrics.n_contacts, data.xpos[bottle_id][:2])

        if phase == Phase.LIFT and controller.lift_enabled and controller._phase_time > 5.0:
            controller.try_enable_stand(metrics.n_contacts, data.xpos[bottle_id][2], initial_bottle_z)

        if keyframe_renderer is not None and phase != last_phase:
            kf_path = KEYFRAME_DIR / f"{phase.name.lower()}.png"
            save_keyframe(keyframe_renderer, data, keyframe_camera, kf_path)
            saved_kf[phase.name] = str(kf_path)
            last_phase = phase

        if phase == Phase.DONE and step > total_steps - 50:
            break

    if keyframe_renderer is not None:
        keyframe_renderer.close()

    video_out: Path | None = None
    if recorder is not None:
        video_out = recorder.save()
        recorder.close()

    if save_log and rows:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    final_pos = data.xpos[bottle_id]
    final_tilt = bottle_tilt_deg(model, data)
    final_metrics = measure_bottle_grasp(model, data, hand_geom_ids)

    anchor = controller.BOTTLE_ANCHOR_XY
    final_xy_dist = float(np.linalg.norm(final_pos[:2] - anchor))
    simulation_stable = final_xy_dist < 0.20 and final_pos[2] < 2.5

    max_bottle_z = max(float(r["bottle_z"]) for r in rows) if rows else initial_bottle_z
    done_steps = sum(1 for r in rows if r["phase"] == "DONE")

    grasp_ok = hold_contact_steps > 400 and simulation_stable and max_contacts >= 2
    lift_ok = (
        simulation_stable
        and max_bottle_z > initial_bottle_z + 0.15
        and lift_contact_steps > max(1, len(lift_metrics) * 0.2)
    )
    stand_ok = simulation_stable and lift_ok and final_tilt < 20.0 and stand_contact_steps > max(1, len(stand_metrics) * 0.15)
    upright_hold_ok = (
        simulation_stable
        and stand_ok
        and upright_hold_steps > max(50, done_steps * 0.4)
        and final_metrics.n_contacts >= 2
        and final_tilt < 20.0
    )

    return {
        "scenario": "horizontal → lift 20cm → stand upright → hold",
        "hand_model": "XHAND1 (worldstring URDF → MJCF)",
        "final_bottle_pos": final_pos.tolist(),
        "final_tilt_deg": final_tilt,
        "initial_bottle_z": initial_bottle_z,
        "initial_tilt_deg": initial_tilt,
        "required_mg": mg,
        "simulation_stable": simulation_stable,
        "max_bottle_z": max_bottle_z,
        "lift_ok": lift_ok,
        "stand_ok": stand_ok,
        "upright_hold_ok": upright_hold_ok,
        "grasp_ok": grasp_ok,
        "max_contacts": max_contacts,
        "max_tactile_fn": max_tactile_fn,
        "hold_contact_steps": hold_contact_steps,
        "upright_hold_steps": upright_hold_steps,
        "final_contacts": final_metrics.n_contacts,
        "slip_events": slip_count,
        "log_path": str(LOG_PATH) if save_log else None,
        "video_path": str(video_out) if video_out else None,
        "video_frames": recorder.frame_count if recorder else 0,
        "keyframes": saved_kf,
        "physics_mode": "contact_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="XHAND horizontal bottle stand-up simulation")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--no-keyframes", action="store_true")
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args()

    print("XHAND: horizontal bottle → mid grasp → lift 20cm → stand upright → hold")
    print("-" * 60)

    summary = run_sim(
        save_log=not args.no_log,
        render_video=args.video,
        save_keyframes=not args.no_keyframes,
        video_path=args.video_path,
        video_fps=args.video_fps,
    )

    pos = summary["final_bottle_pos"]
    print(f"Scenario              : {summary['scenario']}")
    print(f"Final bottle position : ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
    print(f"Initial tilt / final  : {summary['initial_tilt_deg']:.1f}° → {summary['final_tilt_deg']:.1f}°")
    print(f"Max bottle z          : {summary['max_bottle_z']:.3f} m (lift target +0.20 m)")
    print(f"Peak contacts         : {summary['max_contacts']}")
    print(f"Simulation stable     : {'YES' if summary['simulation_stable'] else 'NO'}")
    print("-" * 60)
    print(f"  1. Mid grasp         : {'PASS' if summary['grasp_ok'] else 'FAIL'}")
    print(f"  2. Lift 20 cm        : {'PASS' if summary['lift_ok'] else 'FAIL'}")
    print(f"  3. Stand upright     : {'PASS' if summary['stand_ok'] else 'FAIL'}")
    print(f"  4. Hold without slip : {'PASS' if summary['upright_hold_ok'] else 'FAIL'}")
    if summary["log_path"]:
        print(f"Log: {summary['log_path']}")
    if summary["video_path"]:
        print(f"Video: {summary['video_path']} ({summary['video_frames']} frames)")
    if summary["keyframes"]:
        print("Keyframes:")
        for phase_name, path in summary["keyframes"].items():
            print(f"  {phase_name}: {path}")


if __name__ == "__main__":
    main()
