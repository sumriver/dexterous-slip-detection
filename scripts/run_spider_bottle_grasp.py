#!/usr/bin/env python3
"""Path B: CPU grasp search on SPIDER-style XHAND + horizontal bottle (no GPU, no RL)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.grasp_physics import required_support_force
from sim.spider_grasp import (
    evaluate_grasp,
    get_spider_collision_geom_ids,
    settle,
    simulate_lift,
)

SCENE = ROOT / "models/xhand_spider/bottle_scene.xml"
OUT_DIR = ROOT / "data/spider_bottle"

# Lateral pinch (build_xhand_mjcf)
FINGER_GRASP = np.array(
    [1.00, 0.40, 0.90, 0.0, 0.50, 0.35, 0.50, 0.35, 0.50, 0.35, 0.45, 0.30],
    dtype=np.float64,
)
FINGER_OPEN = np.zeros(12, dtype=np.float64)


def _make_ctrl(arm6: np.ndarray, fingers: np.ndarray) -> np.ndarray:
    return np.concatenate([arm6, fingers])


def coarse_search(model, data, hand_geoms, mg: float) -> tuple[np.ndarray, dict]:
    best_score = -1e9
    best_ctrl = None
    best_info: dict = {}
    rng = np.random.default_rng(1)

    # Grid seed around prior best + random samples
    grid_arms = []
    for tx in np.linspace(0.52, 0.58, 4):
        for ty in np.linspace(-0.09, -0.02, 5):
            for tz in np.linspace(0.03, 0.08, 4):
                for pitch in np.linspace(-1.4, -0.7, 5):
                    grid_arms.append([tx, ty, tz, 0.0, pitch, 0.0])

    for _ in range(200):
        arm = [
            rng.uniform(0.52, 0.59),
            rng.uniform(-0.13, -0.03),
            rng.uniform(0.03, 0.09),
            rng.uniform(-0.4, 0.4),
            rng.uniform(-1.6, -0.6),
            rng.uniform(-0.5, 0.5),
        ]
        grid_arms.append(arm)

    for arm in grid_arms:
        arm6 = np.array(arm, dtype=np.float64)
        for finger_scale in (0.85, 1.0, 1.1, 1.15):
            fingers = FINGER_OPEN + finger_scale * (FINGER_GRASP - FINGER_OPEN)
            ctrl = _make_ctrl(arm6, fingers)
            for i in range(model.nu):
                lo, hi = model.actuator_ctrlrange[i]
                ctrl[i] = np.clip(ctrl[i], lo, hi)
            score, info = evaluate_grasp(model, data, ctrl, hand_geoms, mg)
            if score > best_score:
                best_score = score
                best_ctrl = ctrl.copy()
                best_info = {**info, "score": score}

    if best_ctrl is None:
        raise RuntimeError("Coarse search found no stable configuration")
    return best_ctrl, best_info


def refine_search(
    model, data, hand_geoms, mg: float, init_ctrl: np.ndarray, iterations: int = 80
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(0)
    best_ctrl = init_ctrl.copy()
    best_score, best_info = evaluate_grasp(
        model, data, best_ctrl, hand_geoms, mg, lift_probe=True
    )

    for it in range(iterations):
        trial = best_ctrl.copy()
        scale = 0.004 if it < iterations // 2 else 0.002
        trial[:3] += rng.normal(0, scale, size=3)
        trial[3:6] += rng.normal(0, 0.04, size=3)
        trial[6:] += rng.normal(0, 0.03, size=12)
        for i in range(model.nu):
            lo, hi = model.actuator_ctrlrange[i]
            trial[i] = np.clip(trial[i], lo, hi)
        score, info = evaluate_grasp(
            model, data, trial, hand_geoms, mg, lift_probe=True
        )
        if score > best_score:
            best_score = score
            best_ctrl = trial
            best_info = {**info, "score": score}

    return best_ctrl, best_info


def _step_and_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    cam: mujoco.MjvCamera,
    ctrl: np.ndarray,
    substeps: int,
    frames: list[np.ndarray],
    every: int = 1,
) -> None:
    data.ctrl[:] = ctrl
    for s in range(substeps):
        mujoco.mj_step(model, data)
        if s % every == 0 or s == substeps - 1:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())


def record_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl_grasp: np.ndarray,
    path: Path,
    lift_dz: float = 0.20,
) -> None:
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = np.array([0.55, 0.0, 0.08])
    cam.distance = 0.7
    cam.azimuth = 135
    cam.elevation = -10

    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    frames: list[np.ndarray] = []

    mujoco.mj_resetData(model, data)
    ctrl = ctrl_grasp.copy()

    # Settle at grasp (matches lift test), then lift — skip approach to avoid state drift
    _step_and_frame(model, data, renderer, cam, ctrl, 2500, frames, every=100)

    z0 = float(data.xpos[bottle_id][2])
    tz0 = ctrl[2]
    for i in range(80):
        alpha = (i + 1) / 80
        lift_ctrl = ctrl.copy()
        lift_ctrl[2] = tz0 + lift_dz * alpha
        _step_and_frame(model, data, renderer, cam, lift_ctrl, 12, frames, every=2)

    # hold lifted pose
    hold_ctrl = ctrl.copy()
    hold_ctrl[2] = tz0 + lift_dz
    _step_and_frame(model, data, renderer, cam, hold_ctrl, 400, frames, every=20)

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), fps=25, codec="libx264", pixelformat="yuv420p")
    renderer.close()
    z_end = float(data.xpos[bottle_id][2])
    print(
        f"Video: {path} ({len(frames)} frames), "
        f"bottle_dz={(z_end - z0) * 100:.1f}cm z_end={z_end:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SPIDER-style CPU bottle grasp search")
    parser.add_argument("--build", action="store_true", help="Regenerate bottle_scene.xml")
    parser.add_argument("--refine", type=int, default=150)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    if args.build:
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "scripts/build_spider_bottle_scene.py")], check=True)

    if not SCENE.exists():
        print(f"Missing {SCENE}. Run with --build after setup_spider.sh", file=sys.stderr)
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    hand_geoms = get_spider_collision_geom_ids(model)
    mg = required_support_force(model)

    print(f"Scene: {SCENE.name}  nu={model.nu}  mg={mg:.3f}N  collision_geoms={len(hand_geoms)}")
    print("Coarse search...")
    ctrl0, info0 = coarse_search(model, data, hand_geoms, mg)
    print(f"  coarse: score={info0.get('score',0):.2f} contacts={info0['n_contacts']:.0f} "
          f"support={info0['support_z']:.3f}N")

    print(f"Refine ({args.refine} iter)...")
    ctrl, info = refine_search(model, data, hand_geoms, mg, ctrl0, args.refine)
    print(f"  best:   score={info.get('score',0):.2f} contacts={info['n_contacts']:.0f} "
          f"support={info['support_z']:.3f}N xy_err={info['xy_err']:.4f} "
          f"probe_dz={info.get('lift_probe_dz',0)*100:.1f}cm")
    print(f"  arm ctrl: tx={ctrl[0]:.3f} ty={ctrl[1]:.3f} tz={ctrl[2]:.3f} "
          f"rpy=({ctrl[3]:.2f},{ctrl[4]:.2f},{ctrl[5]:.2f})")

    dz, sup, nc = simulate_lift(model, data, ctrl, lift_dz=0.20)
    print(f"Lift test: bottle_dz={dz*100:.1f}cm  support_end={sup:.3f}N  contacts={nc}")
    ok = dz >= 0.05 and info["n_contacts"] >= 2 and sup >= 0.25 * mg
    print(
        f"PASS lift>=5cm + hold: {ok} "
        f"(support_grasp={info['support_z']:.3f}N support_end={sup:.3f}N mg={mg:.2f}N)"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "best_grasp_ctrl.npz", ctrl=ctrl, info=info, lift_dz=dz)
    (OUT_DIR / "result.json").write_text(
        json.dumps(
            {
                "support_z": info["support_z"],
                "mg": mg,
                "n_contacts": info["n_contacts"],
                "lift_dz_m": dz,
                "pass": bool(ok),
                "arm": ctrl[:6].tolist(),
            },
            indent=2,
        )
    )

    if args.video:
        record_video(model, data, ctrl, OUT_DIR / "spider_bottle_grasp.mp4")


if __name__ == "__main__":
    main()
