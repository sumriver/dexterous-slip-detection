#!/usr/bin/env python3
"""÷4 horizontal anti-slip experiment: slow grasp + normal/tangential grip release.

Idea (user proposal):
  1. Slow down the grasp (time-stretch the trajectory).
  2. Monitor the horizontal net normal force |F_n| and net tangential force |F_t|.
  3. When |F_n| > |F_t| (friction can't balance the inward normal push -> the
     object is being squeezed out), REDUCE grip; restore when balanced.

Compares three configs on friction ÷4:
  A. normal-speed open-loop           (current failing baseline)
  B. slow grasp, open-loop            (does slowing alone help?)
  C. slow grasp + grip-release control

Outputs a comparison table + plot to data/horizontal_grip_release/ and
docs/assets/plots/.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.spider_ketchup import DEFAULT_WORKSPACE
from sim.spider_replay import (
    SpiderTaskConfig,
    build_extend_mimic_lift_controls,
    get_object_geom_ids,
    get_spider_hand_collision_geom_ids,
    load_trajectory_arrays,
    upsample_controls,
)
from sim.spider_scene_modify import apply_object_physics
from sim.antislip_control import NormalTangentGripController
from sim.slip_horizontal import compute_hand_horizontal_frame, measure_horizontal_forces

SPIDER = ROOT / "third_party" / "spider"
OUT_DIR = ROOT / "data" / "horizontal_grip_release"
DOCS_PLOT = ROOT / "docs" / "assets" / "plots"

SIM_DT = 0.01
FRICTION = 0.25   # ÷4
SLOW_FACTOR = 2.0
HOLD_S = 1.0
EXTEND_S = 2.0
MIMIC_S = 1.0
LIFT_M = 0.10
SMOOTH_STEPS = 10  # ~100ms smoothing on Fn, Ft for the control decision


def stretch_ctrl(ctrl: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return ctrl
    T = ctrl.shape[0]
    new_T = int(round(T * factor))
    src = np.linspace(0, T - 1, new_T)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    f = (src - lo)[:, None]
    return ctrl[lo] * (1 - f) + ctrl[hi] * f


@dataclass
class RunTrace:
    name: str
    color: str
    sim_time: list[float] = field(default_factory=list)
    n_contacts: list[int] = field(default_factory=list)
    fn: list[float] = field(default_factory=list)
    ft: list[float] = field(default_factory=list)
    grip_delta: list[float] = field(default_factory=list)
    obj_x: list[float] = field(default_factory=list)
    obj_y: list[float] = field(default_factory=list)
    obj_z: list[float] = field(default_factory=list)
    z_after_traj: float = 0.0
    z_after_hold: float = 0.0
    z_end: float = 0.0


def _run(name: str, color: str, *, slow: bool, release: bool) -> RunTrace:
    cfg = SpiderTaskConfig(
        dataset_dir=SPIDER / "example_datasets",
        dataset_name="arcticv2",
        robot_type="xhand",
        embodiment_type="right",
        task="s01-ketchup_use_01",
        workspace_root=DEFAULT_WORKSPACE,
    )
    model = mujoco.MjModel.from_xml_path(str(cfg.scene_path))
    apply_object_physics(model, friction_scale=FRICTION)
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)

    qpos, qvel, ctrl = load_trajectory_arrays(cfg.trajectory_path, model, cfg.data_type)
    qpos, qvel, ctrl = upsample_controls(qpos, qvel, ctrl, SIM_DT, 0.02)
    traj_ctrl = stretch_ctrl(ctrl, SLOW_FACTOR) if slow else ctrl

    hand_geoms = get_spider_hand_collision_geom_ids(model)
    object_geoms = get_object_geom_ids(model, "right_object")
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")

    ctl = NormalTangentGripController(
        release_step=0.005, restore_step=0.004, max_release=0.05,
        trigger_ratio=1.25, min_normal=5.0,
    )
    fn_buf: list[float] = []
    ft_buf: list[float] = []

    data.qpos[:] = qpos[0]
    data.qvel[:] = qvel[0]
    data.ctrl[:] = ctrl[0]
    mujoco.mj_forward(model, data)

    trace = RunTrace(name=name, color=color)

    def measure_and_control(base_ctrl: np.ndarray) -> np.ndarray:
        frame = compute_hand_horizontal_frame(model, data)
        r = measure_horizontal_forces(model, data, hand_geoms, object_geoms, frame)
        fn_mag = float(np.hypot(r.fx_normal, r.fy_normal))
        ft_mag = float(np.hypot(r.fx_tangent, r.fy_tangent))
        fn_buf.append(fn_mag)
        ft_buf.append(ft_mag)
        if len(fn_buf) > SMOOTH_STEPS:
            fn_buf.pop(0)
            ft_buf.pop(0)
        fn_s = float(np.mean(fn_buf))
        ft_s = float(np.mean(ft_buf))
        applied = base_ctrl
        if release:
            ctl.update(fn_s, ft_s)
            applied = ctl.apply(base_ctrl, model)
        trace.sim_time.append(float(data.time))
        trace.n_contacts.append(r.n_contacts)
        trace.fn.append(fn_s)
        trace.ft.append(ft_s)
        trace.grip_delta.append(ctl.grip_delta)
        p = data.xpos[object_id]
        trace.obj_x.append(float(p[0]))
        trace.obj_y.append(float(p[1]))
        trace.obj_z.append(float(p[2]))
        return applied

    for c in traj_ctrl:
        applied = measure_and_control(c)
        data.ctrl[:] = applied
        mujoco.mj_step(model, data)
    trace.z_after_traj = float(data.xpos[object_id][2])

    hold_ctrl = build_extend_mimic_lift_controls(
        traj_ctrl, sim_dt=SIM_DT, extend_s=HOLD_S, mimic_s=MIMIC_S, lift_m=0.0
    )
    for c in hold_ctrl:
        applied = measure_and_control(c)
        data.ctrl[:] = applied
        mujoco.mj_step(model, data)
    trace.z_after_hold = float(data.xpos[object_id][2])

    lift_ctrl = build_extend_mimic_lift_controls(
        traj_ctrl, sim_dt=SIM_DT, extend_s=EXTEND_S, mimic_s=MIMIC_S, lift_m=LIFT_M
    )
    for c in lift_ctrl:
        applied = measure_and_control(c)
        data.ctrl[:] = applied
        mujoco.mj_step(model, data)
    trace.z_end = float(data.xpos[object_id][2])
    return trace


def plot(traces: list[RunTrace]) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for tr in traces:
        t = np.array(tr.sim_time)
        axes[0].plot(t, np.array(tr.obj_z) * 100, color=tr.color, lw=1.6, label=tr.name)
        axes[1].plot(t, tr.fn, color=tr.color, lw=1.2, label=f"{tr.name} |F_n|")
        axes[1].plot(t, tr.ft, color=tr.color, lw=1.0, ls="--", alpha=0.8,
                     label=f"{tr.name} |F_t|")
        axes[2].plot(t, tr.grip_delta, color=tr.color, lw=1.4, label=f"{tr.name} grip_delta")
    axes[0].set_ylabel("object z [cm]")
    axes[0].set_title("÷4 grasp: object height / horizontal Fn vs Ft / grip release")
    axes[1].set_ylabel("horizontal net force [N]")
    axes[2].set_ylabel("grip delta [rad] (<0 = release)")
    axes[2].set_xlabel("Simulation time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = OUT_DIR / "grip_release_div4.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def main() -> None:
    if not DEFAULT_WORKSPACE.joinpath("scene.xml").exists():
        print("Run: python3 scripts/build_spider_ketchup_right.py", file=sys.stderr)
        sys.exit(1)

    configs = [
        ("A_normal_openloop", "#7f8c8d", dict(slow=False, release=False)),
        ("B_slow_openloop", "#2980b9", dict(slow=True, release=False)),
        ("C_slow_release", "#e74c3c", dict(slow=True, release=True)),
    ]
    traces = []
    for name, color, kw in configs:
        print(f"Running {name} ...")
        traces.append(_run(name, color, **kw))

    plot_paths = plot(traces)
    DOCS_PLOT.mkdir(parents=True, exist_ok=True)
    for p in plot_paths:
        (DOCS_PLOT / p.name).write_bytes(p.read_bytes())

    summary = {"friction_scale": FRICTION, "slow_factor": SLOW_FACTOR, "configs": {}}
    for tr in traces:
        x0, y0, z0 = tr.obj_x[0], tr.obj_y[0], tr.obj_z[0]
        drift = float(np.hypot(tr.obj_x[-1] - x0, tr.obj_y[-1] - y0) * 100)
        min_z = float(np.min(tr.obj_z))
        contact_steps = int(sum(1 for n in tr.n_contacts if n > 0))
        summary["configs"][tr.name] = {
            "z_start_cm": z0 * 100,
            "z_after_traj_cm": tr.z_after_traj * 100,
            "z_after_hold_cm": tr.z_after_hold * 100,
            "z_end_cm": tr.z_end * 100,
            "min_z_cm": min_z * 100,
            "horiz_drift_cm": drift,
            "contact_steps": contact_steps,
            "max_release_rad": float(min(tr.grip_delta)) if tr.grip_delta else 0.0,
        }
    (OUT_DIR / "grip_release_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nPlots:")
    for p in plot_paths:
        print(f"  {p}")
    print("\nSummary (÷4):")
    for name, s in summary["configs"].items():
        print(
            f"  {name}: z_traj={s['z_after_traj_cm']:.1f} z_hold={s['z_after_hold_cm']:.1f} "
            f"z_end={s['z_end_cm']:.1f} cm  drift={s['horiz_drift_cm']:.1f}cm "
            f"contacts={s['contact_steps']} release={s['max_release_rad']:.3f}rad"
        )


if __name__ == "__main__":
    main()
