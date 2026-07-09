#!/usr/bin/env python3
"""Horizontal-plane force analysis for ketchup grasp under friction ÷2/÷4/÷8.

Defines a hand-referenced horizontal frame:
  X = 四指平展方向 (four-finger extension direction, horizontal projection)
  Y = 垂直于 X 指向右侧 (perpendicular, to the right)

For each friction case we integrate over all hand→object contacts the force
(normal + tangential) projected onto X and Y, giving net Fx(t), Fy(t), and
their time integrals (impulse). Plots + JSON summary are written to
data/horizontal_curves/ and copied to docs/assets/plots/.
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
from sim.slip_horizontal import (
    HorizontalImpulseIntegrator,
    compute_hand_horizontal_frame,
    measure_horizontal_forces,
)

SPIDER = ROOT / "third_party" / "spider"
OUT_DIR = ROOT / "data" / "horizontal_curves"
DOCS_PLOT = ROOT / "docs" / "assets" / "plots"

SIM_DT = 0.01
EXTEND_S = 2.0
MIMIC_S = 1.0
LIFT_M = 0.10

CASES = [
    ("friction_div2", 0.5, "#e74c3c"),
    ("friction_div4", 0.25, "#e67e22"),
    ("friction_div8", 0.125, "#8e44ad"),
]


@dataclass
class RunTrace:
    name: str
    friction_scale: float
    color: str
    sim_time: list[float] = field(default_factory=list)
    phase: list[str] = field(default_factory=list)
    n_contacts: list[int] = field(default_factory=list)
    fx: list[float] = field(default_factory=list)
    fy: list[float] = field(default_factory=list)
    fx_normal: list[float] = field(default_factory=list)
    fy_normal: list[float] = field(default_factory=list)
    fx_tangent: list[float] = field(default_factory=list)
    fy_tangent: list[float] = field(default_factory=list)
    f_horiz_mag: list[float] = field(default_factory=list)
    impulse_x: list[float] = field(default_factory=list)
    impulse_y: list[float] = field(default_factory=list)
    impulse_mag: list[float] = field(default_factory=list)
    obj_x: list[float] = field(default_factory=list)
    obj_y: list[float] = field(default_factory=list)
    obj_z: list[float] = field(default_factory=list)
    frame_x: list[float] = field(default_factory=list)
    frame_y: list[float] = field(default_factory=list)


def _run_trace(name: str, friction_scale: float, color: str) -> RunTrace:
    cfg = SpiderTaskConfig(
        dataset_dir=SPIDER / "example_datasets",
        dataset_name="arcticv2",
        robot_type="xhand",
        embodiment_type="right",
        task="s01-ketchup_use_01",
        workspace_root=DEFAULT_WORKSPACE,
    )
    model = mujoco.MjModel.from_xml_path(str(cfg.scene_path))
    apply_object_physics(model, friction_scale=friction_scale)
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)

    qpos, qvel, ctrl = load_trajectory_arrays(cfg.trajectory_path, model, cfg.data_type)
    qpos, qvel, ctrl = upsample_controls(qpos, qvel, ctrl, SIM_DT, 0.02)

    hand_geoms = get_spider_hand_collision_geom_ids(model)
    object_geoms = get_object_geom_ids(model, "right_object")
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")

    data.qpos[:] = qpos[0]
    data.qvel[:] = qvel[0]
    data.ctrl[:] = ctrl[0]
    mujoco.mj_forward(model, data)

    trace = RunTrace(name=name, friction_scale=friction_scale, color=color)
    integ = HorizontalImpulseIntegrator(SIM_DT)

    def record(phase_name: str) -> None:
        # Dynamic hand frame each step: X=four-finger extension, Y=perpendicular right.
        frame = compute_hand_horizontal_frame(model, data)
        r = measure_horizontal_forces(model, data, hand_geoms, object_geoms, frame)
        ix, iy, im = integ.update(r)
        trace.sim_time.append(float(data.time))
        trace.phase.append(phase_name)
        trace.n_contacts.append(r.n_contacts)
        trace.fx.append(r.fx)
        trace.fy.append(r.fy)
        trace.fx_normal.append(r.fx_normal)
        trace.fy_normal.append(r.fy_normal)
        trace.fx_tangent.append(r.fx_tangent)
        trace.fy_tangent.append(r.fy_tangent)
        trace.f_horiz_mag.append(r.f_horiz_mag)
        trace.impulse_x.append(ix)
        trace.impulse_y.append(iy)
        trace.impulse_mag.append(im)
        p = data.xpos[object_id]
        trace.obj_x.append(float(p[0]))
        trace.obj_y.append(float(p[1]))
        trace.obj_z.append(float(p[2]))
        return frame

    # Grasp trajectory (contacts form here; at very low friction the object
    # may slip out during this phase already).
    last_frame = None
    for c in ctrl:
        data.ctrl[:] = c
        mujoco.mj_step(model, data)
        last_frame = record("trajectory")

    if last_frame is not None:
        trace.frame_x = [float(v) for v in last_frame.x_hat]
        trace.frame_y = [float(v) for v in last_frame.y_hat]

    # Extend phase: mimic last 1s of grip + wrist lift 10cm.
    extend_ctrl = build_extend_mimic_lift_controls(
        ctrl, sim_dt=SIM_DT, extend_s=EXTEND_S, mimic_s=MIMIC_S, lift_m=LIFT_M
    )
    for c in extend_ctrl:
        data.ctrl[:] = c
        mujoco.mj_step(model, data)
        record("extend")

    return trace


def _extend_start_time(tr: RunTrace) -> float:
    for t, ph in zip(tr.sim_time, tr.phase):
        if ph == "extend":
            return t
    return tr.sim_time[-1] if tr.sim_time else 0.0


def _mark_extend(ax, traces: list[RunTrace]) -> None:
    if not traces:
        return
    t_ext = _extend_start_time(traces[0])
    ax.axvline(t_ext, color="black", ls=":", lw=1, alpha=0.6)


def plot_traces(traces: list[RunTrace], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # --- Figure 1: Fx(t) and Fy(t) (total force) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for tr in traces:
        t = np.array(tr.sim_time)
        axes[0].plot(t, tr.fx, label=f"{tr.name}  Fx", color=tr.color, lw=1.5)
        axes[1].plot(t, tr.fy, label=f"{tr.name}  Fy", color=tr.color, lw=1.5)
    for ax in axes:
        ax.axhline(0.0, color="gray", ls="--", lw=1, alpha=0.6)
        _mark_extend(ax, traces)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Fx = sum f.Xhat  [N]  (along fingers)")
    axes[1].set_ylabel("Fy = sum f.Yhat  [N]  (perp. right)")
    axes[1].set_xlabel("Simulation time [s]  (dotted line = extend/lift start)")
    axes[0].set_title("Horizontal net contact force on object  (X=finger-extension, Y=perp-right)")
    fig.tight_layout()
    p = out_dir / "horizontal_fx_fy.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- Figure 2: normal vs tangential contributions to Fx, Fy ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for tr in traces:
        t = np.array(tr.sim_time)
        axes[0].plot(t, tr.fx_normal, color=tr.color, lw=1.5, label=f"{tr.name}  Fx_normal")
        axes[0].plot(t, tr.fx_tangent, color=tr.color, lw=1.0, ls="--", alpha=0.8,
                     label=f"{tr.name}  Fx_tangent")
        axes[1].plot(t, tr.fy_normal, color=tr.color, lw=1.5, label=f"{tr.name}  Fy_normal")
        axes[1].plot(t, tr.fy_tangent, color=tr.color, lw=1.0, ls="--", alpha=0.8,
                     label=f"{tr.name}  Fy_tangent")
    for ax in axes:
        ax.axhline(0.0, color="gray", ls="--", lw=1, alpha=0.6)
        _mark_extend(ax, traces)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("X-axis force [N]")
    axes[1].set_ylabel("Y-axis force [N]")
    axes[1].set_xlabel("Simulation time [s]")
    axes[0].set_title("Normal vs tangential contribution to horizontal force")
    fig.tight_layout()
    p = out_dir / "horizontal_normal_tangent.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- Figure 3: impulse (time integral of force) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for tr in traces:
        t = np.array(tr.sim_time)
        axes[0].plot(t, tr.impulse_x, color=tr.color, lw=2, label=f"{tr.name}  int Fx dt")
        axes[0].plot(t, tr.impulse_y, color=tr.color, lw=1.2, ls="--",
                     label=f"{tr.name}  int Fy dt")
        axes[1].plot(t, tr.impulse_mag, color=tr.color, lw=2,
                     label=f"{tr.name}  int |F_h| dt")
    for ax in axes:
        _mark_extend(ax, traces)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].axhline(0.0, color="gray", ls="--", lw=1, alpha=0.6)
    axes[0].set_ylabel("Signed impulse [N.s]")
    axes[1].set_ylabel("Magnitude impulse [N.s]")
    axes[1].set_xlabel("Simulation time [s]")
    axes[0].set_title("Horizontal force impulse (time integral) over whole run")
    fig.tight_layout()
    p = out_dir / "horizontal_impulse.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- Figure 4: horizontal object displacement + force magnitude ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for tr in traces:
        t = np.array(tr.sim_time)
        x0, y0 = tr.obj_x[0], tr.obj_y[0]
        dxy = np.hypot(np.array(tr.obj_x) - x0, np.array(tr.obj_y) - y0) * 100
        axes[0].plot(t, dxy, color=tr.color, lw=2, label=f"{tr.name}  |d_horiz|")
        axes[1].plot(t, tr.f_horiz_mag, color=tr.color, lw=1.5,
                     label=f"{tr.name}  |F_horiz|")
    for ax in axes:
        _mark_extend(ax, traces)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Object horizontal drift [cm]")
    axes[1].set_ylabel("|F_horiz| [N]")
    axes[1].set_xlabel("Simulation time [s]")
    axes[0].set_title("Horizontal object drift vs net horizontal force")
    fig.tight_layout()
    p = out_dir / "horizontal_drift_force.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    return paths


def _phase_mean(vals: list[float], phase: list[str], want: str) -> float:
    sel = [v for v, ph in zip(vals, phase) if ph == want]
    return float(np.mean(sel)) if sel else float("nan")


def _contact_steps(tr: RunTrace, want: str) -> int:
    return int(sum(1 for n, ph in zip(tr.n_contacts, tr.phase) if ph == want and n > 0))


def main() -> None:
    if not DEFAULT_WORKSPACE.joinpath("scene.xml").exists():
        print("Run: python3 scripts/build_spider_ketchup_right.py", file=sys.stderr)
        sys.exit(1)

    traces: list[RunTrace] = []
    for name, fscale, color in CASES:
        print(f"Running {name} (friction×{fscale})...")
        traces.append(_run_trace(name, fscale, color))

    plot_paths = plot_traces(traces, OUT_DIR)

    DOCS_PLOT.mkdir(parents=True, exist_ok=True)
    for p in plot_paths:
        (DOCS_PLOT / p.name).write_bytes(p.read_bytes())

    summary = {
        "frame": "X=four-finger extension (horizontal), Y=X x up (perpendicular right)",
        "sim_dt": SIM_DT,
        "extend_s": EXTEND_S,
        "lift_m": LIFT_M,
        "traces": {
            tr.name: {
                "friction_scale": tr.friction_scale,
                "x_hat": tr.frame_x,
                "y_hat": tr.frame_y,
                "traj_contact_steps": _contact_steps(tr, "trajectory"),
                "extend_contact_steps": _contact_steps(tr, "extend"),
                "traj_fx_mean": _phase_mean(tr.fx, tr.phase, "trajectory"),
                "traj_fy_mean": _phase_mean(tr.fy, tr.phase, "trajectory"),
                "extend_fx_mean": _phase_mean(tr.fx, tr.phase, "extend"),
                "extend_fy_mean": _phase_mean(tr.fy, tr.phase, "extend"),
                "extend_f_horiz_mean": _phase_mean(tr.f_horiz_mag, tr.phase, "extend"),
                "final_impulse_x": tr.impulse_x[-1] if tr.impulse_x else None,
                "final_impulse_y": tr.impulse_y[-1] if tr.impulse_y else None,
                "final_impulse_mag": tr.impulse_mag[-1] if tr.impulse_mag else None,
                "object_horiz_drift_cm": (
                    float(
                        np.hypot(tr.obj_x[-1] - tr.obj_x[0], tr.obj_y[-1] - tr.obj_y[0]) * 100
                    )
                    if tr.obj_x
                    else None
                ),
                "object_dz_cm": (
                    float((tr.obj_z[-1] - tr.obj_z[0]) * 100) if tr.obj_z else None
                ),
            }
            for tr in traces
        },
    }
    (OUT_DIR / "horizontal_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nPlots:")
    for p in plot_paths:
        print(f"  {p}")
    print(f"  docs copy: {DOCS_PLOT}")
    print("\nSummary (whole run):")
    for name, stats in summary["traces"].items():
        print(
            f"  {name}: traj_contacts={stats['traj_contact_steps']} "
            f"extend_contacts={stats['extend_contact_steps']} "
            f"| int Fx={stats['final_impulse_x']:.3f} int Fy={stats['final_impulse_y']:.3f} "
            f"int|F|={stats['final_impulse_mag']:.3f} N.s "
            f"| drift={stats['object_horiz_drift_cm']:.2f}cm dz={stats['object_dz_cm']:.2f}cm"
        )


if __name__ == "__main__":
    main()
