#!/usr/bin/env python3
"""Plot scheme-2 vertical support S(t) and window integral for ketchup extend phase."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
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
from sim.slip_vertical_support import VerticalSupportWindow, gravity_up, measure_vertical_support

SPIDER = ROOT / "third_party" / "spider"
OUT_DIR = ROOT / "data" / "scheme2_curves"
DOCS_PLOT = ROOT / "docs" / "assets" / "plots"

SIM_DT = 0.01
EXTEND_S = 2.0
MIMIC_S = 1.0
LIFT_M = 0.10
WINDOW_S = 0.5


@dataclass
class RunTrace:
    name: str
    friction_scale: float
    sim_time: list[float] = field(default_factory=list)
    phase: list[str] = field(default_factory=list)
    object_z: list[float] = field(default_factory=list)
    n_contacts: list[int] = field(default_factory=list)
    support_z: list[float] = field(default_factory=list)
    support_normal_z: list[float] = field(default_factory=list)
    support_tangent_z: list[float] = field(default_factory=list)
    support_ratio: list[float] = field(default_factory=list)
    window_integral: list[float] = field(default_factory=list)
    mg: float = 0.0


def _run_trace(name: str, friction_scale: float) -> RunTrace:
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
    g_hat = gravity_up(model)

    trace = RunTrace(name=name, friction_scale=friction_scale)
    win = VerticalSupportWindow(WINDOW_S, SIM_DT)

    data.qpos[:] = qpos[0]
    data.qvel[:] = qvel[0]
    data.ctrl[:] = ctrl[0]
    mujoco.mj_forward(model, data)

    def record(phase_name: str) -> None:
        r = measure_vertical_support(model, data, hand_geoms, object_geoms, g_hat=g_hat)
        trace.mg = r.mg
        trace.sim_time.append(float(data.time))
        trace.phase.append(phase_name)
        trace.object_z.append(float(data.xpos[object_id][2]))
        trace.n_contacts.append(r.n_contacts)
        trace.support_z.append(r.support_z)
        trace.support_normal_z.append(r.support_normal_z)
        trace.support_tangent_z.append(r.support_tangent_z)
        trace.support_ratio.append(r.support_ratio)
        trace.window_integral.append(win.push(r.support_z))

    for c in ctrl:
        data.ctrl[:] = c
        mujoco.mj_step(model, data)
        record("trajectory")

    extend_ctrl = build_extend_mimic_lift_controls(
        ctrl, sim_dt=SIM_DT, extend_s=EXTEND_S, mimic_s=MIMIC_S, lift_m=LIFT_M
    )
    win.reset()
    for c in extend_ctrl:
        data.ctrl[:] = c
        mujoco.mj_step(model, data)
        record("extend")

    return trace


def _shade_extend(ax, t_extend_start: float, t_end: float) -> None:
    ax.axvspan(t_extend_start, t_end, color="#fff3cd", alpha=0.35, label="extend 段")


def plot_traces(traces: list[RunTrace], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # --- Figure 1: S(t) and rho(t) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = {"baseline": "#2ecc71", "friction_div2": "#e74c3c"}

    for tr in traces:
        t = np.array(tr.sim_time)
        c = colors.get(tr.name, "#3498db")
        t_ext = t[len(t) - len([p for p in tr.phase if p == "extend"])]
        _shade_extend(axes[0], t_ext, t[-1])
        _shade_extend(axes[1], t_ext, t[-1])

        axes[0].plot(t, tr.support_z, label=f"{tr.name}  S(t) [N]", color=c, lw=1.5)
        axes[1].plot(t, tr.support_ratio, label=f"{tr.name}  S/mg", color=c, lw=1.5)

    for ax in traces[:1]:
        mg = traces[0].mg
        axes[0].axhline(mg, color="gray", ls="--", lw=1, alpha=0.7, label=f"mg={mg:.2f}N")
        axes[1].axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="S/mg=1")

    axes[0].set_ylabel("Vertical support S(t) [N]")
    axes[1].set_ylabel("Support ratio S / (mg)")
    axes[1].set_xlabel("Simulation time [s]")
    axes[0].set_title("Scheme 2: vertical contact force (normal + tangential projection)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "scheme2_support_z_and_ratio.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    # --- Figure 2: normal vs tangent vertical components (extend zoom) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for tr in traces:
        t = np.array(tr.sim_time)
        mask = np.array(tr.phase) == "extend"
        c = colors.get(tr.name, "#3498db")
        ax.plot(t[mask], np.array(tr.support_normal_z)[mask], label=f"{tr.name}  Σ(f_n·ĝ)", color=c, lw=1.5)
        ax.plot(
            t[mask],
            np.array(tr.support_tangent_z)[mask],
            label=f"{tr.name}  Σ(f_t·ĝ)",
            color=c,
            lw=1.0,
            ls="--",
            alpha=0.8,
        )
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Vertical projection [N]")
    ax.set_title("Extend phase: normal vs tangential force · g_hat (upward part)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "scheme2_normal_tangent_extend.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    # --- Figure 3: sliding-window integral ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for tr in traces:
        t = np.array(tr.sim_time)
        mask = np.array(tr.phase) == "extend"
        c = colors.get(tr.name, "#3498db")
        ax.plot(
            t[mask],
            np.array(tr.window_integral)[mask],
            label=f"{tr.name}  ∫_W S dt  (W={WINDOW_S}s)",
            color=c,
            lw=2,
        )
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel(f"Window integral [{WINDOW_S}s] [N·s]")
    ax.set_title(f"Scheme 2: sliding-window integral of S(t) (extend phase)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p3 = out_dir / "scheme2_window_integral_extend.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths.append(p3)

    # --- Figure 4: object z + contacts ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for tr in traces:
        t = np.array(tr.sim_time)
        c = colors.get(tr.name, "#3498db")
        axes[0].plot(t, tr.object_z, label=tr.name, color=c, lw=1.5)
        axes[1].plot(t, tr.n_contacts, label=tr.name, color=c, lw=1.5)
    axes[0].set_ylabel("Object z [m]")
    axes[1].set_ylabel("Hand-object contacts")
    axes[1].set_xlabel("Simulation time [s]")
    axes[0].set_title("Reference: object height and contact count")
    axes[0].legend()
    axes[1].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    p4 = out_dir / "scheme2_object_z_contacts.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    paths.append(p4)

    # --- Figure 5: extend zoom S(t) normalized ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for tr in traces:
        t = np.array(tr.sim_time)
        mask = np.array(tr.phase) == "extend"
        t_e = t[mask]
        S_e = np.array(tr.support_z)[mask]
        z_e = np.array(tr.object_z)[mask]
        c = colors.get(tr.name, "#3498db")
        S0 = S_e[0] if S_e[0] > 1e-6 else 1.0
        axes[0].plot(t_e, S_e / S0, label=f"{tr.name}  S/S0", color=c, lw=2)
        axes[1].plot(t_e, z_e * 100, label=f"{tr.name}  object z", color=c, lw=2)

    axes[0].axhline(0.7, color="gray", ls="--", alpha=0.6, label="70% threshold")
    axes[0].set_ylabel("Normalized support S/S0")
    axes[1].set_ylabel("Object z [cm]")
    axes[1].set_xlabel("Simulation time [s]")
    axes[0].set_title("Extend phase zoom: support collapse vs stable lift")
    axes[0].legend()
    axes[1].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    p5 = out_dir / "scheme2_extend_zoom_normalized.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    paths.append(p5)

    return paths


def main() -> None:
    if not DEFAULT_WORKSPACE.joinpath("scene.xml").exists():
        print("Run: python3 scripts/build_spider_ketchup_right.py", file=sys.stderr)
        sys.exit(1)

    print("Running baseline (friction×1.0)...")
    baseline = _run_trace("baseline", friction_scale=1.0)
    print("Running friction÷2 (friction×0.5)...")
    div2 = _run_trace("friction_div2", friction_scale=0.5)

    traces = [baseline, div2]
    plot_paths = plot_traces(traces, OUT_DIR)

    DOCS_PLOT.mkdir(parents=True, exist_ok=True)
    for p in plot_paths:
        dst = DOCS_PLOT / p.name
        dst.write_bytes(p.read_bytes())

    summary = {
        "window_s": WINDOW_S,
        "sim_dt": SIM_DT,
        "traces": {
            tr.name: {
                "friction_scale": tr.friction_scale,
                "mg": tr.mg,
                "extend_support_z_mean": float(
                    np.mean([s for s, ph in zip(tr.support_z, tr.phase) if ph == "extend"])
                ),
                "extend_support_ratio_mean": float(
                    np.mean([r for r, ph in zip(tr.support_ratio, tr.phase) if ph == "extend"])
                ),
                "extend_window_integral_end": float(
                    [w for w, ph in zip(tr.window_integral, tr.phase) if ph == "extend"][-1]
                ),
            }
            for tr in traces
        },
    }
    (OUT_DIR / "scheme2_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nPlots:")
    for p in plot_paths:
        print(f"  {p}")
    print(f"  docs copy: {DOCS_PLOT}")
    print("\nSummary:")
    for name, stats in summary["traces"].items():
        print(
            f"  {name}: extend mean S={stats['extend_support_z_mean']:.2f}N "
            f"ratio={stats['extend_support_ratio_mean']:.2f} "
            f"I_W_end={stats['extend_window_integral_end']:.2f}N·s"
        )


if __name__ == "__main__":
    main()
