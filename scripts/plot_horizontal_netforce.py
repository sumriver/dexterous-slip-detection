#!/usr/bin/env python3
"""Horizontal NET force: magnitude + direction, with a no-slip baseline.

Tests the hypothesis: a stable (non-slipping) grasp has opposing contact
forces that cancel, so the net horizontal force |F_h| ~ 0 and its direction
is undefined/wandering; when the object slips the forces stop cancelling, so
|F_h| grows and the direction locks onto the slip/push direction.

Frame (see slip_horizontal):
  X = 四指平展方向 (four-finger extension, horizontal)
  Y = X x up (perpendicular, right)

Cases: baseline (friction x1, no slip) vs friction div2 / div4 / div8.
The signed net force Fx, Fy is smoothed (short moving average) to remove
transient contact spikes before magnitude/direction are computed — this is
the quasi-static net force the balance argument refers to.

Outputs plots + JSON to data/horizontal_netforce/ and docs/assets/plots/.
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
    compute_hand_horizontal_frame,
    measure_horizontal_forces,
)

SPIDER = ROOT / "third_party" / "spider"
OUT_DIR = ROOT / "data" / "horizontal_netforce"
DOCS_PLOT = ROOT / "docs" / "assets" / "plots"

SIM_DT = 0.01
HOLD_S = 1.5   # quasi-static hold (no lift) to test net-force~0 hypothesis
EXTEND_S = 2.0
MIMIC_S = 1.0
LIFT_M = 0.10
SMOOTH_WINDOW_S = 0.1  # moving average on signed Fx, Fy (remove transient spikes)
DIR_MIN_FORCE = 2.0    # only report direction when |F_h| exceeds this [N]

CASES = [
    ("baseline", 1.0, "#2ecc71"),
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
    sum_abs: list[float] = field(default_factory=list)
    obj_x: list[float] = field(default_factory=list)
    obj_y: list[float] = field(default_factory=list)
    obj_z: list[float] = field(default_factory=list)


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

    def record(phase_name: str) -> None:
        frame = compute_hand_horizontal_frame(model, data)
        r = measure_horizontal_forces(model, data, hand_geoms, object_geoms, frame)
        trace.sim_time.append(float(data.time))
        trace.phase.append(phase_name)
        trace.n_contacts.append(r.n_contacts)
        trace.fx.append(r.fx)
        trace.fy.append(r.fy)
        trace.sum_abs.append(r.sum_abs_horiz)
        p = data.xpos[object_id]
        trace.obj_x.append(float(p[0]))
        trace.obj_y.append(float(p[1]))
        trace.obj_z.append(float(p[2]))

    for c in ctrl:
        data.ctrl[:] = c
        mujoco.mj_step(model, data)
        record("trajectory")

    # Quasi-static hold: keep gripping (mimic last 1s) with NO wrist lift.
    hold_ctrl = build_extend_mimic_lift_controls(
        ctrl, sim_dt=SIM_DT, extend_s=HOLD_S, mimic_s=MIMIC_S, lift_m=0.0
    )
    for c in hold_ctrl:
        data.ctrl[:] = c
        mujoco.mj_step(model, data)
        record("hold")

    # Lift: mimic grip + wrist lift 10cm.
    extend_ctrl = build_extend_mimic_lift_controls(
        ctrl, sim_dt=SIM_DT, extend_s=EXTEND_S, mimic_s=MIMIC_S, lift_m=LIFT_M
    )
    for c in extend_ctrl:
        data.ctrl[:] = c
        mujoco.mj_step(model, data)
        record("extend")

    return trace


def _smooth(arr: list[float], window_steps: int) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if window_steps <= 1 or a.size == 0:
        return a
    kernel = np.ones(window_steps) / window_steps
    return np.convolve(a, kernel, mode="same")


def _derived(tr: RunTrace):
    w = max(1, int(round(SMOOTH_WINDOW_S / SIM_DT)))
    fxs = _smooth(tr.fx, w)
    fys = _smooth(tr.fy, w)
    mag = np.hypot(fxs, fys)
    ang = np.degrees(np.arctan2(fys, fxs))
    ang_masked = np.where(mag >= DIR_MIN_FORCE, ang, np.nan)
    sum_abs = np.asarray(tr.sum_abs)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sum_abs > 1e-9, np.hypot(tr.fx, tr.fy) / sum_abs, np.nan)
    return fxs, fys, mag, ang_masked, ratio


def _phase_start_time(tr: RunTrace, want: str) -> float:
    for t, ph in zip(tr.sim_time, tr.phase):
        if ph == want:
            return t
    return tr.sim_time[-1] if tr.sim_time else 0.0


def _mark_phases(ax, t_hold: float, t_lift: float) -> None:
    ax.axvline(t_hold, color="gray", ls=":", lw=1, alpha=0.7)
    ax.axvline(t_lift, color="black", ls=":", lw=1, alpha=0.7)


def plot_traces(traces: list[RunTrace], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    t_hold = _phase_start_time(traces[0], "hold") if traces else 0.0
    t_lift = _phase_start_time(traces[0], "extend") if traces else 0.0
    derived = {tr.name: _derived(tr) for tr in traces}

    # --- Figure 1: net horizontal force MAGNITUDE + DIRECTION ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for tr in traces:
        t = np.array(tr.sim_time)
        _, _, mag, ang, _ = derived[tr.name]
        axes[0].plot(t, mag, color=tr.color, lw=1.6, label=f"{tr.name}")
        axes[1].plot(t, ang, color=tr.color, lw=1.2, marker=".", ms=2, ls="none",
                     label=f"{tr.name}")
    for ax in axes:
        _mark_phases(ax, t_hold, t_lift)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("|F_h|  net horizontal force  [N]")
    axes[1].set_ylabel("direction  [deg]  (0=+X finger, 90=+Y right)")
    axes[1].set_ylim(-180, 180)
    axes[1].set_yticks([-180, -90, 0, 90, 180])
    axes[1].set_xlabel("Simulation time [s]  (gray=hold start, black=lift start)")
    axes[0].set_title(
        f"Net horizontal force magnitude + direction (smoothed {int(SMOOTH_WINDOW_S*1000)}ms)"
    )
    fig.tight_layout()
    p = out_dir / "horizontal_netforce_mag_dir.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- Figure 2: imbalance ratio |net| / sum|contact| ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for tr in traces:
        t = np.array(tr.sim_time)
        _, _, _, _, ratio = derived[tr.name]
        ax.plot(t, ratio, color=tr.color, lw=1.4, label=f"{tr.name}")
    _mark_phases(ax, t_hold, t_lift)
    ax.set_ylabel("imbalance r = |ΣF| / Σ|F|")
    ax.set_xlabel("Simulation time [s]")
    ax.set_title("Force imbalance ratio (0 = balanced/cancelling, 1 = fully net)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / "horizontal_imbalance_ratio.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- Figure 3: 2D net-force vector trajectory (magnitude+direction together) ---
    fig, axes = plt.subplots(1, len(traces), figsize=(4.2 * len(traces), 4.4))
    if len(traces) == 1:
        axes = [axes]
    for ax, tr in zip(axes, traces):
        fxs, fys, mag, _, _ = derived[tr.name]
        ph = np.array(tr.phase)
        ext = (ph == "hold") | (ph == "extend")
        # color post-grasp points by time to show evolution
        te = np.array(tr.sim_time)[ext]
        sc = ax.scatter(np.array(fxs)[ext], np.array(fys)[ext], c=te, cmap="viridis",
                        s=8, alpha=0.8)
        ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
        ax.axvline(0, color="gray", lw=0.8, alpha=0.6)
        lim = max(50.0, float(np.nanmax(mag)) * 1.05)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(tr.name, fontsize=10)
        ax.set_xlabel("Fx [N] (finger dir)")
        ax.set_ylabel("Fy [N] (right)")
        ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=axes, label="time [s]", shrink=0.8)
    fig.suptitle("Net horizontal force vector during hold+lift (point = Fx,Fy)")
    p = out_dir / "horizontal_netforce_vector.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- Figure 4: object horizontal drift (reference) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for tr in traces:
        t = np.array(tr.sim_time)
        dxy = np.hypot(np.array(tr.obj_x) - tr.obj_x[0], np.array(tr.obj_y) - tr.obj_y[0]) * 100
        ax.plot(t, dxy, color=tr.color, lw=1.8, label=f"{tr.name}")
    _mark_phases(ax, t_hold, t_lift)
    ax.set_ylabel("object horizontal drift [cm]")
    ax.set_xlabel("Simulation time [s]")
    ax.set_title("Reference: object horizontal drift")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / "horizontal_netforce_drift.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    return paths


def _phase(vals, phase, want):
    return [v for v, ph in zip(vals, phase) if ph == want]


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
        "frame": "X=four-finger extension, Y=X x up (perpendicular right)",
        "smooth_window_s": SMOOTH_WINDOW_S,
        "dir_min_force_N": DIR_MIN_FORCE,
        "traces": {},
    }
    for tr in traces:
        fxs, fys, mag, ang, ratio = _derived(tr)
        ph = np.array(tr.phase)
        n = np.array(tr.n_contacts)

        def _stats(mask):
            m = mask & (n > 0)
            if not m.any():
                return {"contact_steps": 0, "netmag_mean": None,
                        "imbalance_mean": None, "dir_mean_deg": None, "dir_std_deg": None}
            return {
                "contact_steps": int(m.sum()),
                "netmag_mean": float(np.mean(mag[m])),
                "imbalance_mean": float(np.nanmean(ratio[m])),
                "dir_mean_deg": float(np.nanmean(ang[m])),
                "dir_std_deg": float(np.nanstd(ang[m])),
            }

        summary["traces"][tr.name] = {
            "friction_scale": tr.friction_scale,
            "hold": _stats(ph == "hold"),
            "lift": _stats(ph == "extend"),
            "object_horiz_drift_cm": float(
                np.hypot(tr.obj_x[-1] - tr.obj_x[0], tr.obj_y[-1] - tr.obj_y[0]) * 100
            ),
            "object_dz_cm": float((tr.obj_z[-1] - tr.obj_z[0]) * 100),
        }
    (OUT_DIR / "horizontal_netforce_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nPlots:")
    for p in plot_paths:
        print(f"  {p}")
    print("\nSummary (HOLD phase = quasi-static, contact steps only):")
    for name, s in summary["traces"].items():
        h = s["hold"]
        print(
            f"  {name}: hold |F_h|~{h['netmag_mean']} N  imbalance~{h['imbalance_mean']}  "
            f"dir_std={h['dir_std_deg']} deg  steps={h['contact_steps']}  "
            f"| drift={s['object_horiz_drift_cm']:.1f}cm dz={s['object_dz_cm']:.1f}cm"
        )


if __name__ == "__main__":
    main()
