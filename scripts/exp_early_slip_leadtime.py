#!/usr/bin/env python3
"""Early slip-warning LEAD TIME on the friction sweep (÷2/÷4/÷8).

Question: how much earlier than the macroscopic drop can we detect a slip?

We run grasp trajectory -> hold -> lift and, gated to the post-grasp window
(hold+lift, where the object should stay put relative to the hand), track:

  * util = rho/mu = (sum|f_t|)/(mu*sum|f_n|)  — friction-cone utilization (Tier-1)
  * descent = how far the object has dropped RELATIVE TO THE HAND from its
    held baseline (robust to grip micro-motion).

Two slip levels on the relative descent:
  * incipient slip: descent crosses INCIPIENT_CM (micro-slip, early)
  * macroscopic slip: descent crosses MACRO_CM or contacts collapse (drop)

lead_time = t(macroscopic) - t(incipient)   [and vs util warning]

Findings (see docs): the ÷ cases fail abruptly at lift onset, so force
utilization saturates only ~tens of ms before release, while the incipient
relative descent leads the macroscopic drop by ~0.1-0.2 s. ÷4/÷8 lose the
object already during the grasp trajectory (no post-grasp window).

Outputs plot + JSON to data/early_slip/ and docs/assets/plots/.
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
from sim.slip_early_warning import measure_early_warning, object_pair_friction

SPIDER = ROOT / "third_party" / "spider"
OUT_DIR = ROOT / "data" / "early_slip"
DOCS_PLOT = ROOT / "docs" / "assets" / "plots"

SIM_DT = 0.01
HOLD_S = 1.0
EXTEND_S = 2.0
MIMIC_S = 1.0
LIFT_M = 0.10
SMOOTH_STEPS = 8
INCIPIENT_CM = 1.5    # relative descent for incipient (micro) slip
MACRO_CM = 6.0        # relative descent for macroscopic slip
UTIL_TH = 0.7         # friction utilization warning threshold
SUSTAIN = 3
BASELINE_S = 0.3      # window at hold start to set the "held" baseline

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
    mu: float = 0.0
    t: list[float] = field(default_factory=list)      # time since hold start
    util: list[float] = field(default_factory=list)
    n_contacts: list[int] = field(default_factory=list)
    rel_z: list[float] = field(default_factory=list)  # obj_z - palm_z


def _run(name, fscale, color) -> RunTrace:
    cfg = SpiderTaskConfig(
        dataset_dir=SPIDER / "example_datasets",
        dataset_name="arcticv2", robot_type="xhand", embodiment_type="right",
        task="s01-ketchup_use_01", workspace_root=DEFAULT_WORKSPACE,
    )
    model = mujoco.MjModel.from_xml_path(str(cfg.scene_path))
    apply_object_physics(model, friction_scale=fscale)
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    q, v, c = load_trajectory_arrays(cfg.trajectory_path, model, cfg.data_type)
    q, v, c = upsample_controls(q, v, c, SIM_DT, 0.02)

    hg = get_spider_hand_collision_geom_ids(model)
    og = get_object_geom_ids(model, "right_object")
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    mu = object_pair_friction(model, og)

    data.qpos[:] = q[0]; data.qvel[:] = v[0]; data.ctrl[:] = c[0]
    mujoco.mj_forward(model, data)
    for cc in c:  # grasp trajectory
        data.ctrl[:] = cc
        mujoco.mj_step(model, data)

    tr = RunTrace(name=name, friction_scale=fscale, color=color, mu=mu)
    t0 = float(data.time)

    def rec():
        r = measure_early_warning(model, data, hg, og, mu)
        tr.t.append(float(data.time) - t0)
        tr.util.append(r.util_wmean)
        tr.n_contacts.append(r.n_contacts)
        tr.rel_z.append(float(data.xpos[oid][2] - data.site_xpos[palm][2]))

    hold_ctrl = build_extend_mimic_lift_controls(c, sim_dt=SIM_DT, extend_s=HOLD_S, mimic_s=MIMIC_S, lift_m=0.0)
    lift_ctrl = build_extend_mimic_lift_controls(c, sim_dt=SIM_DT, extend_s=EXTEND_S, mimic_s=MIMIC_S, lift_m=LIFT_M)
    for seq in (hold_ctrl, lift_ctrl):
        for cc in seq:
            data.ctrl[:] = cc
            mujoco.mj_step(model, data)
            rec()
    return tr


def _smooth(a, w):
    a = np.asarray(a, float)
    if w <= 1 or a.size == 0:
        return a
    return np.convolve(a, np.ones(w) / w, mode="same")


def _first_ge(sig, th, t, sustain):
    run = 0
    for i in range(len(sig)):
        if np.isfinite(sig[i]) and sig[i] >= th:
            run += 1
            if run >= sustain:
                return float(t[i - sustain + 1])
        else:
            run = 0
    return None


def _contact_collapse_time(tr, t):
    established = False
    run0 = 0
    for i, n in enumerate(tr.n_contacts):
        if n >= 3:
            established = True
        if established:
            if n < 2:
                run0 += 1
                if run0 >= SUSTAIN:
                    return float(t[i - SUSTAIN + 1])
            else:
                run0 = 0
    return None


def analyze(tr: RunTrace):
    t = np.asarray(tr.t)
    util = _smooth(tr.util, SMOOTH_STEPS)
    rel_z = np.asarray(tr.rel_z)
    nb = max(1, int(BASELINE_S / SIM_DT))
    baseline = float(np.median(rel_z[:nb])) if rel_z.size else 0.0

    # guard: is there a stable grasp at the start of the hold window?
    stable = bool(np.mean(tr.n_contacts[:nb]) >= 3.0) if tr.n_contacts else False
    if not stable:
        return dict(
            mu=tr.mu, stable_grasp_at_hold=False,
            note="object lost during grasp trajectory; no post-grasp window",
            t_incipient=None, t_macro=None, t_util_warn=None,
            lead_incipient_s=None, lead_util_s=None,
        ), dict(t=t, util=util, descent=np.zeros_like(t),
                t_incip=None, t_macro=None, t_util=None)
    descent_cm = (baseline - rel_z) * 100.0  # positive = dropped relative to hand
    descent_s = _smooth(descent_cm, SMOOTH_STEPS)

    t_incip = _first_ge(descent_s, INCIPIENT_CM, t, SUSTAIN)
    t_macro_desc = _first_ge(descent_s, MACRO_CM, t, SUSTAIN)
    t_collapse = _contact_collapse_time(tr, t)
    t_macro = min([x for x in (t_macro_desc, t_collapse) if x is not None], default=None)
    t_util = _first_ge(util, UTIL_TH, t, SUSTAIN)

    def lead(a, b):
        return None if (a is None or b is None) else round(a - b, 3)

    return dict(
        mu=tr.mu, stable_grasp_at_hold=True, baseline_relz_cm=baseline * 100,
        t_incipient=t_incip, t_macro=t_macro, t_util_warn=t_util,
        lead_incipient_s=lead(t_macro, t_incip),
        lead_util_s=lead(t_macro, t_util),
    ), dict(t=t, util=util, descent=descent_s,
            t_incip=t_incip, t_macro=t_macro, t_util=t_util)


def plot(traces, series):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for tr in traces:
        s = series[tr.name]
        axes[0].plot(s["t"], s["descent"], color=tr.color, lw=1.6, label=f"{tr.name} descent")
        axes[1].plot(s["t"], s["util"], color=tr.color, lw=1.6, label=f"{tr.name} util=rho/mu")
        if s["t_incip"] is not None:
            axes[0].axvline(s["t_incip"], color=tr.color, ls=":", lw=1.3, alpha=0.8)
        if s["t_macro"] is not None:
            axes[0].axvline(s["t_macro"], color=tr.color, ls="--", lw=1.3, alpha=0.7)
    axes[0].axhline(INCIPIENT_CM, color="gray", ls=":", lw=1, alpha=0.6, label=f"incipient {INCIPIENT_CM}cm")
    axes[0].axhline(MACRO_CM, color="gray", ls="--", lw=1, alpha=0.6, label=f"macro {MACRO_CM}cm")
    axes[1].axhline(UTIL_TH, color="gray", ls="--", lw=1, alpha=0.6, label=f"warn {UTIL_TH}")
    axes[0].set_ylabel("object descent vs hand [cm]")
    axes[1].set_ylabel("friction utilization rho/mu")
    axes[1].set_xlabel("Time since hold start [s]  (dotted=incipient, dashed=macroscopic)")
    axes[0].set_title("Early slip warning: incipient relative descent leads macroscopic drop")
    axes[1].set_ylim(0, 1.2)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    p = OUT_DIR / "early_slip_leadtime.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return [p]


def main():
    if not DEFAULT_WORKSPACE.joinpath("scene.xml").exists():
        print("Run: python3 scripts/build_spider_ketchup_right.py", file=sys.stderr)
        sys.exit(1)

    traces, series = [], {}
    summary = {"params": dict(incipient_cm=INCIPIENT_CM, macro_cm=MACRO_CM,
                              util_th=UTIL_TH, smooth_steps=SMOOTH_STEPS), "cases": {}}
    for name, fs, color in CASES:
        print(f"Running {name} ...")
        tr = _run(name, fs, color)
        traces.append(tr)
        stats, s = analyze(tr)
        series[name] = s
        summary["cases"][name] = stats

    plot_paths = plot(traces, series)
    DOCS_PLOT.mkdir(parents=True, exist_ok=True)
    for p in plot_paths:
        (DOCS_PLOT / p.name).write_bytes(p.read_bytes())
    (OUT_DIR / "early_slip_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nLead time (macroscopic - earlier signal; positive = warned early):")
    for name, s in summary["cases"].items():
        if not s.get("stable_grasp_at_hold", True):
            print(f"  {name} (mu={s['mu']:.3f}): {s['note']}")
            continue
        print(
            f"  {name} (mu={s['mu']:.3f}): incipient@{s['t_incipient']} macro@{s['t_macro']} "
            f"util_warn@{s['t_util_warn']}  |  lead_incipient={s['lead_incipient_s']}s "
            f"lead_util={s['lead_util_s']}s"
        )
    print(f"\nPlot: {plot_paths[0]}")


if __name__ == "__main__":
    main()
