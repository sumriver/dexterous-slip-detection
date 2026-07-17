#!/usr/bin/env python3
"""Generate NN-1 presentation figures from sweep / closed-loop JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/opt/cursor/artifacts")
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "slip_nn"


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f1419",
            "axes.facecolor": "#1a222d",
            "axes.edgecolor": "#3d4f61",
            "axes.labelcolor": "#e7eef7",
            "text.color": "#e7eef7",
            "xtick.color": "#b7c4d4",
            "ytick.color": "#b7c4d4",
            "grid.color": "#2a3848",
            "font.size": 11,
        }
    )


def fig_teacher_false_triggers():
    """Bar: baseline nn_slip for y_fused / y_scheme2 / y_event vs rule."""
    rows = {
        "Rule scheme-2": 191,
        "NN y_fused\nτ=0.99": 200,
        "NN y_scheme2\nτ=0.9": 190,
        "NN y_event\nτ=0.7 · confirm=15": 93,
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    names = list(rows.keys())
    vals = list(rows.values())
    colors = ["#6b7c8f", "#c45c5c", "#c45c5c", "#3dba7a"]
    bars = ax.bar(names, vals, color=colors, width=0.65)
    ax.axhline(100, color="#f0c14b", ls="--", lw=1.5, label="Gate: < 100 / 200")
    ax.set_ylabel("Baseline extend slip steps")
    ax.set_title("Baseline over-trigger: teachers vs NN-1 fix")
    ax.set_ylim(0, 220)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, str(v), ha="center", color="#e7eef7")
    fig.tight_layout()
    path = OUT / "nn1_baseline_false_triggers.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    # also copy under data for repo
    (DATA / "figs").mkdir(exist_ok=True)
    fig2_path = DATA / "figs" / "nn1_baseline_false_triggers.png"
    # re-save via copy
    import shutil

    shutil.copy(path, fig2_path)
    return path


def fig_closedloop_compare():
    smoke = json.loads((DATA / "closedloop_smoke.json").read_text())
    cases = {c["name"]: c for c in smoke["cases"]}
    labels = ["baseline", "friction_div2"]
    dz = [cases[k]["extend_dz_cm"] for k in labels]
    slips = [cases[k]["nn_slip_events"] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ax = axes[0]
    colors = ["#5b9fd4", "#e08a3c"]
    ax.bar(labels, dz, color=colors, width=0.55)
    ax.axhline(6.0, color="#f0c14b", ls="--", lw=1.4, label="Gate Δz ≥ 6 cm")
    ax.set_ylabel("Extend Δz (cm)")
    ax.set_title("Closed-loop lift (NN-1 y_event)")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(framealpha=0.2)
    for i, v in enumerate(dz):
        ax.text(i, v + 0.25, f"{v:.1f}", ha="center")

    ax = axes[1]
    ax.bar(labels, slips, color=colors, width=0.55)
    ax.axhline(100, color="#f0c14b", ls="--", lw=1.4, label="Baseline gate < 100")
    ax.set_ylabel("Raw NN fire steps / 200")
    ax.set_title("NN fire count on extend")
    ax.set_ylim(0, 220)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(framealpha=0.2)
    for i, v in enumerate(slips):
        ax.text(i, v + 4, str(v), ha="center")

    fig.suptitle("NN-1 closed-loop smoke · τ=0.7 · confirm=15", y=1.02)
    fig.tight_layout()
    path = OUT / "nn1_closedloop_metrics.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    import shutil

    shutil.copy(path, DATA / "figs" / "nn1_closedloop_metrics.png")
    return path


def fig_tau_sweep_event():
    raw = json.loads((DATA / "tau_sweep_y_event.json").read_text())
    # support both list and {rows:[]}
    rows = raw if isinstance(raw, list) else raw.get("rows", [])
    if not rows:
        return None
    # Prefer confirm=15 style if present in later files; else plot available
    by_tau = {}
    for r in rows:
        by_tau.setdefault(r["tau"], {})[r["case"]] = r
    taus = sorted(by_tau.keys())
    base = [by_tau[t].get("baseline", {}).get("nn_slip", by_tau[t].get("baseline", {}).get("nn_slip_events", 0)) for t in taus]
    # keys vary
    def _slip(t, case):
        d = by_tau[t].get(case, {})
        return d.get("nn_slip", d.get("nn_slip_events", 0))

    def _dz(t, case):
        d = by_tau[t].get(case, {})
        return d.get("dz", d.get("extend_dz_cm", 0))

    base_s = [_slip(t, "baseline") for t in taus]
    fd_s = [_slip(t, "friction_div2") for t in taus]
    fd_dz = [_dz(t, "friction_div2") for t in taus]

    fig, ax1 = plt.subplots(figsize=(8.2, 4.5))
    ax1.plot(taus, base_s, "o-", color="#5b9fd4", lw=2, label="baseline nn_slip")
    ax1.plot(taus, fd_s, "s-", color="#e08a3c", lw=2, label="friction÷2 nn_slip")
    ax1.axhline(100, color="#f0c14b", ls="--", lw=1.2)
    ax1.set_xlabel("Threshold τ")
    ax1.set_ylabel("NN fire steps / 200")
    ax1.set_ylim(0, 220)
    ax1.grid(alpha=0.35)
    ax2 = ax1.twinx()
    ax2.plot(taus, fd_dz, "^-", color="#3dba7a", lw=2, label="friction÷2 Δz (cm)")
    ax2.set_ylabel("Extend Δz (cm)", color="#3dba7a")
    ax2.tick_params(axis="y", labelcolor="#3dba7a")
    ax2.axhline(6, color="#3dba7a", ls=":", alpha=0.7)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right", framealpha=0.2)
    ax1.set_title("y_event τ sweep (see tau_sweep_y_event.json)")
    fig.tight_layout()
    path = OUT / "nn1_tau_sweep_y_event.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    import shutil

    shutil.copy(path, DATA / "figs" / "nn1_tau_sweep_y_event.png")
    return path


def fig_offline_metrics():
    val = json.loads((ROOT / "models" / "slip_nn" / "eval_val.json").read_text())
    # Also show at default train threshold if present
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    metrics = ["precision", "recall", "f1"]
    vals = [val.get(m, 0) for m in metrics]
    bars = ax.bar(metrics, vals, color=["#5b9fd4", "#e08a3c", "#3dba7a"], width=0.55)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="#f0c14b", ls="--", lw=1.2, label="F1 gate 0.90 (soft for y_event)")
    ax.set_title(f"Offline val @ τ={val.get('threshold', '?')} · label={val.get('label')}")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(framealpha=0.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center")
    fig.tight_layout()
    path = OUT / "nn1_offline_val_metrics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    import shutil

    shutil.copy(path, DATA / "figs" / "nn1_offline_val_metrics.png")
    return path


def main() -> None:
    _style()
    paths = [
        fig_teacher_false_triggers(),
        fig_closedloop_compare(),
        fig_tau_sweep_event(),
        fig_offline_metrics(),
    ]
    print("Wrote:")
    for p in paths:
        if p:
            print(" ", p)


if __name__ == "__main__":
    main()
