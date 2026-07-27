#!/usr/bin/env python3
"""Closed-loop ranking for models trained on the unified dataset.

All models share the same train windows (``data/slip_nn_unified``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ketchup_robustness_sweep import CaseSpec, EXTEND_STEPS, _run_case  # noqa: E402

OUT = ROOT / "data" / "slip_eval" / "unified_suite_latest.json"

# Frontier + economy cells (discriminative, same for everyone)
CASES = [
    # frontier
    ("A", CaseSpec("mass_x2_s045", mass_scale=2.0, friction_scale=0.45, sweep="frontier"), 0.25),
    ("A", CaseSpec("mass_x4_s045", mass_scale=4.0, friction_scale=0.45, sweep="frontier"), 0.25),
    ("A", CaseSpec("mass_x2_s042", mass_scale=2.0, friction_scale=0.42, sweep="frontier"), 0.25),
    ("A", CaseSpec("friction_s045", friction_scale=0.45, sweep="frontier"), 0.25),
    ("A", CaseSpec("mass_x4_div2_g012", mass_scale=4.0, friction_scale=0.50, sweep="frontier"), 0.12),
    ("A", CaseSpec("mass_x2_div2_g010", mass_scale=2.0, friction_scale=0.50, sweep="frontier"), 0.10),
    ("A", CaseSpec("mass_x8_s045_g015", mass_scale=8.0, friction_scale=0.45, sweep="frontier"), 0.15),
    # economy
    ("B", CaseSpec("baseline", sweep="economy"), 0.25),
    ("B", CaseSpec("friction_div2", friction_scale=0.5, sweep="economy"), 0.25),
    ("B", CaseSpec("mass_x4", mass_scale=4.0, sweep="economy"), 0.25),
    # envelope
    ("C", CaseSpec("friction_s040", friction_scale=0.40, sweep="envelope"), 0.25),
]

DEFAULT_MODELS = {
    "nn2": {"dir": "models/slip_nn_unified_nn2", "policy_mode": "off", "label": "NN-2"},
    "p1": {"dir": "models/slip_nn_unified_p1", "policy_mode": "replace", "label": "P1 grip-only"},
    "p2_grip": {
        "dir": "models/slip_nn_unified_p2_grip",
        "policy_mode": "replace",
        "label": "P2 grip-only",
    },
    "p2_wrist": {
        "dir": "models/slip_nn_unified_p2_wrist",
        "policy_mode": "p2a",
        "label": "P2 grip+wrist",
    },
}


def _run_model(key: str, cfg: dict) -> dict:
    model_dir = ROOT / cfg["dir"] if not Path(cfg["dir"]).is_absolute() else Path(cfg["dir"])
    if not any(model_dir.glob("*.pt")):
        return {"model": key, "label": cfg["label"], "error": f"missing {model_dir}"}

    meta_path = model_dir / "train_meta.json"
    thr = 0.5
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        default = (
            0.99
            if meta.get("arch") in ("detect_and_policy", "detect_and_policy2", "tcn_multi", "multitask")
            else 0.5
        )
        thr = float(meta.get("default_threshold", default))

    cells = []
    for tier, spec, gcap in CASES:
        print(f"[{cfg['label']}] {tier} {spec.name} g≤{gcap} ...", flush=True)
        row = _run_case(
            spec,
            antislip_nn=True,
            nn_model_dir=model_dir,
            nn_threshold=thr,
            policy_mode=cfg["policy_mode"],
            antislip_grip_max=gcap,
        )
        gate_ok = row.extend_dz_cm >= 6.0 and row.extend_contact_steps >= EXTEND_STEPS
        cells.append(
            {
                "tier": tier,
                "name": spec.name,
                "grip_cap": gcap,
                "status": row.status,
                "gate_ok": bool(gate_ok),
                "extend_dz_cm": float(row.extend_dz_cm),
                "lift_margin_cm": float(row.extend_dz_cm - 6.0),
                "antislip_max_grip": float(row.antislip_max_grip),
                "nn_slip_events": int(row.nn_slip_events),
                "extend_contact_steps": int(row.extend_contact_steps),
            }
        )
        print(
            f"  -> {'PASS' if gate_ok else row.status} "
            f"dz={row.extend_dz_cm:+.1f} grip={row.antislip_max_grip:.3f}",
            flush=True,
        )

    front = [c for c in cells if c["tier"] == "A"]
    econ = [c for c in cells if c["tier"] == "B"]
    front_pass = [c for c in front if c["gate_ok"]]
    econ_pass = [c for c in econ if c["gate_ok"]]
    return {
        "model": key,
        "label": cfg["label"],
        "train_data": "data/slip_nn_unified",
        "nn_model_dir": str(model_dir),
        "frontier_pass_n": len(front_pass),
        "frontier_n": len(front),
        "frontier_pass_rate": len(front_pass) / len(front) if front else 0.0,
        "mean_lift_margin_frontier_pass": (
            float(sum(c["lift_margin_cm"] for c in front_pass) / len(front_pass))
            if front_pass
            else None
        ),
        "mean_grip_economy_pass": (
            float(sum(c["antislip_max_grip"] for c in econ_pass) / len(econ_pass))
            if econ_pass
            else None
        ),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified same-data closed-loop suite")
    parser.add_argument("--models-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    models = dict(DEFAULT_MODELS)
    if args.models_json and args.models_json.exists():
        override = json.loads(args.models_json.read_text())
        for k, d in override.items():
            if k in models:
                models[k]["dir"] = d
            elif k == "p2_grip_only":
                models["p2_grip"]["dir"] = d
            elif k == "p2":
                models["p2_wrist"]["dir"] = d

    results = [_run_model(k, cfg) for k, cfg in models.items()]
    ok = [m for m in results if not m.get("error")]

    def rank_key(m):
        return (
            m.get("frontier_pass_rate") or 0.0,
            m.get("mean_lift_margin_frontier_pass") or -1e9,
            -(m.get("mean_grip_economy_pass") or 1e9),
        )

    ranking = sorted(ok, key=rank_key, reverse=True)
    print("\n=== SAME-DATA ranking (all trained on slip_nn_unified) ===")
    for i, m in enumerate(ranking, 1):
        print(
            f"  #{i} {m['label']:18} frontier={m['frontier_pass_n']}/{m['frontier_n']} "
            f"({100*m['frontier_pass_rate']:.0f}%) "
            f"econ_grip={m['mean_grip_economy_pass']}"
        )

    # matrix
    names = [c[1].name for c in CASES]
    print(f"\n{'cell':24}", end="")
    for m in ok:
        print(f" {m['model']:>12}", end="")
    print()
    for name in names:
        print(f"{name:24}", end="")
        for m in ok:
            cell = next(x for x in m["cells"] if x["name"] == name)
            tag = "P" if cell["gate_ok"] else "F"
            print(f" {tag}{cell['extend_dz_cm']:+.0f}/{cell['antislip_max_grip']:.2f}".rjust(13), end="")
        print()

    out = {
        "fairness": "same_train_data=data/slip_nn_unified",
        "models": results,
        "ranking": [
            {
                "rank": i,
                "model": m["model"],
                "label": m["label"],
                "frontier_pass_n": m["frontier_pass_n"],
                "frontier_n": m["frontier_n"],
                "frontier_pass_rate": m["frontier_pass_rate"],
                "mean_lift_margin_frontier_pass": m["mean_lift_margin_frontier_pass"],
                "mean_grip_economy_pass": m["mean_grip_economy_pass"],
            }
            for i, m in enumerate(ranking, 1)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
