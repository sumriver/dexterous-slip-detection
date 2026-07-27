#!/usr/bin/env python3
"""Closed-loop ranking on unified same-train models.

Algorithm classes only (no seed-vs-seed fake ranking):
  - NN-2          detect-driven grip (policy_mode=off)
  - grip-only     frozen detect + grip head (replace)
  - grip+wrist    frozen detect + grip+wrist (p2a)

Optional second grip-only seed is reported as variance, not ranked.
Envelope cells (e.g. friction_s040) are diagnostic, not in ranking.
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

# A = frontier (ranked), B = economy (tie-break), C = envelope (diagnostic only)
CASES = [
    ("A", CaseSpec("mass_x2_s045", mass_scale=2.0, friction_scale=0.45, sweep="frontier"), 0.25),
    ("A", CaseSpec("mass_x4_s045", mass_scale=4.0, friction_scale=0.45, sweep="frontier"), 0.25),
    ("A", CaseSpec("mass_x2_s042", mass_scale=2.0, friction_scale=0.42, sweep="frontier"), 0.25),
    ("A", CaseSpec("friction_s045", friction_scale=0.45, sweep="frontier"), 0.25),
    ("A", CaseSpec("mass_x4_div2_g012", mass_scale=4.0, friction_scale=0.50, sweep="frontier"), 0.12),
    ("A", CaseSpec("mass_x2_div2_g010", mass_scale=2.0, friction_scale=0.50, sweep="frontier"), 0.10),
    ("A", CaseSpec("mass_x8_s045_g015", mass_scale=8.0, friction_scale=0.45, sweep="frontier"), 0.15),
    ("B", CaseSpec("baseline", sweep="economy"), 0.25),
    ("B", CaseSpec("friction_div2", friction_scale=0.5, sweep="economy"), 0.25),
    ("B", CaseSpec("mass_x4", mass_scale=4.0, sweep="economy"), 0.25),
    ("C", CaseSpec("friction_s040", friction_scale=0.40, sweep="envelope"), 0.25),
]

# Ranked algorithm classes (same train data, same backbone family)
RANKED_MODELS = {
    "nn2": {
        "dir": "models/slip_nn_unified_nn2",
        "policy_mode": "off",
        "label": "NN-2",
        "rank": True,
    },
    "grip": {
        "dir": "models/slip_nn_unified_grip",
        "policy_mode": "replace",
        "label": "grip-only",
        "rank": True,
        # fallbacks for older dirs
        "fallback_dirs": ["models/slip_nn_unified_p1"],
    },
    "wrist": {
        "dir": "models/slip_nn_unified_wrist",
        "policy_mode": "p2a",
        "label": "grip+wrist",
        "rank": True,
        "fallback_dirs": ["models/slip_nn_unified_p2_wrist"],
    },
}

# Variance-only (not ranked)
VARIANCE_MODELS = {
    "grip_s43": {
        "dir": "models/slip_nn_unified_grip_s43",
        "policy_mode": "replace",
        "label": "grip-only (seed43)",
        "rank": False,
        "fallback_dirs": ["models/slip_nn_unified_p2_grip"],
    },
}


def _resolve_dir(cfg: dict) -> Path | None:
    candidates = [cfg["dir"], *cfg.get("fallback_dirs", [])]
    for rel in candidates:
        p = ROOT / rel if not Path(rel).is_absolute() else Path(rel)
        if any(p.glob("*.pt")):
            return p
    return None


def _run_model(key: str, cfg: dict) -> dict:
    model_dir = _resolve_dir(cfg)
    if model_dir is None:
        return {"model": key, "label": cfg["label"], "rank": cfg.get("rank", True), "error": "missing checkpoint"}

    meta_path = model_dir / "train_meta.json"
    thr = 0.5
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        default = (
            0.99
            if meta.get("arch")
            in ("detect_and_policy", "detect_and_policy2", "tcn_multi", "multitask")
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
    env = [c for c in cells if c["tier"] == "C"]
    front_pass = [c for c in front if c["gate_ok"]]
    econ_pass = [c for c in econ if c["gate_ok"]]
    return {
        "model": key,
        "label": cfg["label"],
        "rank": bool(cfg.get("rank", True)),
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
        "envelope_pass_n": sum(1 for c in env if c["gate_ok"]),
        "envelope_n": len(env),
        "cells": cells,
    }


def _wrist_ablation(grip: dict, wrist: dict) -> list[dict]:
    """Per-case grip-only vs grip+wrist delta (same train)."""
    rows = []
    for name in [c[1].name for c in CASES]:
        g = next(x for x in grip["cells"] if x["name"] == name)
        w = next(x for x in wrist["cells"] if x["name"] == name)
        rows.append(
            {
                "name": name,
                "tier": g["tier"],
                "grip_ok": g["gate_ok"],
                "wrist_ok": w["gate_ok"],
                "grip_dz_cm": g["extend_dz_cm"],
                "wrist_dz_cm": w["extend_dz_cm"],
                "delta_dz_cm": float(w["extend_dz_cm"] - g["extend_dz_cm"]),
                "wrist_helps_pass": (not g["gate_ok"]) and w["gate_ok"],
                "wrist_hurts_pass": g["gate_ok"] and (not w["gate_ok"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified same-data closed-loop suite (v2)")
    parser.add_argument("--models-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--with-variance",
        action="store_true",
        default=True,
        help="Also eval second grip seed (not ranked)",
    )
    parser.add_argument("--no-variance", action="store_true")
    args = parser.parse_args()
    if args.no_variance:
        args.with_variance = False

    models = {**RANKED_MODELS}
    if args.with_variance:
        models.update(VARIANCE_MODELS)

    if args.models_json and args.models_json.exists():
        override = json.loads(args.models_json.read_text())
        alias = {
            "nn2": "nn2",
            "grip": "grip",
            "grip_only": "grip",
            "p1": "grip",
            "wrist": "wrist",
            "p2": "wrist",
            "p2_wrist": "wrist",
            "grip_s43": "grip_s43",
            "p2_grip": "grip_s43",
            "p2_grip_only": "grip_s43",
        }
        for k, d in override.items():
            mk = alias.get(k, k)
            if mk in models:
                models[mk]["dir"] = d

    results = [_run_model(k, cfg) for k, cfg in models.items()]
    ok = [m for m in results if not m.get("error")]
    ranked = [m for m in ok if m.get("rank")]
    variance = [m for m in ok if not m.get("rank")]

    def rank_key(m):
        return (
            m.get("frontier_pass_rate") or 0.0,
            m.get("mean_lift_margin_frontier_pass") or -1e9,
            -(m.get("mean_grip_economy_pass") or 1e9),
        )

    ranking = sorted(ranked, key=rank_key, reverse=True)
    print("\n=== SAME-DATA algorithm ranking (envelope excluded) ===")
    for i, m in enumerate(ranking, 1):
        print(
            f"  #{i} {m['label']:14} frontier={m['frontier_pass_n']}/{m['frontier_n']} "
            f"({100 * m['frontier_pass_rate']:.0f}%) "
            f"econ_grip={m['mean_grip_economy_pass']} "
            f"envelope={m['envelope_pass_n']}/{m['envelope_n']}"
        )

    if variance:
        print("\n=== grip-only seed variance (NOT ranked) ===")
        primary = next((m for m in ranked if m["model"] == "grip"), None)
        for m in variance:
            agree = None
            if primary:
                agree = sum(
                    1
                    for a, b in zip(primary["cells"], m["cells"])
                    if a["gate_ok"] == b["gate_ok"] and a["tier"] == "A"
                )
                agree = f"{agree}/{primary['frontier_n']} frontier agree"
            print(
                f"  {m['label']:22} frontier={m['frontier_pass_n']}/{m['frontier_n']} "
                f"({agree})"
            )

    # wrist ablation
    grip_m = next((m for m in ranked if m["model"] == "grip"), None)
    wrist_m = next((m for m in ranked if m["model"] == "wrist"), None)
    ablation = _wrist_ablation(grip_m, wrist_m) if grip_m and wrist_m else []
    if ablation:
        print("\n=== wrist ablation (grip-only → grip+wrist, same train) ===")
        helps = [r for r in ablation if r["wrist_helps_pass"]]
        hurts = [r for r in ablation if r["wrist_hurts_pass"]]
        for r in ablation:
            tag = (
                "HELP"
                if r["wrist_helps_pass"]
                else ("HURT" if r["wrist_hurts_pass"] else "same")
            )
            print(
                f"  {r['name']:24} {tag:4}  "
                f"grip={'P' if r['grip_ok'] else 'F'}{r['grip_dz_cm']:+.1f}  "
                f"wrist={'P' if r['wrist_ok'] else 'F'}{r['wrist_dz_cm']:+.1f}  "
                f"Δdz={r['delta_dz_cm']:+.1f}"
            )
        print(f"  helps_pass={len(helps)} hurts_pass={len(hurts)}")

    # matrix
    names = [c[1].name for c in CASES]
    cols = ranked + variance
    print(f"\n{'cell':24}", end="")
    for m in cols:
        print(f" {m['model']:>10}", end="")
    print()
    for name in names:
        tier = next(c[0] for c in CASES if c[1].name == name)
        print(f"{name:24}", end="")
        for m in cols:
            cell = next(x for x in m["cells"] if x["name"] == name)
            tag = "P" if cell["gate_ok"] else "F"
            print(f" {tag}{cell['extend_dz_cm']:+.0f}/{cell['antislip_max_grip']:.2f}".rjust(11), end="")
        print(f"  [{tier}]")

    hard = {
        "friction_div2": {
            c["model"]: next(x for x in c["cells"] if x["name"] == "friction_div2")
            for c in ok
        },
        "friction_s040": {
            c["model"]: next(x for x in c["cells"] if x["name"] == "friction_s040")
            for c in ok
        },
    }

    out = {
        "fairness": "same_train_data=data/slip_nn_unified",
        "ranking_rule": "algorithm classes only: nn2 / grip-only / grip+wrist; envelope not ranked",
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
                "envelope_pass_n": m["envelope_pass_n"],
                "envelope_n": m["envelope_n"],
            }
            for i, m in enumerate(ranking, 1)
        ],
        "grip_seed_variance": [
            {
                "model": m["model"],
                "label": m["label"],
                "frontier_pass_n": m["frontier_pass_n"],
                "frontier_n": m["frontier_n"],
            }
            for m in variance
        ],
        "wrist_ablation": ablation,
        "hard_cases": {
            name: {
                mk: {"gate_ok": v["gate_ok"], "extend_dz_cm": v["extend_dz_cm"]}
                for mk, v in cells.items()
            }
            for name, cells in hard.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
