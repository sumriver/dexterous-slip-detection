#!/usr/bin/env python3
"""Discriminative closed-loop suite for ranking slip policies.

Uses a curated frontier grid (mass × μ × grip_cap) instead of easy all-PASS
or hard all-FAIL gates. Emits per-cell metrics + model rankings.

  python3 scripts/eval_slip_discriminative_suite.py
  python3 scripts/eval_slip_discriminative_suite.py --models nn2,p1,p2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ketchup_robustness_sweep import CaseSpec, EXTEND_STEPS, _run_case  # noqa: E402

GRID_PATH = ROOT / "data" / "slip_eval" / "discriminative_case_grid.json"
OUT_DEFAULT = ROOT / "data" / "slip_eval" / "discriminative_suite_latest.json"

MODEL_PRESETS = {
    "nn2": {
        "dir": ROOT / "models" / "slip_nn_v2",
        "policy_mode": "off",
        "label": "NN-2",
    },
    "p1": {
        "dir": ROOT / "models" / "slip_nn_policy1",
        "policy_mode": "replace",
        "label": "Policy-1",
    },
    "p2": {
        "dir": ROOT / "models" / "slip_nn_policy2",
        "policy_mode": "p2a",
        "label": "Policy-2",
    },
    "p2h": {
        "dir": ROOT / "models" / "slip_nn_policy2_heavy",
        "policy_mode": "p2a",
        "label": "Policy-2-heavy",
    },
}


def _load_grid(path: Path) -> dict:
    return json.loads(path.read_text())


def _iter_tier_cases(grid: dict, tiers: list[str]):
    for tier in tiers:
        block = grid["tiers"][tier]
        for c in block["cases"]:
            yield tier, c


def _metrics(row, *, grip_cap: float, dz_min: float) -> dict:
    dz = float(row.extend_dz_cm)
    grip = float(row.antislip_max_grip)
    events = int(row.nn_slip_events)
    contacts = int(row.extend_contact_steps)
    gate_ok = bool(dz >= dz_min and contacts >= EXTEND_STEPS)
    lift_margin = dz - dz_min
    composite = None
    if gate_ok:
        composite = lift_margin - 20.0 * grip - 0.05 * events
    return {
        "name": row.name,
        "status": row.status,
        "gate_ok": gate_ok,
        "extend_dz_cm": dz,
        "lift_margin_cm": lift_margin,
        "extend_contact_steps": contacts,
        "contact_ratio": contacts / float(EXTEND_STEPS),
        "antislip_max_grip": grip,
        "grip_cap": grip_cap,
        "grip_headroom": grip_cap - grip,
        "nn_slip_events": events,
        "fail_reason": row.fail_reason,
        "composite_score": composite,
    }


def _run_model(
    key: str,
    *,
    grid: dict,
    tiers: list[str],
    dz_min: float,
) -> dict:
    preset = MODEL_PRESETS[key]
    model_dir = Path(preset["dir"])
    if not any(model_dir.glob("*.pt")):
        return {"model": key, "label": preset["label"], "error": f"missing ckpt in {model_dir}"}

    thr = None
    meta_path = model_dir / "train_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        default = (
            0.99
            if meta.get("arch") in ("detect_and_policy", "detect_and_policy2")
            else 0.5
        )
        thr = float(meta.get("default_threshold", default))

    by_tier: dict[str, list] = {t: [] for t in tiers}
    cells = []
    for tier, c in _iter_tier_cases(grid, tiers):
        grip_cap = float(c["grip_cap"])
        spec = CaseSpec(
            c["name"],
            mass_scale=float(c["mass_scale"]),
            friction_scale=float(c["friction_scale"]),
            sweep=tier,
        )
        print(
            f"[{preset['label']}] {tier} {c['name']} "
            f"m×{c['mass_scale']} μ×{c['friction_scale']} g≤{grip_cap} ...",
            flush=True,
        )
        row = _run_case(
            spec,
            save_video=False,
            antislip_nn=True,
            nn_model_dir=model_dir,
            nn_threshold=thr,
            policy_mode=preset["policy_mode"],
            antislip_grip_max=grip_cap,
        )
        # restore logical name (CaseSpec name is used)
        m = _metrics(row, grip_cap=grip_cap, dz_min=dz_min)
        m["tier"] = tier
        m["mass_scale"] = float(c["mass_scale"])
        m["friction_scale"] = float(c["friction_scale"])
        m["why"] = c.get("why", "")
        by_tier[tier].append(m)
        cells.append(m)
        mark = "PASS" if m["gate_ok"] else m["status"]
        print(
            f"  -> {mark} dz={m['extend_dz_cm']:+.1f}cm "
            f"margin={m['lift_margin_cm']:+.1f} grip={m['antislip_max_grip']:.3f} "
            f"ev={m['nn_slip_events']}",
            flush=True,
        )

    # Aggregates
    front = by_tier.get("A_frontier", [])
    econ = by_tier.get("B_economy", [])
    env = by_tier.get("C_envelope", [])
    front_pass = [x for x in front if x["gate_ok"]]
    econ_pass = [x for x in econ if x["gate_ok"]]
    env_pass = [x for x in env if x["gate_ok"]]

    def _mean(xs, key):
        vals = [x[key] for x in xs if x.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    summary = {
        "model": key,
        "label": preset["label"],
        "nn_model_dir": str(model_dir),
        "policy_mode": preset["policy_mode"],
        "frontier_n": len(front),
        "frontier_pass_n": len(front_pass),
        "frontier_pass_rate": (len(front_pass) / len(front)) if front else None,
        "mean_lift_margin_cm_frontier_pass": _mean(front_pass, "lift_margin_cm"),
        "mean_grip_peak_economy_pass": _mean(econ_pass, "antislip_max_grip"),
        "mean_lift_margin_cm_economy_pass": _mean(econ_pass, "lift_margin_cm"),
        "mean_composite_frontier_pass": _mean(front_pass, "composite_score"),
        "envelope_unexpected_pass_n": len(env_pass),
        "baseline_slip_events": next(
            (x["nn_slip_events"] for x in econ if x["name"] == "baseline"), None
        ),
        "by_tier": by_tier,
        "cells": cells,
    }
    return summary


def _rank(models: list[dict]) -> list[dict]:
    """Rank by frontier_pass_rate, then lift margin, then lower grip, then fewer events."""
    def key(m):
        if m.get("error"):
            return (-1.0, -1e9, -1e9, -1e9)
        rate = m.get("frontier_pass_rate") or 0.0
        margin = m.get("mean_lift_margin_cm_frontier_pass")
        if margin is None:
            margin = -1e9
        grip = m.get("mean_grip_peak_economy_pass")
        neg_grip = -(grip if grip is not None else 1e9)
        ev = m.get("baseline_slip_events")
        neg_ev = -(ev if ev is not None else 1e9)
        # Penalize unexpected envelope passes lightly (shouldn't happen)
        env_pen = -10.0 * (m.get("envelope_unexpected_pass_n") or 0)
        return (rate + env_pen, margin, neg_grip, neg_ev)

    ordered = sorted([m for m in models if not m.get("error")], key=key, reverse=True)
    ranking = []
    for i, m in enumerate(ordered, 1):
        ranking.append(
            {
                "rank": i,
                "model": m["model"],
                "label": m["label"],
                "frontier_pass_rate": m.get("frontier_pass_rate"),
                "frontier_pass_n": m.get("frontier_pass_n"),
                "frontier_n": m.get("frontier_n"),
                "mean_lift_margin_cm_frontier_pass": m.get(
                    "mean_lift_margin_cm_frontier_pass"
                ),
                "mean_grip_peak_economy_pass": m.get("mean_grip_peak_economy_pass"),
                "baseline_slip_events": m.get("baseline_slip_events"),
                "envelope_unexpected_pass_n": m.get("envelope_unexpected_pass_n"),
            }
        )
    return ranking


def _print_matrix(models: list[dict], grid: dict) -> None:
    # Collect all A+B cell names in order
    names = []
    for tier in ("A_frontier", "B_economy", "C_envelope"):
        for c in grid["tiers"][tier]["cases"]:
            names.append((tier, c["name"], float(c["grip_cap"])))

    header = f"{'cell':28} {'g':>4}"
    cols = [m for m in models if not m.get("error")]
    for m in cols:
        header += f" {m['model']:>12}"
    print("\n=== Matrix (P=pass gate, F=fail; dz/grip) ===")
    print(header)
    for tier, name, gcap in names:
        line = f"{name:28} {gcap:4.2f}"
        for m in cols:
            cell = next((x for x in m["cells"] if x["name"] == name), None)
            if cell is None:
                line += f" {'—':>12}"
                continue
            tag = "P" if cell["gate_ok"] else "F"
            line += f" {tag}{cell['extend_dz_cm']:+.0f}/{cell['antislip_max_grip']:.2f}".rjust(13)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discriminative slip-policy suite")
    parser.add_argument("--grid", type=Path, default=GRID_PATH)
    parser.add_argument(
        "--models",
        default="nn2,p1,p2,p2h",
        help="Comma list from: nn2,p1,p2,p2h",
    )
    parser.add_argument(
        "--tiers",
        default="A_frontier,B_economy,C_envelope",
        help="Comma list of tiers to run",
    )
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    grid = _load_grid(args.grid)
    dz_min = float(grid["gate"]["extend_dz_cm_min"])
    model_keys = [x.strip() for x in args.models.split(",") if x.strip()]
    tiers = [x.strip() for x in args.tiers.split(",") if x.strip()]
    for k in model_keys:
        if k not in MODEL_PRESETS:
            print(f"Unknown model {k}; choose from {list(MODEL_PRESETS)}", file=sys.stderr)
            sys.exit(2)

    results = []
    for k in model_keys:
        results.append(_run_model(k, grid=grid, tiers=tiers, dz_min=dz_min))

    ranking = _rank(results)
    _print_matrix(results, grid)

    print("\n=== Ranking (frontier_pass_rate → lift margin → low grip) ===")
    for r in ranking:
        print(
            f"  #{r['rank']} {r['label']:16} "
            f"frontier={r['frontier_pass_n']}/{r['frontier_n']} "
            f"({(r['frontier_pass_rate'] or 0)*100:.0f}%) "
            f"margin={r['mean_lift_margin_cm_frontier_pass']} "
            f"econ_grip={r['mean_grip_peak_economy_pass']} "
            f"base_ev={r['baseline_slip_events']}"
        )

    out = {
        "version": grid.get("version", "v1"),
        "grid": str(args.grid),
        "extend_steps": EXTEND_STEPS,
        "gate": grid["gate"],
        "ranking_rule": grid.get("ranking"),
        "models": results,
        "ranking": ranking,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    # Also write slim rankings sidecar
    rank_path = args.out.parent / "discriminative_rankings.json"
    rank_path.write_text(json.dumps({"ranking": ranking, "source": str(args.out)}, indent=2))
    print(f"\nWrote {args.out}")
    print(f"Wrote {rank_path}")


if __name__ == "__main__":
    main()
