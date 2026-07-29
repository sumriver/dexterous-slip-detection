#!/usr/bin/env python3
"""Discriminative closed-loop suite — rank ONLY same-train-domain models.

Fairness rule: never put differently-trained checkpoints in one ranking.
Cross-domain numbers may be printed as transfer diagnostics only.

  # Friction P2-A domain (grip+wrist vs grip-only on same windows)
  python3 scripts/eval_slip_discriminative_suite.py --domain friction_p2a

  # Heavy+gripcap domain
  python3 scripts/eval_slip_discriminative_suite.py --domain heavy_gripcap
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

GRID_PATH = ROOT / "data" / "slip_eval" / "discriminative_case_grid.json"
OUT_DEFAULT = ROOT / "data" / "slip_eval" / "discriminative_suite_latest.json"

MODEL_PRESETS = {
    "p2": {
        "dir": ROOT / "models" / "slip_nn_policy2",
        "policy_mode": "p2a",
        "label": "P2 grip+wrist",
        "train_domain": "friction_p2a",
        "train_data": "data/slip_nn_policy2",
    },
    "p2_grip_only": {
        "dir": ROOT / "models" / "slip_nn_policy2_grip_only",
        "policy_mode": "replace",
        "label": "P2 grip-only (same data)",
        "train_domain": "friction_p2a",
        "train_data": "data/slip_nn_policy2",
    },
    "p2h": {
        "dir": ROOT / "models" / "slip_nn_policy2_heavy",
        "policy_mode": "p2a",
        "label": "P2H grip+wrist",
        "train_domain": "heavy_gripcap",
        "train_data": "data/slip_nn_policy2_heavy",
    },
    "p2h_grip_only": {
        "dir": ROOT / "models" / "slip_nn_policy2_heavy_grip_only",
        "policy_mode": "replace",
        "label": "P2H grip-only (same data)",
        "train_domain": "heavy_gripcap",
        "train_data": "data/slip_nn_policy2_heavy",
    },
    # Reference-only: different train sets — never mixed into fair ranking.
    "nn2": {
        "dir": ROOT / "models" / "slip_nn_v2",
        "policy_mode": "off",
        "label": "NN-2 (ref, different train)",
        "train_domain": "reference_only",
        "train_data": "data/slip_nn (multitask)",
    },
    "p1": {
        "dir": ROOT / "models" / "slip_nn_policy1",
        "policy_mode": "replace",
        "label": "Policy-1 (ref, different train)",
        "train_domain": "reference_only",
        "train_data": "data/slip_nn_policy",
    },
}


def _load_grid(path: Path) -> dict:
    return json.loads(path.read_text())


def _metrics(row, *, grip_cap: float, dz_min: float) -> dict:
    dz = float(row.extend_dz_cm)
    grip = float(row.antislip_max_grip)
    events = int(row.nn_slip_events)
    contacts = int(row.extend_contact_steps)
    gate_ok = bool(dz >= dz_min and contacts >= EXTEND_STEPS)
    lift_margin = dz - dz_min
    composite = (lift_margin - 20.0 * grip - 0.05 * events) if gate_ok else None
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
        "nn_slip_events": events,
        "fail_reason": row.fail_reason,
        "composite_score": composite,
    }


def _run_model(key: str, *, tiers: list[str], grid: dict, dz_min: float) -> dict:
    preset = MODEL_PRESETS[key]
    model_dir = Path(preset["dir"])
    if not any(model_dir.glob("*.pt")):
        return {
            "model": key,
            "label": preset["label"],
            "train_domain": preset["train_domain"],
            "train_data": preset["train_data"],
            "error": f"missing ckpt in {model_dir}",
        }

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
    for tier in tiers:
        for c in grid["tiers"][tier]["cases"]:
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
            m = _metrics(row, grip_cap=grip_cap, dz_min=dz_min)
            m["tier"] = tier
            m["mass_scale"] = float(c["mass_scale"])
            m["friction_scale"] = float(c["friction_scale"])
            by_tier[tier].append(m)
            cells.append(m)
            mark = "PASS" if m["gate_ok"] else m["status"]
            print(
                f"  -> {mark} dz={m['extend_dz_cm']:+.1f}cm "
                f"grip={m['antislip_max_grip']:.3f} ev={m['nn_slip_events']}",
                flush=True,
            )

    front_tiers = [t for t in tiers if t.startswith("A_")]
    econ_tiers = [t for t in tiers if t.startswith("B_")]
    env_tiers = [t for t in tiers if t.startswith("C_")]
    front = [x for t in front_tiers for x in by_tier.get(t, [])]
    econ = [x for t in econ_tiers for x in by_tier.get(t, [])]
    env = [x for t in env_tiers for x in by_tier.get(t, [])]
    front_pass = [x for x in front if x["gate_ok"]]
    econ_pass = [x for x in econ if x["gate_ok"]]

    def _mean(xs, key):
        vals = [x[key] for x in xs if x.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    return {
        "model": key,
        "label": preset["label"],
        "train_domain": preset["train_domain"],
        "train_data": preset["train_data"],
        "nn_model_dir": str(model_dir),
        "policy_mode": preset["policy_mode"],
        "frontier_n": len(front),
        "frontier_pass_n": len(front_pass),
        "frontier_pass_rate": (len(front_pass) / len(front)) if front else None,
        "mean_lift_margin_cm_frontier_pass": _mean(front_pass, "lift_margin_cm"),
        "mean_grip_peak_economy_pass": _mean(econ_pass, "antislip_max_grip"),
        "mean_lift_margin_cm_economy_pass": _mean(econ_pass, "lift_margin_cm"),
        "envelope_unexpected_pass_n": len([x for x in env if x["gate_ok"]]),
        "baseline_slip_events": next(
            (x["nn_slip_events"] for x in econ if "baseline" in x["name"]), None
        ),
        "by_tier": by_tier,
        "cells": cells,
    }


def _rank(models: list[dict]) -> list[dict]:
    def key(m):
        rate = m.get("frontier_pass_rate") or 0.0
        margin = m.get("mean_lift_margin_cm_frontier_pass")
        if margin is None:
            margin = -1e9
        grip = m.get("mean_grip_peak_economy_pass")
        neg_grip = -(grip if grip is not None else 1e9)
        ev = m.get("baseline_slip_events")
        neg_ev = -(ev if ev is not None else 1e9)
        return (rate, margin, neg_grip, neg_ev)

    ordered = sorted([m for m in models if not m.get("error")], key=key, reverse=True)
    out = []
    for i, m in enumerate(ordered, 1):
        out.append(
            {
                "rank": i,
                "model": m["model"],
                "label": m["label"],
                "train_domain": m["train_domain"],
                "train_data": m["train_data"],
                "frontier_pass_rate": m.get("frontier_pass_rate"),
                "frontier_pass_n": m.get("frontier_pass_n"),
                "frontier_n": m.get("frontier_n"),
                "mean_lift_margin_cm_frontier_pass": m.get(
                    "mean_lift_margin_cm_frontier_pass"
                ),
                "mean_grip_peak_economy_pass": m.get("mean_grip_peak_economy_pass"),
                "baseline_slip_events": m.get("baseline_slip_events"),
            }
        )
    return out


def _print_matrix(models: list[dict], tiers: list[str], grid: dict) -> None:
    names = []
    for tier in tiers:
        for c in grid["tiers"][tier]["cases"]:
            names.append((tier, c["name"], float(c["grip_cap"])))
    cols = [m for m in models if not m.get("error")]
    header = f"{'cell':28} {'g':>4}"
    for m in cols:
        header += f" {m['model']:>14}"
    print("\n=== Matrix (same-domain) ===")
    print(header)
    for tier, name, gcap in names:
        line = f"{name:28} {gcap:4.2f}"
        for m in cols:
            cell = next((x for x in m["cells"] if x["name"] == name), None)
            if cell is None:
                line += f" {'—':>14}"
                continue
            tag = "P" if cell["gate_ok"] else "F"
            line += f" {tag}{cell['extend_dz_cm']:+.0f}/{cell['antislip_max_grip']:.2f}".rjust(15)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Same-train-domain discriminative suite")
    parser.add_argument("--grid", type=Path, default=GRID_PATH)
    parser.add_argument(
        "--domain",
        choices=("friction_p2a", "heavy_gripcap", "all_fair"),
        default="all_fair",
        help="Train domain to rank within (default: run both fair leagues)",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Also run NN-2/P1 as unlabeled transfer diagnostics (NOT ranked)",
    )
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    grid = _load_grid(args.grid)
    dz_min = float(grid["gate"]["extend_dz_cm_min"])
    domains = (
        ["friction_p2a", "heavy_gripcap"]
        if args.domain == "all_fair"
        else [args.domain]
    )

    league_reports = []
    for domain in domains:
        dmeta = grid["train_domains"][domain]
        fair_keys = list(dmeta["fair_models"])
        tiers = list(dmeta["eval_tiers"])
        print(f"\n######## FAIR LEAGUE: {domain}  data={dmeta['data']} ########")
        print(f"models={fair_keys}  (same train set required)")
        missing = [k for k in fair_keys if not any(Path(MODEL_PRESETS[k]["dir"]).glob("*.pt"))]
        if missing:
            print(
                f"ERROR: missing same-domain checkpoints {missing}. Train them first:\n"
                f"  # friction_p2a:\n"
                f"  python3 scripts/train_slip_policy.py --data data/slip_nn_policy2 "
                f"--max-grip 0.25 --out models/slip_nn_policy2_grip_only\n"
                f"  # heavy_gripcap:\n"
                f"  python3 scripts/train_slip_policy.py --data data/slip_nn_policy2_heavy "
                f"--max-grip 0.15 --out models/slip_nn_policy2_heavy_grip_only",
                file=sys.stderr,
            )
            sys.exit(3)

        results = [_run_model(k, tiers=tiers, grid=grid, dz_min=dz_min) for k in fair_keys]
        ranking = _rank(results)
        _print_matrix(results, tiers, grid)
        print(f"\n=== Ranking within `{domain}` (ONLY same train data) ===")
        for r in ranking:
            print(
                f"  #{r['rank']} {r['label']:28} "
                f"frontier={r['frontier_pass_n']}/{r['frontier_n']} "
                f"({(r['frontier_pass_rate'] or 0)*100:.0f}%) "
                f"econ_grip={r['mean_grip_peak_economy_pass']} "
                f"data={r['train_data']}"
            )
        league_reports.append(
            {
                "domain": domain,
                "train_data": dmeta["data"],
                "fairness": "same_train_data_only",
                "models": results,
                "ranking": ranking,
            }
        )

        if args.include_reference:
            print("\n--- reference (DIFFERENT train; not ranked) ---")
            for k in ("nn2", "p1"):
                ref = _run_model(k, tiers=tiers, grid=grid, dz_min=dz_min)
                if ref.get("error"):
                    print(f"  {k}: {ref['error']}")
                else:
                    print(
                        f"  {ref['label']}: frontier "
                        f"{ref['frontier_pass_n']}/{ref['frontier_n']} "
                        f"(transfer only; train={ref['train_data']})"
                    )
                league_reports[-1].setdefault("reference_transfer", []).append(ref)

    out = {
        "version": grid.get("version", "v2"),
        "fairness_rule": grid.get("fairness_rule"),
        "grid": str(args.grid),
        "leagues": league_reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    rank_path = args.out.parent / "discriminative_rankings.json"
    rank_path.write_text(
        json.dumps(
            {
                "fairness_rule": out["fairness_rule"],
                "leagues": [
                    {"domain": L["domain"], "train_data": L["train_data"], "ranking": L["ranking"]}
                    for L in league_reports
                ],
            },
            indent=2,
        )
    )
    print(f"\nWrote {args.out}")
    print(f"Wrote {rank_path}")


if __name__ == "__main__":
    main()
