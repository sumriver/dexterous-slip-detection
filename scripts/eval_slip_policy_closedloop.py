#!/usr/bin/env python3
"""NN-Policy-1 closed-loop eval vs NN-2: baseline + friction÷2.

Compares ``models/slip_nn_policy1`` (replace) against ``models/slip_nn_v2``.
Writes ``data/slip_nn_policy/closedloop_policy1.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ketchup_robustness_sweep import (  # noqa: E402
    CaseSpec,
    EXTEND_STEPS,
    _print_table,
    _run_case,
)

OUT = ROOT / "data" / "slip_nn_policy" / "closedloop_policy1.json"
CASES = [
    CaseSpec("baseline", sweep="baseline"),
    CaseSpec("friction_div2", friction_scale=0.5, sweep="friction"),
]


def _run_suite(model_dir: Path, *, threshold: float | None, policy_mode: str | None) -> list:
    thr = threshold
    if thr is None:
        meta_path = model_dir / "train_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            default = 0.99 if meta.get("arch") == "detect_and_policy" else 0.5
            thr = float(meta.get("default_threshold", default))
        else:
            thr = 0.5
    rows = []
    for spec in CASES:
        print(f"[{model_dir.name} mode={policy_mode or 'auto'}] {spec.name} ...")
        rows.append(
            _run_case(
                spec,
                save_video=False,
                antislip_nn=True,
                nn_model_dir=model_dir,
                nn_threshold=thr,
                policy_mode=policy_mode,
            )
        )
    return rows


def _pack(model_dir: Path, policy_mode: str | None, rows) -> dict:
    pack = {
        "nn_model_dir": str(model_dir),
        "policy_mode": policy_mode or "auto",
        "cases": [
            {
                "name": r.name,
                "status": r.status,
                "extend_dz_cm": r.extend_dz_cm,
                "extend_contact_steps": r.extend_contact_steps,
                "nn_slip_events": r.nn_slip_events,
                "antislip_max_grip": r.antislip_max_grip,
                "fail_reason": r.fail_reason,
            }
            for r in rows
        ],
    }
    for r in rows:
        if r.name == "friction_div2":
            pack["friction_div2_gate_ok"] = (
                r.extend_dz_cm >= 6.0 and r.extend_contact_steps >= 200
            )
            pack["friction_div2_extend_dz_cm"] = r.extend_dz_cm
            pack["friction_div2_max_grip"] = r.antislip_max_grip
        if r.name == "baseline":
            pack["baseline_false_trigger_ok"] = r.nn_slip_events < 100
            pack["baseline_nn_slip_events"] = r.nn_slip_events
            pack["baseline_max_grip"] = r.antislip_max_grip
    return pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy-1 vs NN-2 closed-loop eval")
    parser.add_argument("--policy-dir", type=Path, default=ROOT / "models" / "slip_nn_policy1")
    parser.add_argument("--nn2-dir", type=Path, default=ROOT / "models" / "slip_nn_v2")
    parser.add_argument("--policy-mode", default="replace", choices=("off", "replace", "residual"))
    parser.add_argument("--skip-nn2", action="store_true", help="Only run policy suite")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if not any(args.policy_dir.glob("*.pt")):
        print(f"Missing policy checkpoint in {args.policy_dir}", file=sys.stderr)
        sys.exit(2)

    summary: dict = {
        "mode": "policy_vs_nn2",
        "extend_steps": EXTEND_STEPS,
        "gates": {
            "friction_div2_extend_dz_cm_min": 6.0,
            "friction_div2_extend_contacts_min": 200,
            "baseline_nn_slip_events_max": 43,
            "baseline_false_trigger_steps_max": 100,
            "note": "Policy-1: ÷2 not worse than NN-2; baseline grip peak better or events ≤43",
        },
    }

    if not args.skip_nn2:
        nn2_rows = _run_suite(args.nn2_dir, threshold=None, policy_mode="off")
        print("\n=== NN-2 (policy=off) ===")
        _print_table(nn2_rows)
        summary["nn2"] = _pack(args.nn2_dir, "off", nn2_rows)

    pol_rows = _run_suite(args.policy_dir, threshold=None, policy_mode=args.policy_mode)
    print(f"\n=== Policy-1 (mode={args.policy_mode}) ===")
    _print_table(pol_rows)
    summary["policy1"] = _pack(args.policy_dir, args.policy_mode, pol_rows)

    if "nn2" in summary:
        n2 = summary["nn2"]
        p1 = summary["policy1"]
        summary["compare"] = {
            "baseline_nn_slip_delta": p1.get("baseline_nn_slip_events", 0)
            - n2.get("baseline_nn_slip_events", 0),
            "baseline_max_grip_delta": p1.get("baseline_max_grip", 0)
            - n2.get("baseline_max_grip", 0),
            "friction_div2_dz_delta_cm": p1.get("friction_div2_extend_dz_cm", 0)
            - n2.get("friction_div2_extend_dz_cm", 0),
            "friction_div2_max_grip_delta": p1.get("friction_div2_max_grip", 0)
            - n2.get("friction_div2_max_grip", 0),
            "policy_div2_gate_ok": bool(p1.get("friction_div2_gate_ok")),
            "policy_baseline_events_ok": p1.get("baseline_nn_slip_events", 999) <= 43,
            "policy_grip_better_or_equal": p1.get("baseline_max_grip", 1)
            <= n2.get("baseline_max_grip", 0) + 1e-9,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")
    if "compare" in summary:
        c = summary["compare"]
        print(
            "compare: "
            f"baseline_events Δ={c['baseline_nn_slip_delta']:+d} "
            f"baseline_grip Δ={c['baseline_max_grip_delta']:+.3f} "
            f"÷2_dz Δ={c['friction_div2_dz_delta_cm']:+.1f}cm "
            f"÷2_gate={c['policy_div2_gate_ok']} "
            f"events≤43={c['policy_baseline_events_ok']} "
            f"grip≤nn2={c['policy_grip_better_or_equal']}"
        )


if __name__ == "__main__":
    main()
