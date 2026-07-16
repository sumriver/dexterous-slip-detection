#!/usr/bin/env python3
"""NN-1 closed-loop smoke table: baseline + friction_div2 with --antislip-nn.

Requires a trained checkpoint in models/slip_nn/. Does not train.
Writes data/slip_nn/closedloop_smoke.json.
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

OUT = ROOT / "data" / "slip_nn" / "closedloop_smoke.json"
CASES = [
    CaseSpec("baseline", sweep="baseline"),
    CaseSpec("friction_div2", friction_scale=0.5, sweep="friction"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="NN-1 closed-loop smoke (baseline + friction÷2)")
    parser.add_argument("--nn-model-dir", type=Path, default=ROOT / "models" / "slip_nn")
    parser.add_argument("--nn-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if not any(args.nn_model_dir.glob("*.pt")):
        print(
            f"Missing checkpoint in {args.nn_model_dir}\n"
            "Train first, then re-run this smoke:\n"
            "  python3 scripts/train_slip_tcn.py --label y_fused\n"
            "  python3 scripts/eval_slip_nn_closedloop.py",
            file=sys.stderr,
        )
        sys.exit(2)

    results = []
    for spec in CASES:
        print(f"NN closed-loop: {spec.name} ...")
        results.append(
            _run_case(
                spec,
                save_video=False,
                antislip_nn=True,
                nn_model_dir=args.nn_model_dir,
                nn_threshold=args.nn_threshold,
            )
        )

    _print_table(results)
    summary = {
        "mode": "antislip_nn",
        "nn_model_dir": str(args.nn_model_dir),
        "nn_threshold": args.nn_threshold,
        "extend_steps": EXTEND_STEPS,
        "gates": {
            "friction_div2_extend_dz_cm_min": 6.0,
            "friction_div2_extend_contacts_min": 200,
            "baseline_false_trigger_steps_max": 100,
        },
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
            for r in results
        ],
    }
    for r in results:
        if r.name == "friction_div2":
            summary["friction_div2_gate_ok"] = (
                r.extend_dz_cm >= 6.0 and r.extend_contact_steps >= 200
            )
        if r.name == "baseline":
            summary["baseline_false_trigger_ok"] = r.nn_slip_events < 100

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
