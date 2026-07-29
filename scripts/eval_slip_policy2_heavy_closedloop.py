#!/usr/bin/env python3
"""Closed-loop eval for Policy-2 trained on heavy + grip-cap data.

Cases: baseline, mass×2/×4, mass×2/×4 + friction÷2.
Sim grip ceiling matches training: antislip_grip_max=0.15.
Writes ``data/slip_nn_policy2_heavy/closedloop_policy2_heavy.json``.
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

OUT = ROOT / "data" / "slip_nn_policy2_heavy" / "closedloop_policy2_heavy.json"
CASES = [
    CaseSpec("baseline", sweep="baseline"),
    CaseSpec("mass_x2", mass_scale=2.0, sweep="mass"),
    CaseSpec("mass_x4", mass_scale=4.0, sweep="mass"),
    CaseSpec("mass_x2_friction_div2", mass_scale=2.0, friction_scale=0.5, sweep="mass"),
    CaseSpec("mass_x4_friction_div2", mass_scale=4.0, friction_scale=0.5, sweep="mass"),
]


def _run_suite(
    model_dir: Path,
    *,
    threshold: float | None,
    policy_mode: str | None,
    grip_max: float,
) -> list:
    thr = threshold
    if thr is None:
        meta_path = model_dir / "train_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            default = (
                0.99
                if meta.get("arch") in ("detect_and_policy", "detect_and_policy2")
                else 0.5
            )
            thr = float(meta.get("default_threshold", default))
        else:
            thr = 0.5
    rows = []
    for spec in CASES:
        print(f"[{model_dir.name} mode={policy_mode or 'auto'} g≤{grip_max}] {spec.name} ...")
        rows.append(
            _run_case(
                spec,
                save_video=False,
                antislip_nn=True,
                nn_model_dir=model_dir,
                nn_threshold=thr,
                policy_mode=policy_mode,
                antislip_grip_max=grip_max,
            )
        )
    return rows


def _pack(model_dir: Path, policy_mode: str | None, grip_max: float, rows) -> dict:
    pack: dict = {
        "nn_model_dir": str(model_dir),
        "policy_mode": policy_mode or "auto",
        "antislip_grip_max": grip_max,
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
        pack[f"{r.name}_extend_dz_cm"] = r.extend_dz_cm
        pack[f"{r.name}_max_grip"] = r.antislip_max_grip
        pack[f"{r.name}_nn_slip_events"] = r.nn_slip_events
        pack[f"{r.name}_gate_ok"] = r.extend_dz_cm >= 6.0 and r.extend_contact_steps >= 200
    return pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Heavy+gripcap Policy-2 closed-loop eval")
    parser.add_argument(
        "--policy2-dir", type=Path, default=ROOT / "models" / "slip_nn_policy2_heavy"
    )
    parser.add_argument("--policy1-dir", type=Path, default=ROOT / "models" / "slip_nn_policy1")
    parser.add_argument("--nn2-dir", type=Path, default=ROOT / "models" / "slip_nn_v2")
    parser.add_argument("--grip-max", type=float, default=0.15)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if not any(args.policy2_dir.glob("*.pt")):
        print(f"Missing checkpoint in {args.policy2_dir}", file=sys.stderr)
        sys.exit(2)

    summary: dict = {
        "mode": "policy2_heavy_gripcap",
        "extend_steps": EXTEND_STEPS,
        "grip_max": args.grip_max,
        "gates": {
            "extend_dz_cm_min": 6.0,
            "extend_contacts_min": 200,
            "note": "Train distribution: mass×2/×4 × {μ=1, ÷2}, g_max=0.15",
        },
    }

    if not args.skip_baselines:
        if any(args.nn2_dir.glob("*.pt")):
            nn2_rows = _run_suite(
                args.nn2_dir, threshold=None, policy_mode="off", grip_max=args.grip_max
            )
            print("\n=== NN-2 (grip-capped) ===")
            _print_table(nn2_rows)
            summary["nn2"] = _pack(args.nn2_dir, "off", args.grip_max, nn2_rows)
        if any(args.policy1_dir.glob("*.pt")):
            p1_rows = _run_suite(
                args.policy1_dir,
                threshold=None,
                policy_mode="replace",
                grip_max=args.grip_max,
            )
            print("\n=== Policy-1 (grip-capped) ===")
            _print_table(p1_rows)
            summary["policy1"] = _pack(args.policy1_dir, "replace", args.grip_max, p1_rows)

    p2_rows = _run_suite(
        args.policy2_dir, threshold=None, policy_mode="p2a", grip_max=args.grip_max
    )
    print("\n=== Policy-2 heavy (p2a, grip-capped) ===")
    _print_table(p2_rows)
    summary["policy2_heavy"] = _pack(args.policy2_dir, "p2a", args.grip_max, p2_rows)

    p2 = summary["policy2_heavy"]
    summary["compare"] = {
        name: bool(p2.get(f"{name}_gate_ok"))
        for name in [c.name for c in CASES]
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")
    print("gates:", summary["compare"])


if __name__ == "__main__":
    main()
