#!/usr/bin/env python3
"""Step-3: same-exam closed-loop compare — BC vs PPO(always) vs PPO(on_detect).

All rows go through ``replay_spider_task`` (full trajectory + extend), not the
Gym-only MDP.

Methods
  - bc_grip:   unified grip-only, policy_mode=replace
  - bc_wrist:  unified grip+wrist, policy_mode=p2a
  - ppo_always: PPO every extend step (detect optional; no BC policy)
  - ppo_on_detect: NN-2 detect only + PPO when soft/hard fires
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ketchup_robustness_sweep import CaseSpec, EXTEND_STEPS, _run_case
from sim.ppo_extend_hook import PPOExtendHook

OUT = ROOT / "data" / "slip_eval" / "ppo_closedloop_vs_bc.json"

CASES = [
    CaseSpec("friction_div2", friction_scale=0.5, sweep="cl_cmp"),
    CaseSpec("baseline", friction_scale=1.0, sweep="cl_cmp"),
    CaseSpec("friction_s045", friction_scale=0.45, sweep="cl_cmp"),
    CaseSpec("friction_s040", friction_scale=0.40, sweep="cl_cmp"),
]


def _pack(row, *, method: str, extra: dict | None = None) -> dict:
    gate_ok = row.extend_dz_cm >= 6.0 and row.extend_contact_steps >= EXTEND_STEPS
    out = {
        "method": method,
        "case": row.name,
        "gate_ok": bool(gate_ok),
        "status": row.status,
        "extend_dz_cm": float(row.extend_dz_cm),
        "extend_contact_steps": int(row.extend_contact_steps),
        "antislip_max_grip": float(row.antislip_max_grip),
        "nn_slip_events": int(row.nn_slip_events),
        "hook_mode": getattr(row, "_extend_hook_mode", "off"),
        "hook_queries": int(getattr(row, "_extend_hook_queries", 0)),
        "hook_overrides": int(getattr(row, "_extend_hook_overrides", 0)),
    }
    if extra:
        out.update(extra)
    return out


def _thr(model_dir: Path, default: float = 0.99) -> float:
    meta = model_dir / "train_meta.json"
    if meta.exists():
        return float(json.loads(meta.read_text()).get("default_threshold", default))
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop BC vs PPO compare")
    parser.add_argument(
        "--ppo",
        type=Path,
        default=ROOT / "models" / "rl" / "policy2_extend" / "p2_extend_ppo_final.zip",
    )
    parser.add_argument("--grip-bc", type=Path, default=ROOT / "models" / "slip_nn_unified_grip")
    parser.add_argument("--wrist-bc", type=Path, default=ROOT / "models" / "slip_nn_unified_wrist")
    parser.add_argument("--detect", type=Path, default=ROOT / "models" / "slip_nn_unified_nn2")
    parser.add_argument("--g-max", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--methods",
        default="bc_grip,bc_wrist,ppo_always,ppo_on_detect",
        help="Comma subset to run",
    )
    args = parser.parse_args()
    want = {m.strip() for m in args.methods.split(",") if m.strip()}

    results: dict[str, list[dict]] = {m: [] for m in want}

    for spec in CASES:
        print(f"\n=== {spec.name} ===", flush=True)

        if "bc_grip" in want:
            print("  [bc_grip] ...", flush=True)
            row = _run_case(
                spec,
                antislip_nn=True,
                nn_model_dir=args.grip_bc,
                nn_threshold=_thr(args.grip_bc),
                policy_mode="replace",
                antislip_grip_max=args.g_max,
            )
            results["bc_grip"].append(_pack(row, method="bc_grip"))
            print(
                f"    -> {'PASS' if results['bc_grip'][-1]['gate_ok'] else 'FAIL'} "
                f"dz={row.extend_dz_cm:+.1f} grip={row.antislip_max_grip:.3f}",
                flush=True,
            )

        if "bc_wrist" in want:
            print("  [bc_wrist] ...", flush=True)
            row = _run_case(
                spec,
                antislip_nn=True,
                nn_model_dir=args.wrist_bc,
                nn_threshold=_thr(args.wrist_bc),
                policy_mode="p2a",
                antislip_grip_max=args.g_max,
            )
            results["bc_wrist"].append(_pack(row, method="bc_wrist"))
            print(
                f"    -> {'PASS' if results['bc_wrist'][-1]['gate_ok'] else 'FAIL'} "
                f"dz={row.extend_dz_cm:+.1f} grip={row.antislip_max_grip:.3f}",
                flush=True,
            )

        if "ppo_always" in want:
            print("  [ppo_always] ...", flush=True)
            hook = PPOExtendHook.from_zip(args.ppo, g_max=args.g_max, apply_actions=True)
            row = _run_case(
                spec,
                antislip_nn=False,
                antislip_grip_max=args.g_max,
                extend_action_hook=hook,
                extend_hook_mode="always",
            )
            results["ppo_always"].append(
                _pack(row, method="ppo_always", extra={"ppo_records": len(hook.records)})
            )
            print(
                f"    -> {'PASS' if results['ppo_always'][-1]['gate_ok'] else 'FAIL'} "
                f"dz={row.extend_dz_cm:+.1f} ov={row._extend_hook_overrides}",
                flush=True,
            )

        if "ppo_on_detect" in want:
            print("  [ppo_on_detect] ...", flush=True)
            hook = PPOExtendHook.from_zip(args.ppo, g_max=args.g_max, apply_actions=True)
            # Detect-only NN-2 (policy_mode=off); PPO acts when soft/hard fires.
            row = _run_case(
                spec,
                antislip_nn=True,
                nn_model_dir=args.detect,
                nn_threshold=_thr(args.detect, 0.7),
                policy_mode="off",
                antislip_grip_max=args.g_max,
                extend_action_hook=hook,
                extend_hook_mode="on_detect",
            )
            results["ppo_on_detect"].append(
                _pack(
                    row,
                    method="ppo_on_detect",
                    extra={
                        "ppo_records": len(hook.records),
                        "detect_dir": str(args.detect),
                    },
                )
            )
            print(
                f"    -> {'PASS' if results['ppo_on_detect'][-1]['gate_ok'] else 'FAIL'} "
                f"dz={row.extend_dz_cm:+.1f} q={row._extend_hook_queries} "
                f"ov={row._extend_hook_overrides} nn_ev={row.nn_slip_events}",
                flush=True,
            )

    # Matrix print
    methods = [m for m in ("bc_grip", "bc_wrist", "ppo_always", "ppo_on_detect") if m in results]
    print(f"\n{'case':16}", end="")
    for m in methods:
        print(f" {m:>14}", end="")
    print()
    for spec in CASES:
        print(f"{spec.name:16}", end="")
        for m in methods:
            cell = next(x for x in results[m] if x["case"] == spec.name)
            tag = "P" if cell["gate_ok"] else "F"
            print(f" {tag}{cell['extend_dz_cm']:+.0f}/{cell['antislip_max_grip']:.2f}".rjust(15), end="")
        print()

    summary = {
        "fairness": "all via replay_spider_task full traj+extend",
        "gate": "extend_dz>=6cm and contacts>=200",
        "methods": {
            "bc_grip": "unified grip, replace",
            "bc_wrist": "unified wrist, p2a",
            "ppo_always": "PPO every extend step, no BC policy",
            "ppo_on_detect": "NN-2 detect-only + PPO on soft/hard",
        },
        "results": results,
        "pass_counts": {
            m: sum(1 for r in rows if r["gate_ok"]) for m, rows in results.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\npass_counts={summary['pass_counts']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
