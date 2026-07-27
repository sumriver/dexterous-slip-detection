#!/usr/bin/env python3
"""Eval PPO extend policy vs BC grip / grip+wrist on hard cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stable_baselines3 import PPO

from run_ketchup_robustness_sweep import CaseSpec, EXTEND_STEPS, _run_case
from sim.ketchup_extend_rl_env import EXTEND_STEPS as ENV_STEPS
from sim.ketchup_extend_rl_env import ExtendCase, KetchupExtendRLEnv

OUT = ROOT / "data" / "slip_eval" / "policy2_extend_rl_smoke.json"

EVAL_CASES = [
    ("hard", CaseSpec("friction_div2", friction_scale=0.5, sweep="hard"), 0.25),
    ("envelope", CaseSpec("friction_s040", friction_scale=0.4, sweep="envelope"), 0.25),
    ("ref", CaseSpec("friction_s045", friction_scale=0.45, sweep="ref"), 0.25),
    ("ref", CaseSpec("baseline", sweep="ref"), 0.25),
]


def _eval_ppo(model_path: Path, case_name: str, *, n_ep: int = 5, seed: int = 0) -> dict:
    model = PPO.load(model_path, device="cpu")
    env = KetchupExtendRLEnv(
        cases=(
            ExtendCase(case_name, friction_scale={"friction_div2": 0.5, "friction_s040": 0.4, "friction_s045": 0.45, "baseline": 1.0}[case_name]),
        ),
        case_probs=(1.0,),
        grip_ratchet=False,
    )
    rows = []
    for i in range(n_ep):
        obs, info = env.reset(seed=seed + i, options={"case": case_name})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
        rows.append(
            {
                "extend_dz_cm": float(info["extend_dz_cm"]),
                "extend_contact_steps": int(info["extend_contact_steps"]),
                "max_grip": float(info["max_grip"]),
                "gate_ok": bool(info.get("gate_ok", False)),
            }
        )
    pass_n = sum(1 for r in rows if r["gate_ok"])
    return {
        "backend": "ppo_env",
        "case": case_name,
        "pass_n": pass_n,
        "n": n_ep,
        "pass_rate": pass_n / n_ep,
        "mean_dz_cm": float(np.mean([r["extend_dz_cm"] for r in rows])),
        "mean_grip": float(np.mean([r["max_grip"] for r in rows])),
        "episodes": rows,
    }


def _eval_bc(model_dir: Path, policy_mode: str, spec: CaseSpec, gcap: float) -> dict:
    if not any(model_dir.glob("*.pt")):
        return {"error": f"missing {model_dir}", "case": spec.name}
    meta_path = model_dir / "train_meta.json"
    thr = 0.99
    if meta_path.exists():
        thr = float(json.loads(meta_path.read_text()).get("default_threshold", thr))
    row = _run_case(
        spec,
        antislip_nn=True,
        nn_model_dir=model_dir,
        nn_threshold=thr,
        policy_mode=policy_mode,
        antislip_grip_max=gcap,
    )
    gate_ok = row.extend_dz_cm >= 6.0 and row.extend_contact_steps >= EXTEND_STEPS
    return {
        "backend": "bc_closedloop",
        "case": spec.name,
        "policy_mode": policy_mode,
        "gate_ok": bool(gate_ok),
        "extend_dz_cm": float(row.extend_dz_cm),
        "extend_contact_steps": int(row.extend_contact_steps),
        "antislip_max_grip": float(row.antislip_max_grip),
        "status": row.status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval Policy-2 extend PPO smoke")
    parser.add_argument(
        "--ppo",
        type=Path,
        default=ROOT / "models" / "rl" / "policy2_extend" / "p2_extend_ppo_final.zip",
    )
    parser.add_argument("--grip-bc", type=Path, default=ROOT / "models" / "slip_nn_unified_grip")
    parser.add_argument("--wrist-bc", type=Path, default=ROOT / "models" / "slip_nn_unified_wrist")
    parser.add_argument("--n-ep", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    results: dict = {"ppo": {}, "bc_grip": {}, "bc_wrist": {}}
    for _, spec, gcap in EVAL_CASES:
        print(f"[PPO] {spec.name} ...", flush=True)
        if args.ppo.exists():
            results["ppo"][spec.name] = _eval_ppo(args.ppo, spec.name, n_ep=args.n_ep)
            print(
                f"  pass={results['ppo'][spec.name]['pass_n']}/{args.n_ep} "
                f"dz={results['ppo'][spec.name]['mean_dz_cm']:.1f}",
                flush=True,
            )
        print(f"[BC grip] {spec.name} ...", flush=True)
        results["bc_grip"][spec.name] = _eval_bc(args.grip_bc, "replace", spec, gcap)
        print(f"  -> {results['bc_grip'][spec.name]}", flush=True)
        print(f"[BC wrist] {spec.name} ...", flush=True)
        results["bc_wrist"][spec.name] = _eval_bc(args.wrist_bc, "p2a", spec, gcap)
        print(f"  -> {results['bc_wrist'][spec.name]}", flush=True)

    out = {
        "note": "RL smoke vs BC on hard/envelope cells; PPO runs in extend-only env",
        "env_steps": ENV_STEPS,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
