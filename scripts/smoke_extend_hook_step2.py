#!/usr/bin/env python3
"""Step-2 smoke: PPOExtendHook behind gated modes (no full BC compare yet).

Checks:
  A) always + apply_actions=False → overrides=0 (propose-only safe)
  B) always + apply_actions=True  → overrides=200, lift should improve on div2
  C) on_detect without NN        → queries=0 (needs detect; documented)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.ppo_extend_hook import PPOExtendHook, PPO_OBS_DIM, build_ppo_obs_from_info
from sim.extend_action_hook import ExtendStepInfo
from sim.spider_ketchup import DEFAULT_WORKSPACE
from sim.spider_replay import SpiderTaskConfig, replay_spider_task

SPIDER = ROOT / "third_party" / "spider"
OUT = ROOT / "data" / "rl" / "hook_step2"
DEFAULT_PPO = ROOT / "models" / "rl" / "policy2_extend" / "p2_extend_ppo_final.zip"


def _cfg() -> SpiderTaskConfig:
    return SpiderTaskConfig(
        dataset_dir=SPIDER / "example_datasets",
        dataset_name="arcticv2",
        robot_type="xhand",
        embodiment_type="right",
        task="s01-ketchup_use_01",
        workspace_root=DEFAULT_WORKSPACE,
    )


def _run(hook, mode: str, friction: float, tag: str):
    return replay_spider_task(
        _cfg(),
        OUT / tag,
        save_video=False,
        log_energy=False,
        post_extend_s=2.0,
        post_lift_m=0.10,
        friction_scale=friction,
        extend_action_hook=hook,
        extend_hook_mode=mode,  # type: ignore[arg-type]
        extend_hook_trace_max=8,
        dataset_case_name=tag,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step2 PPO hook smoke")
    parser.add_argument("--ppo", type=Path, default=DEFAULT_PPO)
    parser.add_argument("--friction", type=float, default=0.5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Obs builder sanity
    dummy = ExtendStepInfo(
        step_i=0,
        n_extend=200,
        features=None,
        p_slip=None,
        slip_now=False,
        slip_active=False,
        soft_fire=False,
        grip_extra=0.0,
        wrist_cmd=(0.0, 0.0, 0.0),
        object_z=0.0,
        z_extend_start=0.0,
        friction_scale=0.5,
        mass_scale=1.0,
    )
    obs = build_ppo_obs_from_info(dummy)
    assert obs.shape == (PPO_OBS_DIM,), obs.shape

    print(f"Loading PPO {args.ppo} ...", flush=True)
    propose = PPOExtendHook.from_zip(args.ppo, apply_actions=False)
    apply = PPOExtendHook.from_zip(args.ppo, apply_actions=True)

    print("A) always + propose-only ...", flush=True)
    a = _run(propose, "always", args.friction, "always_propose")
    print(
        f"  dz={a.post_extend_object_dz*100:+.1f} q={a.extend_hook_queries} "
        f"ov={a.extend_hook_overrides} records={len(propose.records)}",
        flush=True,
    )

    print("B) always + apply ...", flush=True)
    b = _run(apply, "always", args.friction, "always_apply")
    print(
        f"  dz={b.post_extend_object_dz*100:+.1f} q={b.extend_hook_queries} "
        f"ov={b.extend_hook_overrides} records={len(apply.records)} "
        f"max_grip={b.antislip_max_grip:.3f}",
        flush=True,
    )

    print("C) on_detect without NN (expect q=0) ...", flush=True)
    c_hook = PPOExtendHook.from_zip(args.ppo, apply_actions=True)
    c = _run(c_hook, "on_detect", args.friction, "ondetect_no_nn")
    print(
        f"  dz={c.post_extend_object_dz*100:+.1f} q={c.extend_hook_queries} "
        f"ov={c.extend_hook_overrides}",
        flush=True,
    )

    ok = (
        a.extend_hook_overrides == 0
        and a.extend_hook_queries == 200
        and len(propose.records) == 200
        and b.extend_hook_overrides == 200
        and b.extend_hook_queries == 200
        and c.extend_hook_queries == 0
        and c.extend_hook_overrides == 0
    )
    report = {
        "step": 2,
        "ppo": str(args.ppo),
        "friction_scale": args.friction,
        "A_always_propose": {
            "dz_cm": a.post_extend_object_dz * 100,
            "queries": a.extend_hook_queries,
            "overrides": a.extend_hook_overrides,
            "sample": propose.records[:3],
        },
        "B_always_apply": {
            "dz_cm": b.post_extend_object_dz * 100,
            "queries": b.extend_hook_queries,
            "overrides": b.extend_hook_overrides,
            "max_grip": b.antislip_max_grip,
            "sample": apply.records[:3],
            "trace": b.extend_hook_trace[:5],
        },
        "C_on_detect_no_nn": {
            "dz_cm": c.post_extend_object_dz * 100,
            "queries": c.extend_hook_queries,
            "overrides": c.extend_hook_overrides,
            "note": "on_detect needs antislip_nn detect signals; Step3 wires that",
        },
        "checks_ok": bool(ok),
    }
    path = OUT / "step2_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "A_always_propose"}, indent=2)[:2000])
    print(f"Wrote {path} checks_ok={ok}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
