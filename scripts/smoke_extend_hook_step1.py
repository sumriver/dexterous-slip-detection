#!/usr/bin/env python3
"""Step-1 smoke: extend hook API with RecordingExtendHook (no action override).

Default path (no hook) vs inspect path (record-only) should match lift/contact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.extend_action_hook import RecordingExtendHook
from sim.spider_ketchup import DEFAULT_WORKSPACE
from sim.spider_replay import SpiderTaskConfig, replay_spider_task

SPIDER = ROOT / "third_party" / "spider"
OUT = ROOT / "data" / "rl" / "hook_step1"


def _cfg() -> SpiderTaskConfig:
    return SpiderTaskConfig(
        dataset_dir=SPIDER / "example_datasets",
        dataset_name="arcticv2",
        robot_type="xhand",
        embodiment_type="right",
        task="s01-ketchup_use_01",
        workspace_root=DEFAULT_WORKSPACE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step1 extend-hook smoke")
    parser.add_argument("--friction", type=float, default=0.5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _cfg()

    print("A) baseline (hook off) ...", flush=True)
    a = replay_spider_task(
        cfg,
        OUT / "off",
        save_video=False,
        log_energy=False,
        post_extend_s=2.0,
        post_lift_m=0.10,
        friction_scale=args.friction,
    )
    print(
        f"  dz={a.post_extend_object_dz*100:+.1f}cm contacts={a.post_extend_contact_steps} "
        f"hook={a.extend_hook_mode} q={a.extend_hook_queries} ov={a.extend_hook_overrides}",
        flush=True,
    )

    print("B) inspect + RecordingExtendHook (no override) ...", flush=True)
    rec = RecordingExtendHook()
    b = replay_spider_task(
        cfg,
        OUT / "inspect",
        save_video=False,
        log_energy=False,
        post_extend_s=2.0,
        post_lift_m=0.10,
        friction_scale=args.friction,
        extend_action_hook=rec,
        extend_hook_mode="inspect",
        extend_hook_trace_max=16,
    )
    print(
        f"  dz={b.post_extend_object_dz*100:+.1f}cm contacts={b.post_extend_contact_steps} "
        f"hook={b.extend_hook_mode} q={b.extend_hook_queries} ov={b.extend_hook_overrides} "
        f"records={len(rec.records)}",
        flush=True,
    )

    same = (
        abs(a.post_extend_object_dz - b.post_extend_object_dz) < 1e-6
        and a.post_extend_contact_steps == b.post_extend_contact_steps
        and b.extend_hook_overrides == 0
        and b.extend_hook_queries == 200
    )
    report = {
        "step": 1,
        "note": "Hook API only; RecordingExtendHook must not change physics",
        "friction_scale": args.friction,
        "baseline": {
            "dz_cm": a.post_extend_object_dz * 100,
            "contacts": a.post_extend_contact_steps,
        },
        "inspect": {
            "dz_cm": b.post_extend_object_dz * 100,
            "contacts": b.post_extend_contact_steps,
            "queries": b.extend_hook_queries,
            "overrides": b.extend_hook_overrides,
            "records": len(rec.records),
            "trace_head": b.extend_hook_trace[:5],
        },
        "physics_unchanged": bool(same),
    }
    path = OUT / "step1_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {path}")
    if not same:
        sys.exit(1)


if __name__ == "__main__":
    main()
