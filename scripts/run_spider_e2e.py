#!/usr/bin/env python3
"""SPIDER open-source E2E: trajectory replay → video → energy-flow contact log (CPU).

Default task: oakinkv2 / xhand / right / pick_spoon_bowl (no GPU, no RL).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.spider_replay import SpiderTaskConfig, replay_spider_task

SPIDER = ROOT / "third_party" / "spider"
DATASET = SPIDER / "example_datasets"
OUT_DIR = ROOT / "data" / "spider_e2e"


def main() -> None:
    parser = argparse.ArgumentParser(description="SPIDER E2E replay + energy-flow log")
    parser.add_argument("--dataset", default="oakinkv2", choices=["oakinkv2", "oakink", "gigahand"])
    parser.add_argument("--task", default="pick_spoon_bowl")
    parser.add_argument("--robot", default="xhand")
    parser.add_argument("--embodiment", default="right", help="right | bimanual")
    parser.add_argument("--data-type", default="mjwp_fast", help="mjwp_fast | mjwp | ikrollout")
    parser.add_argument("--data-id", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--copy-official-video", action="store_true")
    args = parser.parse_args()

    if not (SPIDER / "pyproject.toml").exists():
        print("SPIDER not found. Run: bash scripts/setup_spider.sh", file=sys.stderr)
        sys.exit(1)

    if args.dataset == "gigahand" and args.embodiment == "right":
        args.embodiment = "bimanual"
    if args.dataset == "gigahand":
        args.task = args.task if args.task != "pick_spoon_bowl" else "p36-tea"
        args.data_type = "mjwp" if args.data_type == "mjwp_fast" else args.data_type

    cfg = SpiderTaskConfig(
        dataset_dir=DATASET,
        dataset_name=args.dataset,
        robot_type=args.robot,
        embodiment_type=args.embodiment,
        task=args.task,
        data_id=args.data_id,
        data_type=args.data_type,
    )

    print(f"Scene:      {cfg.scene_path}")
    print(f"Trajectory: {cfg.trajectory_path}")
    result = replay_spider_task(cfg, OUT_DIR, save_video=not args.no_video)

    print("-" * 60)
    print(f"Steps:         {result.steps}")
    print(f"Contact steps: {result.contact_steps}")
    print(f"Slip events:   {result.slip_events}")
    print(f"Object Δz:     {result.object_dz * 100:.1f} cm  ({result.object_z_start:.3f} → {result.object_z_end:.3f} m)")
    if result.log_path:
        print(f"Energy log:    {result.log_path}")
    if result.video_path:
        print(f"Replay video:  {result.video_path}")

    if args.copy_official_video:
        official = cfg.trajectory_path.parent / f"visualization_{args.data_type}.mp4"
        if not official.exists():
            alt = cfg.trajectory_path.parent / f"visualization_{args.data_type.replace('_fast', '')}.mp4"
            official = alt if alt.exists() else official
        if official.exists() and official.stat().st_size > 1000:
            dst = OUT_DIR / f"{cfg.dataset_name}_{cfg.task}_official.mp4"
            shutil.copy2(official, dst)
            print(f"Official HF:   {dst}")

    ok = result.contact_steps > 0 and abs(result.object_dz) > 0.01
    print(f"E2E pass (contacts + object motion): {ok}")


if __name__ == "__main__":
    main()
