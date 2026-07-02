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

from sim.grasp_validate import format_grasp_report
from sim.spider_ketchup import DEFAULT_WORKSPACE as KETCHUP_RIGHT_WS
from sim.spider_replay import SpiderTaskConfig, replay_spider_task

SPIDER = ROOT / "third_party" / "spider"
DATASET = SPIDER / "example_datasets"
OUT_DIR = ROOT / "data" / "spider_e2e"


def main() -> None:
    parser = argparse.ArgumentParser(description="SPIDER E2E replay + energy-flow log")
    parser.add_argument(
        "--dataset",
        default="oakinkv2",
        choices=["oakinkv2", "oakink", "gigahand", "arcticv2"],
    )
    parser.add_argument("--task", default="pick_spoon_bowl")
    parser.add_argument("--robot", default="xhand")
    parser.add_argument("--embodiment", default="right", help="right | bimanual")
    parser.add_argument("--data-type", default="mjwp_fast", help="mjwp_fast | mjwp | ikrollout")
    parser.add_argument("--data-id", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--copy-official-video", action="store_true")
    parser.add_argument(
        "--lift",
        type=float,
        default=0.0,
        metavar="M",
        help="Wrist tz raise in metres: with --extend, ramps during extension; "
        "otherwise resets to grasp frame then lifts (legacy)",
    )
    parser.add_argument(
        "--extend",
        type=float,
        default=0.0,
        metavar="S",
        help="Append S seconds after trajectory: mimic --mimic-last seconds while raising wrist by --lift",
    )
    parser.add_argument(
        "--mimic-last",
        type=float,
        default=1.0,
        metavar="S",
        help="Tail duration to loop during --extend (default 1s)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Use custom workspace dir (scene.xml + trajectory at root) instead of SPIDER processed path",
    )
    parser.add_argument(
        "--ketchup-right",
        action="store_true",
        help="Replay right-hand-only ketchup workspace (data/spider_ketchup_right)",
    )
    args = parser.parse_args()

    if not (SPIDER / "pyproject.toml").exists():
        print("SPIDER not found. Run: bash scripts/setup_spider.sh", file=sys.stderr)
        sys.exit(1)

    workspace = args.workspace
    if args.ketchup_right:
        workspace = workspace or KETCHUP_RIGHT_WS
        args.dataset = "arcticv2"
        args.task = "s01-ketchup_use_01"
        args.embodiment = "right"
    elif args.dataset in ("gigahand", "arcticv2") and args.embodiment == "right":
        args.embodiment = "bimanual"
    if args.dataset == "arcticv2" and args.task == "pick_spoon_bowl":
        args.task = "s01-ketchup_use_01"
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
        workspace_root=workspace,
    )

    print(f"Scene:      {cfg.scene_path}")
    print(f"Trajectory: {cfg.trajectory_path}")
    lift_m = args.lift
    if args.extend > 0 and lift_m == 0.0:
        lift_m = 0.10
    result = replay_spider_task(
        cfg,
        OUT_DIR,
        save_video=not args.no_video,
        post_lift_m=lift_m,
        post_extend_s=args.extend,
        post_mimic_s=args.mimic_last,
    )

    print("-" * 60)
    print(f"Steps:         {result.steps}")
    print(f"Contact steps: {result.contact_steps}")
    print(f"Slip events:   {result.slip_events}")
    print(f"Object Δz:     {result.object_dz * 100:.1f} cm  ({result.object_z_start:.3f} → {result.object_z_end:.3f} m)")
    if args.extend > 0:
        print(
            f"Post-extend:   {args.extend:.1f}s mimic-last={args.mimic_last:.1f}s  "
            f"wrist +{lift_m * 100:.0f}cm  "
            f"object Δz={result.post_extend_object_dz * 100:.1f} cm  "
            f"contact_steps={result.post_extend_contact_steps}"
        )
    elif args.lift > 0:
        print(
            f"Post-lift:     target={args.lift * 100:.0f} cm  "
            f"object Δz={result.post_lift_dz * 100:.1f} cm  "
            f"contact_steps={result.post_lift_contact_steps}"
        )
        if result.grasp_report:
            print("Grasp physics check:")
            print(format_grasp_report(result.grasp_report))
        if not result.grasp_physics_ok:
            print(
                "Lift skipped: grasp does not satisfy static vertical-support physics. "
                "pick_spoon_bowl is a scoop-on-plate task — contacts are on the handle "
                "above COM while the bowl end rests on the floor."
            )
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

    ok = result.contact_steps > 0 and abs(result.object_dz) > 0.001
    if args.extend > 0:
        ok = ok and result.post_extend_contact_steps > 0
    elif args.lift > 0:
        ok = (
            result.grasp_physics_ok
            and result.post_lift_dz >= 0.8 * args.lift
            and result.post_lift_contact_steps > 0
        )
    print(f"E2E pass: {ok}")


if __name__ == "__main__":
    main()
