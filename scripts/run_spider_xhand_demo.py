#!/usr/bin/env python3
"""Replay SPIDER pre-optimized XHAND trajectory (CPU MuJoCo, no PPO).

Uses Meta SPIDER example data: gigahand / xhand / bimanual / p36-tea.
MJWP *optimization* requires CUDA; this replays an existing mjwp trajectory.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPIDER = ROOT / "third_party" / "spider"
OUT_DIR = ROOT / "data" / "spider"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay SPIDER XHAND mjwp demo on CPU")
    parser.add_argument("--task", default="p36-tea")
    parser.add_argument("--data-id", type=int, default=0)
    parser.add_argument("--data-type", default="mjwp", choices=["kinematic", "mjwp", "ikrollout"])
    parser.add_argument("--copy-official-video", action="store_true", help="Also copy HF bundled mp4")
    args = parser.parse_args()

    if not (SPIDER / "pyproject.toml").exists():
        print("SPIDER not found. Run: bash scripts/setup_spider.sh", file=sys.stderr)
        sys.exit(1)

    traj = (
        SPIDER
        / "example_datasets"
        / "processed"
        / "gigahand"
        / "xhand"
        / "bimanual"
        / args.task
        / str(args.data_id)
        / f"trajectory_{args.data_type}.npz"
    )
    if not traj.exists() or traj.stat().st_size < 1000:
        print(f"Missing trajectory (run setup_spider.sh / git lfs pull): {traj}", file=sys.stderr)
        sys.exit(1)

    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    cmd = [
        uv,
        "run",
        "spider/viewers/mjcpu_viewer.py",
        "--dataset-dir",
        "example_datasets",
        "--dataset-name",
        "gigahand",
        "--robot-type",
        "xhand",
        "--embodiment-type",
        "bimanual",
        "--task",
        args.task,
        "--data-type",
        args.data_type,
        "--data-id",
        str(args.data_id),
        "--no-show-viewer",
        "--save-video",
        "--replay-speed",
        "2",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=SPIDER, check=True)

    src = (
        ROOT
        / "recordings"
        / "xhand"
        / "bimanual"
        / "mjcpu"
        / args.task
        / f"mjcpu_xhand_bimanual_{args.task}.mp4"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f"xhand_{args.task}_{args.data_type}_replay.mp4"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Video: {dst}")

    if args.copy_official_video:
        official = traj.parent / f"visualization_{args.data_type}.mp4"
        if official.exists() and official.stat().st_size > 1000:
            odst = OUT_DIR / f"xhand_{args.task}_{args.data_type}_official.mp4"
            shutil.copy2(official, odst)
            print(f"Official HF video: {odst}")


if __name__ == "__main__":
    main()
