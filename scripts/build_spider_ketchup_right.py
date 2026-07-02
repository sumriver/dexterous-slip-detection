#!/usr/bin/env python3
"""Build right-hand-only ketchup workspace from SPIDER bimanual s01-ketchup_use_01."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.spider_ketchup import DEFAULT_WORKSPACE, build_right_hand_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip left hand from ketchup SPIDER scene")
    parser.add_argument("--out", type=Path, default=DEFAULT_WORKSPACE)
    args = parser.parse_args()

    out = build_right_hand_workspace(args.out)
    model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    print(f"Workspace:  {out}")
    print(f"Scene:      {out / 'scene.xml'}")
    print(f"Trajectory: {out / 'trajectory_mjwp_fast.npz'}")
    print(f"Model DOF:  nq={model.nq} nv={model.nv} nu={model.nu}")


if __name__ == "__main__":
    main()
