#!/usr/bin/env python3
"""NN-0: export slip-detection feature windows from MuJoCo SPIDER replay.

Produces ``data/slip_nn/{train,val,test}/*.npz`` and ``manifest.json`` per
DS-SLIP-NN-001 §6. Labels include kinematic GT (``y_gt``) plus scheme-1/2 teachers.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.slip_dataset_logger import (
    SlipDatasetLogger,
    compute_norm_stats,
    merge_npz_shards,
    split_by_case,
    write_manifest,
)
from sim.slip_nn_features import FEATURE_DIM, SlipFeatureBuilder
from sim.spider_ketchup import DEFAULT_WORKSPACE
from sim.spider_replay import SpiderTaskConfig, replay_spider_task

SPIDER = ROOT / "third_party" / "spider"
OUT_DIR = ROOT / "data" / "slip_nn"

EXTEND_LIFT_TARGET_M = 0.10
EXTEND_S = 2.0
WINDOW_STEPS = 40

# Physics + motion variants multiply window count for NN-0 ≥10k train target.
EXTEND_VARIANTS: tuple[tuple[float, float], ...] = (
    (2.0, 0.10),
    (1.5, 0.08),
    (2.5, 0.12),
    (2.0, 0.14),
)

# Hold friction_div2 trajectory entirely in test (no train/val leakage).
VAL_CASES = {"friction_div4", "mass_x16"}
TEST_CASES = {"friction_div2"}


@dataclass(frozen=True)
class ExportCase:
    name: str
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    extend_s: float = EXTEND_S
    lift_m: float = EXTEND_LIFT_TARGET_M


def build_cases(*, include_variants: bool = True) -> list[ExportCase]:
    physics = [ExportCase("baseline")]
    for scale in (2, 4, 8, 16, 32):
        physics.append(ExportCase(f"mass_x{scale}", mass_scale=float(scale)))
    for div in (2, 4, 8):
        physics.append(ExportCase(f"friction_div{div}", friction_scale=1.0 / div))

    if not include_variants:
        return physics

    cases: list[ExportCase] = []
    for base in physics:
        for vi, (ext_s, lift_m) in enumerate(EXTEND_VARIANTS):
            suffix = "" if vi == 0 else f"_v{vi}"
            cases.append(
                ExportCase(
                    f"{base.name}{suffix}",
                    mass_scale=base.mass_scale,
                    friction_scale=base.friction_scale,
                    extend_s=ext_s,
                    lift_m=lift_m,
                )
            )
    return cases


def _base_case_name(case_name: str) -> str:
    for suffix in ("_v1", "_v2", "_v3"):
        if case_name.endswith(suffix):
            return case_name[: -len(suffix)]
    return case_name


def _workspace_ready() -> bool:
    return DEFAULT_WORKSPACE.joinpath("scene.xml").exists() and (
        DEFAULT_WORKSPACE / "trajectory_mjwp_fast.npz"
    ).stat().st_size > 1000


def _run_case(spec: ExportCase, *, window_steps: int) -> tuple[Path, int]:
    cfg = SpiderTaskConfig(
        dataset_dir=SPIDER / "example_datasets",
        dataset_name="arcticv2",
        robot_type="xhand",
        embodiment_type="right",
        task="s01-ketchup_use_01",
        workspace_root=DEFAULT_WORKSPACE,
    )
    logger = SlipDatasetLogger(window_steps=window_steps)
    builder = SlipFeatureBuilder(sim_dt=0.01)
    shard_dir = OUT_DIR / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{spec.name}.npz"

    replay_spider_task(
        cfg,
        OUT_DIR / "replay_logs" / spec.name,
        save_video=False,
        post_lift_m=spec.lift_m,
        post_extend_s=spec.extend_s,
        post_mimic_s=1.0,
        mass_scale=spec.mass_scale,
        friction_scale=spec.friction_scale,
        log_energy=False,
        antislip=False,
        dataset_logger=logger,
        feature_builder=builder,
        dataset_case_name=spec.name,
    )
    n_win = logger.save_npz(shard_path)
    return shard_path, n_win


def _save_split(name: str, data: dict[str, np.ndarray]) -> int:
    if not data:
        return 0
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "windows.npz"
    np.savez_compressed(path, **data)
    return int(data["X"].shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="NN-0 slip dataset export")
    parser.add_argument("--window", type=int, default=WINDOW_STEPS, help="Window length T")
    parser.add_argument("--case", default="", help="Export single case only")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Subset: baseline + friction_div2/4 + mass_x2/x4 (faster smoke)",
    )
    parser.add_argument(
        "--no-variants",
        action="store_true",
        help="Only base friction/mass grid (fewer windows)",
    )
    args = parser.parse_args()

    if not _workspace_ready():
        print(
            "Missing ketchup workspace. Run:\n"
            "  bash scripts/setup_spider.sh\n"
            "  python3 scripts/build_spider_ketchup_right.py",
            file=sys.stderr,
        )
        sys.exit(1)

    cases = build_cases(include_variants=not args.no_variants)
    if args.quick:
        keep = {"baseline", "friction_div2", "friction_div4", "mass_x2", "mass_x4"}
        cases = [c for c in cases if _base_case_name(c.name) in keep]
    if args.case:
        cases = [c for c in cases if c.name == args.case]
        if not cases:
            print(f"Unknown case: {args.case}", file=sys.stderr)
            sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    per_case: dict[str, int] = {}

    for spec in cases:
        print(f"Exporting {spec.name} (mass×{spec.mass_scale}, μ×{spec.friction_scale})...")
        shard, n_win = _run_case(spec, window_steps=args.window)
        shards.append(shard)
        per_case[spec.name] = n_win
        print(f"  steps logged → {n_win} windows")

    merged = merge_npz_shards(shards)
    if not merged:
        print("No windows exported.", file=sys.stderr)
        sys.exit(1)

    # Split on base physics case (strip motion variant suffix).
    base_cases = np.array([_base_case_name(str(n)) for n in merged["case_name"]], dtype=object)
    merged_for_split = dict(merged)
    merged_for_split["case_name"] = base_cases
    train, val, test = split_by_case(merged_for_split, val_cases=VAL_CASES, test_cases=TEST_CASES)
    n_train = _save_split("train", train)
    n_val = _save_split("val", val)
    n_test = _save_split("test", test)

    norm = compute_norm_stats(train["X"]) if n_train > 0 else {}
    write_manifest(
        OUT_DIR / "manifest.json",
        window_steps=args.window,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        norm_stats=norm,
        extra={
            "feature_dim": FEATURE_DIM,
            "per_case_windows": per_case,
            "val_cases": sorted(VAL_CASES),
            "test_cases": sorted(TEST_CASES),
            "extend_variants": [list(v) for v in EXTEND_VARIANTS],
        },
    )

    summary = {
        "total_windows": n_train + n_val + n_test,
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "per_case": per_case,
        "manifest": str(OUT_DIR / "manifest.json"),
    }
    (OUT_DIR / "export_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}  Total: {summary['total_windows']}")
    if n_train < 10_000:
        print(f"WARNING: train windows {n_train} < 10k — run full sweep (omit --quick)")
    print(f"Manifest: {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
