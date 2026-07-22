#!/usr/bin/env python3
"""Re-merge policy shards and re-split with current POLICY_VAL/TEST_CASES.

Use after changing the policy split without re-running expensive teachers.
Reads ``data/slip_nn_policy/shards/*.npz`` → train/val/test + manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_slip_dataset import (  # noqa: E402
    POLICY_TEST_CASES,
    POLICY_VAL_CASES,
    WINDOW_STEPS,
    _base_case_name,
    build_policy_cases,
)
from sim.slip_dataset_logger import (  # noqa: E402
    compute_norm_stats,
    merge_npz_shards,
    split_by_case,
    write_manifest,
)
from sim.slip_nn_features import FEATURE_DIM  # noqa: E402


def _save_split(out_dir: Path, name: str, data: dict[str, np.ndarray]) -> int:
    if not data:
        return 0
    split = out_dir / name
    split.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(split / "windows.npz", **data)
    return int(data["X"].shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-split policy shards")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "slip_nn_policy")
    args = parser.parse_args()

    shards_all = sorted((args.data / "shards").glob("*.npz"))
    allow = {c.name for c in build_policy_cases(include_variants=True)}
    shards = [p for p in shards_all if p.stem in allow]
    skipped = sorted({p.stem for p in shards_all} - allow)
    if skipped:
        print(f"Skipping {len(skipped)} non-policy shards (e.g. {skipped[:3]}...)")
    if not shards:
        print(f"No policy shards in {args.data / 'shards'}", file=sys.stderr)
        sys.exit(2)

    merged = merge_npz_shards(shards)
    if not merged or "y_policy" not in merged:
        print("Missing y_policy in merged shards", file=sys.stderr)
        sys.exit(1)

    base_cases = np.array([_base_case_name(str(n)) for n in merged["case_name"]], dtype=object)
    merged_for_split = dict(merged)
    merged_for_split["case_name"] = base_cases
    train, val, test = split_by_case(
        merged_for_split,
        val_cases=POLICY_VAL_CASES,
        test_cases=POLICY_TEST_CASES,
    )
    n_train = _save_split(args.data, "train", train)
    n_val = _save_split(args.data, "val", val)
    n_test = _save_split(args.data, "test", test)

    norm = compute_norm_stats(train["X"]) if n_train else {}
    case_meta_path = args.data / "case_meta.json"
    case_meta = json.loads(case_meta_path.read_text()) if case_meta_path.exists() else []

    prev = {}
    man_path = args.data / "manifest.json"
    if man_path.exists():
        prev = json.loads(man_path.read_text())

    write_manifest(
        man_path,
        window_steps=int(prev.get("window_steps", WINDOW_STEPS)),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        norm_stats=norm,
        extra={
            "feature_dim": FEATURE_DIM,
            "dataset": "slip_nn_policy",
            "policy_teacher": prev.get("policy_teacher", "nn2"),
            "nn_model_dir": prev.get("nn_model_dir", ""),
            "dz_margin_m": prev.get("dz_margin_m"),
            "grip_floor": prev.get("grip_floor"),
            "y_policy_agg": prev.get("y_policy_agg", "window_end"),
            "val_cases": sorted(POLICY_VAL_CASES),
            "test_cases": sorted(POLICY_TEST_CASES),
            "policy_dose_grid": True,
            "resplit_only": True,
            "per_case": case_meta if case_meta else prev.get("per_case", []),
        },
    )

    y = train["y_policy"] if n_train else np.zeros(1)
    summary = {
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "total": n_train + n_val + n_test,
        "y_policy_train_max": float(np.max(y)),
        "y_policy_train_mean": float(np.mean(y)),
        "y_policy_train_frac_ge_0_13": float(np.mean(y >= 0.13)),
        "y_policy_train_frac_ge_0_05": float(np.mean(y >= 0.05)),
        "val_cases": sorted(POLICY_VAL_CASES),
        "test_cases": sorted(POLICY_TEST_CASES),
        "dose_grid": True,
        "resplit_only": True,
        "manifest": str(man_path),
    }
    (args.data / "export_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    # Confirm s045 in train
    tr_cases = set(str(c) for c in train.get("case_name", []))
    print(f"friction_s045 in train: {'friction_s045' in tr_cases}")
    print(f"val bases present: {sorted(set(str(c) for c in val.get('case_name', [])))}")


if __name__ == "__main__":
    main()
