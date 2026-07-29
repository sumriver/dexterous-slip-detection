#!/usr/bin/env python3
"""Retrain NN-2 / grip-only / grip+wrist on unified dataset, then eval.

Algorithm classes (same train windows):
  - models/slip_nn_unified_nn2
  - models/slip_nn_unified_grip       (seed 42)
  - models/slip_nn_unified_wrist      (seed 42, mask-zero-wrist)
Optional variance seed:
  - models/slip_nn_unified_grip_s43
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified same-data retrain + eval")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "slip_nn_unified")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--epochs-detect", type=int, default=40)
    parser.add_argument("--epochs-policy", type=int, default=80)
    parser.add_argument(
        "--with-variance-seed",
        action="store_true",
        default=True,
        help="Also train grip-only seed 43 for variance (not ranked)",
    )
    parser.add_argument("--no-variance-seed", action="store_true")
    args = parser.parse_args()
    if args.no_variance_seed:
        args.with_variance_seed = False

    if not (args.data / "train" / "windows.npz").exists():
        print(f"Missing {args.data}/train/windows.npz — build unified data first", file=sys.stderr)
        sys.exit(2)

    out_nn2 = ROOT / "models" / "slip_nn_unified_nn2"
    out_grip = ROOT / "models" / "slip_nn_unified_grip"
    out_grip_s43 = ROOT / "models" / "slip_nn_unified_grip_s43"
    out_wrist = ROOT / "models" / "slip_nn_unified_wrist"

    if not args.skip_train:
        run(
            [
                sys.executable,
                "scripts/train_slip_multitask.py",
                "--data",
                str(args.data),
                "--out",
                str(out_nn2),
                "--epochs",
                str(args.epochs_detect),
                "--max-grip",
                "0.25",
            ]
        )
        run(
            [
                sys.executable,
                "scripts/train_slip_policy.py",
                "--data",
                str(args.data),
                "--backbone",
                str(out_nn2 / "slip_tcn_v1.pt"),
                "--out",
                str(out_grip),
                "--epochs",
                str(args.epochs_policy),
                "--max-grip",
                "0.25",
                "--seed",
                "42",
            ]
        )
        if args.with_variance_seed:
            run(
                [
                    sys.executable,
                    "scripts/train_slip_policy.py",
                    "--data",
                    str(args.data),
                    "--backbone",
                    str(out_nn2 / "slip_tcn_v1.pt"),
                    "--out",
                    str(out_grip_s43),
                    "--epochs",
                    str(args.epochs_policy),
                    "--max-grip",
                    "0.25",
                    "--seed",
                    "43",
                ]
            )
        run(
            [
                sys.executable,
                "scripts/train_slip_policy2.py",
                "--data",
                str(args.data),
                "--backbone",
                str(out_nn2 / "slip_tcn_v1.pt"),
                "--out",
                str(out_wrist),
                "--epochs",
                str(args.epochs_policy),
                "--max-grip",
                "0.25",
                "--seed",
                "42",
                "--mask-zero-wrist",
            ]
        )

    if not args.skip_eval:
        adapter = {
            "nn2": str(out_nn2),
            "grip": str(out_grip),
            "wrist": str(out_wrist),
            "grip_s43": str(out_grip_s43),
        }
        (ROOT / "data" / "slip_eval" / "unified_model_dirs.json").write_text(
            json.dumps(adapter, indent=2)
        )
        run(
            [
                sys.executable,
                "scripts/eval_slip_unified_suite.py",
                "--models-json",
                str(ROOT / "data" / "slip_eval" / "unified_model_dirs.json"),
            ]
        )


if __name__ == "__main__":
    main()
