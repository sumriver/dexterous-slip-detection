#!/usr/bin/env python3
"""Retrain NN-2 / P1 / P2-grip-only / P2-wrist on unified dataset, then eval.

Assumes ``data/slip_nn_unified/{train,val,test}/windows.npz`` already built.
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
    args = parser.parse_args()

    if not (args.data / "train" / "windows.npz").exists():
        print(f"Missing {args.data}/train/windows.npz — build unified data first", file=sys.stderr)
        sys.exit(2)

    out_nn2 = ROOT / "models" / "slip_nn_unified_nn2"
    out_p1 = ROOT / "models" / "slip_nn_unified_p1"
    out_p2g = ROOT / "models" / "slip_nn_unified_p2_grip"
    out_p2w = ROOT / "models" / "slip_nn_unified_p2_wrist"

    if not args.skip_train:
        # NN-2 multitask on unified data (detect backbone for policies)
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
        # P1 / grip-only on same windows, backbone = unified NN-2
        run(
            [
                sys.executable,
                "scripts/train_slip_policy.py",
                "--data",
                str(args.data),
                "--backbone",
                str(out_nn2 / "slip_tcn_v1.pt"),
                "--out",
                str(out_p1),
                "--epochs",
                str(args.epochs_policy),
                "--max-grip",
                "0.25",
            ]
        )
        # Same grip-only path for explicit p2_grip alias dir
        run(
            [
                sys.executable,
                "scripts/train_slip_policy.py",
                "--data",
                str(args.data),
                "--backbone",
                str(out_nn2 / "slip_tcn_v1.pt"),
                "--out",
                str(out_p2g),
                "--epochs",
                str(args.epochs_policy),
                "--max-grip",
                "0.25",
                "--seed",
                "43",
            ]
        )
        # P2 grip+wrist (mask padded wrist=0 from non-P2 sources)
        run(
            [
                sys.executable,
                "scripts/train_slip_policy2.py",
                "--data",
                str(args.data),
                "--backbone",
                str(out_nn2 / "slip_tcn_v1.pt"),
                "--out",
                str(out_p2w),
                "--epochs",
                str(args.epochs_policy),
                "--max-grip",
                "0.25",
                "--mask-zero-wrist",
            ]
        )

    if not args.skip_eval:
        # Point discriminative suite presets at unified models via env-like symlink names
        # Write a small adapter JSON for the suite.
        adapter = {
            "nn2": str(out_nn2),
            "p1": str(out_p1),
            "p2_grip_only": str(out_p2g),
            "p2": str(out_p2w),
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
