#!/usr/bin/env python3
"""CPU latency bench for SlipNeuralDetector (NN-1 gate: mean < 2 ms)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.slip_nn_detector import load_detector_from_dir  # noqa: E402
from sim.slip_nn_features import FEATURE_DIM  # noqa: E402


def bench(model_dir: Path, *, warmup: int, reps: int, threshold: float | None) -> dict:
    det = load_detector_from_dir(model_dir, threshold=threshold, device="cpu")
    x = np.zeros(FEATURE_DIM, dtype=np.float32)
    for _ in range(warmup):
        det.update(x)
    times_ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        det.update(x)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
        "warmup": warmup,
        "reps": reps,
        "gate_ms": 2.0,
        "pass": bool(arr.mean() < 2.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bench NN-1 detector latency")
    parser.add_argument("--nn-model-dir", type=Path, default=ROOT / "models" / "slip_nn")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--nn-threshold", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not any(args.nn_model_dir.glob("*.pt")):
        print(f"Missing checkpoint in {args.nn_model_dir}", file=sys.stderr)
        sys.exit(2)

    metrics = bench(args.nn_model_dir, warmup=args.warmup, reps=args.reps, threshold=args.nn_threshold)
    print(json.dumps(metrics, indent=2))
    out = args.out or (args.nn_model_dir / "latency.json")
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")
    if not metrics["pass"]:
        print(f"FAIL: mean latency {metrics['mean_ms']:.3f} ms >= 2 ms", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
