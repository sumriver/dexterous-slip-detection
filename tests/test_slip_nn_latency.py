"""NN-1 CPU latency gate against shipped checkpoint (< 2 ms mean)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CKPT_DIR = ROOT / "models" / "slip_nn"


@pytest.mark.skipif(not any(CKPT_DIR.glob("*.pt")), reason="no slip_nn checkpoint")
def test_shipped_detector_latency_under_2ms():
    from bench_slip_nn_latency import bench

    m = bench(CKPT_DIR, warmup=20, reps=200, threshold=None)
    assert m["mean_ms"] < 2.0, m
    assert m["p95_ms"] < 5.0, m


@pytest.mark.skipif(not any(CKPT_DIR.glob("*.pt")), reason="no slip_nn checkpoint")
def test_load_detector_from_dir_reads_default_threshold():
    from sim.slip_nn_detector import load_detector_from_dir
    from sim.slip_nn_features import FEATURE_DIM

    det = load_detector_from_dir(CKPT_DIR, threshold=None)
    assert abs(det.threshold - 0.7) < 1e-6
    r = det.update(np.zeros(FEATURE_DIM, dtype=np.float32))
    assert 0.0 <= r.p_slip <= 1.0
