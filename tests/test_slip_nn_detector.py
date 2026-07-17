"""Tests for SlipNeuralDetector ring buffer + norm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim.slip_nn_features import FEATURE_DIM
from sim.slip_nn_detector import NormStats, SlipNeuralDetector
from sim.slip_nn_model import SlipTCN


def test_detector_pads_then_fires(tmp_path):
    model = SlipTCN()
    # Bias last layer so sigmoid ~0.9 for typical inputs after a few steps
    with torch.no_grad():
        model.fc2.bias.fill_(4.0)
    pt = tmp_path / "m.pt"
    torch.save({"model_state": model.state_dict(), "arch": "tcn", "feature_dim": FEATURE_DIM}, pt)

    mean = np.zeros(FEATURE_DIM, dtype=np.float32)
    std = np.ones(FEATURE_DIM, dtype=np.float32)
    det = SlipNeuralDetector(pt, NormStats(mean, std), threshold=0.5, window_steps=5)

    r0 = det.update(np.zeros(FEATURE_DIM, dtype=np.float32))
    assert r0.n_valid_steps == 1
    assert 0.0 <= r0.p_slip <= 1.0

    for _ in range(4):
        r = det.update(np.ones(FEATURE_DIM, dtype=np.float32))
    assert r.n_valid_steps == 5
    assert r.slip_active is True
    assert r.slip_now is True

    det.reset_extend()
    assert det.n_valid_steps == 0


def test_detector_latch(tmp_path):
    model = SlipTCN()
    with torch.no_grad():
        model.fc2.bias.fill_(-4.0)  # low p by default
    pt = tmp_path / "m.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "arch": "tcn",
            "feature_dim": FEATURE_DIM,
            "deploy_latch": True,
        },
        pt,
    )
    mean = np.zeros(FEATURE_DIM, dtype=np.float32)
    std = np.ones(FEATURE_DIM, dtype=np.float32)
    det = SlipNeuralDetector(pt, NormStats(mean, std), threshold=0.5, window_steps=3)
    assert det.latch is True
    # Force one high-p step by temporarily swapping bias
    with torch.no_grad():
        det.model.fc2.bias.fill_(4.0)
    r1 = det.update(np.zeros(FEATURE_DIM, dtype=np.float32))
    assert r1.slip_now and r1.slip_active
    with torch.no_grad():
        det.model.fc2.bias.fill_(-4.0)
    r2 = det.update(np.zeros(FEATURE_DIM, dtype=np.float32))
    assert r2.slip_now is False
    assert r2.slip_active is True  # latched


def test_norm_from_manifest(tmp_path):
    man = {
        "norm": {
            "mean": [0.0] * FEATURE_DIM,
            "std": [2.0] * FEATURE_DIM,
        }
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(man))
    n = NormStats.from_manifest(path)
    x = np.full(FEATURE_DIM, 4.0, dtype=np.float32)
    y = n.transform(x)
    assert np.allclose(y, 2.0)
