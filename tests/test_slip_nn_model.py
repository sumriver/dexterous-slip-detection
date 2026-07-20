"""Tests for NN-1 SlipTCN / SlipGRU."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim.slip_nn_features import FEATURE_DIM
from sim.slip_nn_model import SlipGRU, SlipTCN, build_slip_model, count_params


def test_tcn_forward_shape():
    m = SlipTCN()
    x = torch.randn(8, 40, FEATURE_DIM)
    logits = m(x)
    assert logits.shape == (8,)
    p = m.predict_proba(x)
    assert p.shape == (8,)
    assert torch.all((p >= 0) & (p <= 1))


def test_gru_forward_shape():
    m = SlipGRU()
    x = torch.randn(4, 40, FEATURE_DIM)
    assert m(x).shape == (4,)


def test_param_budget():
    tcn = SlipTCN()
    n = count_params(tcn)
    assert n < 50_000, f"TCN params {n} >= 50k"
    gru = SlipGRU()
    assert count_params(gru) < 100_000


def test_build_slip_model():
    assert isinstance(build_slip_model("tcn"), SlipTCN)
    assert isinstance(build_slip_model("gru"), SlipGRU)


def test_tcn_multi_forward():
    from sim.slip_nn_model import SlipTCNMulti

    m = SlipTCNMulti()
    x = torch.randn(4, 40, FEATURE_DIM)
    assert m(x).shape == (4,)
    slip, grip = m.forward_multi(x)
    assert slip.shape == (4,)
    assert grip.shape == (4,)
    assert torch.all((grip >= 0) & (grip <= 0.25 + 1e-5))
    assert count_params(m) < 80_000
    assert isinstance(build_slip_model("tcn_multi"), SlipTCNMulti)
