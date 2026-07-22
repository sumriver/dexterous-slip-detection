"""Tests for NN-Policy-2 grip+wrist head (P2-A)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim.slip_nn_features import FEATURE_DIM
from sim.slip_nn_policy import DEFAULT_POLICY_WIDTH
from sim.slip_nn_policy2 import (
    DEFAULT_WRIST_MAX,
    SlipDetectAndPolicy2,
    SlipPolicy2Head,
    policy2_param_count,
)


def test_policy2_head_shape_and_clamp():
    head = SlipPolicy2Head(max_grip=0.25, max_wrist=DEFAULT_WRIST_MAX, width=DEFAULT_POLICY_WIDTH)
    b = 8
    h = torch.randn(b, 32)
    p = torch.rand(b)
    g = torch.rand(b) * 0.25
    grip, wrist = head(h, p, g)
    assert grip.shape == (b,)
    assert wrist.shape == (b, 3)
    assert torch.all(grip >= 0) and torch.all(grip <= 0.25 + 1e-5)
    assert torch.all(wrist.abs() <= DEFAULT_WRIST_MAX + 1e-5)


def test_detect_and_policy2_freeze():
    m = SlipDetectAndPolicy2(policy_width=64, max_wrist=DEFAULT_WRIST_MAX)
    m.freeze_detect()
    counts = policy2_param_count(m)
    assert counts["backbone_trainable"] == 0
    assert counts["policy_width"] == 64
    assert 1_000 < counts["policy_trainable"] < 25_000
    x = torch.randn(4, 40, FEATURE_DIM)
    grip = torch.zeros(4)
    p, gref, gpol, w = m.forward_policy(x, grip)
    assert p.shape == gref.shape == gpol.shape == (4,)
    assert w.shape == (4, 3)
    assert torch.all((gpol >= 0) & (gpol <= 0.25 + 1e-5))
    assert torch.all(w.abs() <= DEFAULT_WRIST_MAX + 1e-5)


def test_predict_action_returns_policy2():
    m = SlipDetectAndPolicy2()
    m.eval()
    x = torch.randn(1, 40, FEATURE_DIM)
    grip = torch.zeros(1)
    p, act = m.predict_action(x, grip)
    assert p.shape == (1,)
    assert 0.0 <= act.grip <= 0.25 + 1e-5
    assert len(act.wrist_delta) == 3
