"""Tests for NN-Policy-1 policy head."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim.slip_nn_features import FEATURE_DIM
from sim.slip_nn_policy import SlipDetectAndPolicy, SlipPolicyHead, policy_param_count


def test_policy_head_shape_and_clamp():
    head = SlipPolicyHead(max_grip=0.25)
    b = 8
    h = torch.randn(b, 32)
    p = torch.rand(b)
    g = torch.rand(b) * 0.25
    out = head(h, p, g)
    assert out.shape == (b,)
    assert torch.all(out >= 0) and torch.all(out <= 0.25 + 1e-5)


def test_detect_and_policy_freeze():
    m = SlipDetectAndPolicy()
    m.freeze_detect()
    counts = policy_param_count(m)
    assert counts["backbone_trainable"] == 0
    assert 0 < counts["policy_trainable"] < 20_000
    x = torch.randn(4, 40, FEATURE_DIM)
    grip = torch.zeros(4)
    p, gref, gpol = m.forward_policy(x, grip)
    assert p.shape == gref.shape == gpol.shape == (4,)
    assert torch.all((gpol >= 0) & (gpol <= 0.25 + 1e-5))
