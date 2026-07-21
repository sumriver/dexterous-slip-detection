"""Tests for NN-Policy-1 policy head (tier A default)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim.slip_nn_features import FEATURE_DIM
from sim.slip_nn_policy import (
    DEFAULT_POLICY_WIDTH,
    SlipDetectAndPolicy,
    SlipPolicyHead,
    policy_param_count,
)


def test_policy_head_shape_and_clamp():
    head = SlipPolicyHead(max_grip=0.25, width=DEFAULT_POLICY_WIDTH)
    b = 8
    h = torch.randn(b, 32)
    p = torch.rand(b)
    g = torch.rand(b) * 0.25
    out = head(h, p, g)
    assert out.shape == (b,)
    assert torch.all(out >= 0) and torch.all(out <= 0.25 + 1e-5)


def test_policy_head_width_attribute():
    head = SlipPolicyHead(width=64)
    assert head.width == 64
    # 34 → 64 → 64 → 1 (+ LayerNorm): expect ~6–10k
    n = sum(p.numel() for p in head.parameters())
    assert 5_000 < n < 15_000


def test_detect_and_policy_freeze_tier_a():
    m = SlipDetectAndPolicy(policy_width=64)
    m.freeze_detect()
    counts = policy_param_count(m)
    assert counts["backbone_trainable"] == 0
    assert counts["policy_width"] == 64
    # Tier A: larger than tiny (1153), still under 20k budget
    assert 1_000 < counts["policy_trainable"] < 20_000
    x = torch.randn(4, 40, FEATURE_DIM)
    grip = torch.zeros(4)
    p, gref, gpol = m.forward_policy(x, grip)
    assert p.shape == gref.shape == gpol.shape == (4,)
    assert torch.all((gpol >= 0) & (gpol <= 0.25 + 1e-5))


def test_tiny_width_ablation_still_works():
    """Ablation path: --policy-width 32 keeps a smaller head trainable."""
    m = SlipDetectAndPolicy(policy_width=32)
    m.freeze_detect()
    counts = policy_param_count(m)
    assert counts["policy_trainable"] < counts["total"]
    assert counts["policy_trainable"] > 500
