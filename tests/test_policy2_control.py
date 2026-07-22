"""Tests for Policy-2 open-loop wrist+grip controller."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim.antislip_control import (
    Policy2Action,
    Policy2OpenLoopController,
    apply_wrist_residual,
)


class _FakeModel:
    nu = 18

    def __init__(self):
        self.actuator_ctrlrange = np.zeros((18, 2), dtype=np.float64)
        self.actuator_ctrlrange[:3] = [-5, 5]
        self.actuator_ctrlrange[3:6] = [-6.2, 6.2]
        self.actuator_ctrlrange[6:] = [0, 2]


def test_apply_wrist_residual_clips():
    m = _FakeModel()
    ctrl = np.zeros(18)
    out = apply_wrist_residual(ctrl, m, [0.5, -0.5, 10.0])
    assert abs(out[3] - 0.5) < 1e-9
    assert abs(out[4] + 0.5) < 1e-9
    assert abs(out[5] - 6.2) < 1e-9  # clipped


def test_policy2_rate_limits_and_grip():
    m = _FakeModel()
    act = Policy2Action(grip=0.20, wrist_delta=(0.25, 0.0, -0.25))
    c = Policy2OpenLoopController(act, g_max=0.25, d_max=0.25, rate_g=0.05, rate_w=0.05)
    ctrl = np.zeros(18)
    out1 = c.apply(ctrl, m)
    assert c.grip_extra == 0.05
    assert abs(c.wrist_cmd[0] - 0.05) < 1e-9
    for _ in range(20):
        out1 = c.apply(ctrl, m)
    assert abs(c.grip_extra - 0.20) < 1e-9
    assert abs(c.wrist_cmd[0] - 0.25) < 1e-9
    assert abs(c.wrist_cmd[2] + 0.25) < 1e-9
    # fingers boosted
    assert out1[10] > 0
