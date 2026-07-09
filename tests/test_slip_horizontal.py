"""Tests for horizontal-plane force integral signals."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim.slip_horizontal import HorizontalForceReading, HorizontalImpulseIntegrator


def _reading(fx: float, fy: float) -> HorizontalForceReading:
    return HorizontalForceReading(
        n_contacts=1,
        fx=fx,
        fy=fy,
        fx_normal=fx,
        fy_normal=fy,
        fx_tangent=0.0,
        fy_tangent=0.0,
        f_horiz_mag=float(np.hypot(fx, fy)),
    )


def test_cumulative_impulse():
    integ = HorizontalImpulseIntegrator(sim_dt=0.01)
    for _ in range(100):  # 100 steps * 0.01s = 1s at constant force
        integ.update(_reading(10.0, 0.0))
    assert np.isclose(integ.impulse_x, 10.0 * 1.0, atol=1e-6)
    assert np.isclose(integ.impulse_y, 0.0, atol=1e-6)
    assert np.isclose(integ.impulse_mag, 10.0, atol=1e-6)


def test_signed_impulse_cancels():
    integ = HorizontalImpulseIntegrator(sim_dt=0.01)
    for _ in range(50):
        integ.update(_reading(10.0, 0.0))
    for _ in range(50):
        integ.update(_reading(-10.0, 0.0))
    # signed X impulse cancels, but magnitude integral keeps accumulating
    assert np.isclose(integ.impulse_x, 0.0, atol=1e-6)
    assert np.isclose(integ.impulse_mag, 10.0, atol=1e-6)


def test_sliding_window_impulse():
    integ = HorizontalImpulseIntegrator(sim_dt=0.01, window_s=0.1)  # 10-step window
    for _ in range(100):
        integ.update(_reading(10.0, 0.0))
    # only the last 10 steps count: 10 * 0.01 * 10 = 1.0
    assert np.isclose(integ.impulse_x, 1.0, atol=1e-6)


def test_reset():
    integ = HorizontalImpulseIntegrator(sim_dt=0.01)
    integ.update(_reading(5.0, 5.0))
    integ.reset()
    assert integ.impulse_x == 0.0
    assert integ.impulse_y == 0.0
    assert integ.impulse_mag == 0.0
