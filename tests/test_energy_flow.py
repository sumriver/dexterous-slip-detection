"""Tests for energy-flow slip detection."""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy_flow import SlipDetector, compute_applied_power, compute_mass_estimate


def test_applied_power():
    forces = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    velocities = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
    power = compute_applied_power(forces, velocities)
    np.testing.assert_allclose(power, [0.1, 0.4])


def test_mass_estimate():
    applied = np.array([1.0, 2.0, 3.0])
    assert compute_mass_estimate(applied, 2.0) == pytest.approx(3.0)


def test_slip_detector_no_slip():
    det = SlipDetector(window_size=10, threshold=0.5)
    for _ in range(20):
        assert det.update(1.0) is False


def test_slip_detector_detects_spike():
    det = SlipDetector(window_size=10, threshold=0.1)
    for _ in range(10):
        det.update(1.0)
    assert det.update(5.0) is True
