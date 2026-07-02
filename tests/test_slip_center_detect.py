"""Tests for center-divergence slip detection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim.slip_center_detect import contact_center_weighted


def test_contact_center_weighted():
    forces = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 5.0]])
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    c = contact_center_weighted(forces, positions)
    assert c is not None
    # heavier weight on first contact
    assert c[0] < 0.05
