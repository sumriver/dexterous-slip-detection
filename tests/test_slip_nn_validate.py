"""Unit tests for NN-0 L2/L3 validators (synthetic NPZ windows)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim.slip_nn_features import FEATURE_DIM
from sim.slip_nn_validate import (
    validate_l2,
    validate_l3,
    base_case_name,
    per_case_label_table,
)


def _make_windows(
    *,
    n: int,
    cases: list[str],
    y1: np.ndarray | None = None,
    y2: np.ndarray | None = None,
    phase_extend: float = 1.0,
    friction: float = 1.0,
) -> dict[str, np.ndarray]:
    assert len(cases) == n
    x = np.zeros((n, 40, FEATURE_DIM), dtype=np.float32)
    x[:, :, 0] = 3.0  # n_contacts
    x[:, :, 8] = 10.0  # S_raw
    x[:, -1, 24] = phase_extend
    x[:, -1, 25] = friction
    if y1 is None:
        y1 = np.zeros(n, dtype=np.float32)
    if y2 is None:
        y2 = np.zeros(n, dtype=np.float32)
    yf = np.maximum(y1, y2)
    return {
        "X": x,
        "y_scheme1": y1,
        "y_scheme2": y2,
        "y_gt": y2.copy(),
        "y_fused": yf,
        "y_grip": np.zeros(n, dtype=np.float32),
        "case_name": np.array(cases, dtype=object),
    }


def test_base_case_name():
    assert base_case_name("friction_div2_v2") == "friction_div2"
    assert base_case_name("baseline") == "baseline"


def test_l2_fused_and_leakage_pass():
    n = 60
    train = _make_windows(
        n=n,
        cases=["baseline"] * 30 + ["mass_x2"] * 30,
        y1=np.array([0, 1] * 30, dtype=np.float32),
        y2=np.array([0] * 30 + [1] * 30, dtype=np.float32),
    )
    val = _make_windows(
        n=n,
        cases=["friction_div4"] * 30 + ["mass_x16"] * 30,
        y1=np.ones(n, dtype=np.float32) * 0.0,
        y2=np.array([i % 2 for i in range(n)], dtype=np.float32),
    )
    test = _make_windows(
        n=n,
        cases=["friction_div2"] * n,
        y1=np.zeros(n, dtype=np.float32),
        y2=np.ones(n, dtype=np.float32) * 0.5,  # will fail range if not 0/1 — use 0/1
    )
    test["y_scheme2"] = np.array([i % 2 for i in range(n)], dtype=np.float32)
    test["y_gt"] = test["y_scheme2"].copy()
    test["y_fused"] = np.maximum(test["y_scheme1"], test["y_scheme2"])

    report = validate_l2({"train": train, "val": val, "test": test})
    assert report.ok, [c for c in report.checks if not c.ok]


def test_l2_detects_fused_bug():
    n = 60
    w = _make_windows(
        n=n,
        cases=["baseline"] * n,
        y1=np.ones(n, dtype=np.float32),
        y2=np.zeros(n, dtype=np.float32),
    )
    w["y_fused"] = np.zeros(n, dtype=np.float32)  # wrong
    report = validate_l2({"train": w})
    assert not report.ok
    assert any(c.name.endswith("y_fused=y1|y2") and not c.ok for c in report.checks)


def test_l2_detects_leakage():
    n = 60
    train = _make_windows(
        n=n,
        cases=["friction_div2"] * n,  # should be test-only
        y1=np.array([i % 2 for i in range(n)], dtype=np.float32),
        y2=np.array([i % 3 == 0 for i in range(n)], dtype=np.float32),
    )
    test = _make_windows(
        n=n,
        cases=["friction_div2"] * n,
        y1=np.array([i % 2 for i in range(n)], dtype=np.float32),
        y2=np.array([i % 3 == 0 for i in range(n)], dtype=np.float32),
    )
    report = validate_l2({"train": train, "test": test})
    assert not report.ok
    assert any(c.name == "leak.test_cases_isolated" and not c.ok for c in report.checks)


def test_l3_traj_dz_and_extend_rates():
    n = 80
    x = np.zeros((n, 40, FEATURE_DIM), dtype=np.float32)
    x[:, :, 0] = 4.0
    x[:, :, 8] = 5.0
    # half traj / half extend
    x[:40, -1, 24] = 0.0
    x[40:, -1, 24] = 1.0
    # traj must have dz_traj_end == 0
    x[:40, -1, 19] = 0.0
    x[40:, -1, 19] = 0.05
    y2 = np.zeros(n, dtype=np.float32)
    y2[40:] = 1.0  # extend higher
    y1 = np.zeros(n, dtype=np.float32)
    data = {
        "X": x,
        "y_scheme1": y1,
        "y_scheme2": y2,
        "y_gt": y2.copy(),
        "y_fused": y2.copy(),
        "y_grip": np.zeros(n, dtype=np.float32),
        "case_name": np.array(["baseline"] * n, dtype=object),
    }
    # pad train size soft gate off
    report = validate_l3({"train": data}, require_10k_train=False)
    assert report.ok, [c for c in report.checks if not c.ok]


def test_per_case_table():
    n = 10
    w = _make_windows(n=n, cases=["baseline"] * 5 + ["mass_x2"] * 5)
    rows = per_case_label_table({"train": w})
    assert {r["case"] for r in rows} == {"baseline", "mass_x2"}
