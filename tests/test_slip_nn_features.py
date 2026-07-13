"""Tests for NN-0 feature vector and dataset logger."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim.slip_dataset_logger import SlipDatasetLogger, compute_norm_stats, split_by_case
from sim.slip_nn_features import FEATURE_DIM, FEATURE_NAMES, SlipFeatureLabels


def _dummy_labels(slip: bool = False) -> SlipFeatureLabels:
    return SlipFeatureLabels(
        y_scheme1=slip,
        y_scheme2=slip,
        y_gt=slip,
        y_fused=slip,
        grip_extra=0.0,
        slip_speed_m_s=0.01 if slip else 0.0,
    )


def test_feature_dim_matches_names():
    assert len(FEATURE_NAMES) == FEATURE_DIM == 26


def test_sliding_windows_shape():
    logger = SlipDatasetLogger(window_steps=5)
    for i in range(12):
        feat = np.full(FEATURE_DIM, float(i), dtype=np.float32)
        from sim.slip_dataset_logger import SlipDatasetMeta

        logger.append(
            feat,
            _dummy_labels(i % 3 == 0),
            SlipDatasetMeta(
                step=i,
                sim_time=i * 0.01,
                phase="extend" if i > 6 else "trajectory",
                friction_scale=1.0,
                mass_scale=1.0,
                case_name="baseline",
            ),
        )
    windows = logger.build_windows()
    assert windows["X"].shape == (8, 5, FEATURE_DIM)
    assert windows["y_gt"].shape == (8,)
    assert windows["X"][0, 0, 0] == 0.0
    assert windows["X"][-1, -1, 0] == 11.0


def test_save_npz_roundtrip(tmp_path):
    logger = SlipDatasetLogger(window_steps=3)
    from sim.slip_dataset_logger import SlipDatasetMeta

    for i in range(6):
        logger.append(
            np.arange(FEATURE_DIM, dtype=np.float32),
            _dummy_labels(),
            SlipDatasetMeta(
                step=i,
                sim_time=0.0,
                phase="trajectory",
                friction_scale=0.5,
                mass_scale=2.0,
                case_name="friction_div2",
            ),
        )
    path = tmp_path / "shard.npz"
    n = logger.save_npz(path)
    assert n == 4
    data = np.load(path, allow_pickle=True)
    assert data["X"].shape == (4, 3, FEATURE_DIM)
    assert data["y_scheme2"].dtype == np.float32


def test_split_by_case():
    n = 6
    windows = {
        "X": np.zeros((n, 2, FEATURE_DIM)),
        "y_gt": np.zeros(n),
        "case_name": np.array(
            ["baseline", "baseline", "friction_div2", "friction_div4", "mass_x16", "mass_x2"],
            dtype=object,
        ),
    }
    train, val, test = split_by_case(
        windows,
        val_cases={"friction_div4", "mass_x16"},
        test_cases={"friction_div2"},
    )
    assert train["X"].shape[0] == 3  # baseline×2 + mass_x2
    assert val["X"].shape[0] == 2
    assert test["X"].shape[0] == 1


def test_norm_stats():
    x = np.random.randn(20, 4, FEATURE_DIM).astype(np.float32)
    stats = compute_norm_stats(x)
    assert len(stats["mean"]) == FEATURE_DIM
    assert len(stats["std"]) == FEATURE_DIM
