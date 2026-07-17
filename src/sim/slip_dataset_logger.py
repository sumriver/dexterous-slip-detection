"""NN-0 dataset logger: per-step features → sliding windows → NPZ export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sim.slip_nn_features import FEATURE_DIM, FEATURE_NAMES, SlipFeatureLabels


@dataclass
class SlipDatasetMeta:
    step: int
    sim_time: float
    phase: str
    friction_scale: float
    mass_scale: float
    case_name: str = ""
    object_z: float = 0.0


@dataclass
class SlipDatasetLogger:
    """Accumulate per-step NN-0 features and export fixed-length windows."""

    window_steps: int = 40
    event_horizon_steps: int = 50  # 0.5 s @ 10 ms
    event_drop_m: float = 0.01  # 1 cm future drop → positive
    _features: list[np.ndarray] = field(default_factory=list)
    _labels: list[SlipFeatureLabels] = field(default_factory=list)
    _meta: list[SlipDatasetMeta] = field(default_factory=list)

    def append(
        self,
        features: np.ndarray,
        labels: SlipFeatureLabels,
        meta: SlipDatasetMeta,
    ) -> None:
        assert features.shape == (FEATURE_DIM,)
        self._features.append(features.astype(np.float32))
        self._labels.append(labels)
        self._meta.append(meta)

    @property
    def n_steps(self) -> int:
        return len(self._features)

    def clear(self) -> None:
        self._features.clear()
        self._labels.clear()
        self._meta.clear()

    def _event_labels(self) -> np.ndarray:
        """y_event[t]=1 if object drops by event_drop_m within the next H steps."""
        z = np.array([m.object_z for m in self._meta], dtype=np.float32)
        n = len(z)
        y = np.zeros(n, dtype=np.float32)
        h = self.event_horizon_steps
        for t in range(n):
            end = min(n, t + h + 1)
            if end <= t + 1:
                continue
            if float(np.min(z[t:end]) ) <= float(z[t]) - self.event_drop_m:
                y[t] = 1.0
        return y

    def _label_arrays(self) -> dict[str, np.ndarray]:
        if not self._labels:
            return {}
        return {
            "y_scheme1": np.array([l.y_scheme1 for l in self._labels], dtype=np.float32),
            "y_scheme2": np.array([l.y_scheme2 for l in self._labels], dtype=np.float32),
            "y_gt": np.array([l.y_gt for l in self._labels], dtype=np.float32),
            "y_fused": np.array([l.y_fused for l in self._labels], dtype=np.float32),
            "y_event": self._event_labels(),
            "y_grip": np.array([l.grip_extra for l in self._labels], dtype=np.float32),
            "slip_speed_m_s": np.array(
                [l.slip_speed_m_s for l in self._labels], dtype=np.float32
            ),
            "object_z": np.array([m.object_z for m in self._meta], dtype=np.float32),
        }

    def build_windows(self) -> dict[str, np.ndarray]:
        """Return windowed arrays: X (N,T,D), y_* (N,), meta fields (N,)."""
        n = self.n_steps
        t = self.window_steps
        if n < t:
            return {}

        feats = np.stack(self._features, axis=0)
        labels = self._label_arrays()
        n_win = n - t + 1

        # Stride-trick sliding windows without Python loop over windows.
        strides = feats.strides + feats.strides[:1]
        x = np.lib.stride_tricks.as_strided(
            feats,
            shape=(n_win, t, FEATURE_DIM),
            strides=(strides[0], strides[0], strides[1]),
        ).copy()

        out: dict[str, np.ndarray] = {"X": x}
        for key, arr in labels.items():
            out[key] = arr[t - 1 :]
        out["phase"] = np.array([m.phase for m in self._meta], dtype=object)[t - 1 :]
        out["friction_scale"] = np.array(
            [m.friction_scale for m in self._meta], dtype=np.float32
        )[t - 1 :]
        out["mass_scale"] = np.array(
            [m.mass_scale for m in self._meta], dtype=np.float32
        )[t - 1 :]
        out["case_name"] = np.array([m.case_name for m in self._meta], dtype=object)[t - 1 :]
        return out

    def save_npz(self, path: Path, *, compress: bool = True) -> int:
        """Write windowed dataset shard; returns number of windows."""
        path.parent.mkdir(parents=True, exist_ok=True)
        windows = self.build_windows()
        if not windows:
            return 0
        n_win = int(windows["X"].shape[0])
        if compress:
            np.savez_compressed(path, **windows)
        else:
            np.savez(path, **windows)
        return n_win


def compute_norm_stats(x_train: np.ndarray) -> dict[str, list[float]]:
    """Per-feature mean/std over (N,T,D) training windows."""
    flat = x_train.reshape(-1, FEATURE_DIM)
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return {"mean": mu.tolist(), "std": sigma.tolist()}


def write_manifest(
    path: Path,
    *,
    window_steps: int,
    n_train: int,
    n_val: int,
    n_test: int,
    norm_stats: dict[str, list[float]] | None = None,
    extra: dict | None = None,
) -> None:
    manifest = {
        "version": "nn0-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "window_steps": window_steps,
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "label_keys": ["y_scheme1", "y_scheme2", "y_gt", "y_fused", "y_event", "y_grip"],
        "counts": {"train": n_train, "val": n_val, "test": n_test},
        "norm": norm_stats or {},
    }
    if extra:
        manifest.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def merge_npz_shards(shards: list[Path]) -> dict[str, np.ndarray]:
    """Load and concatenate multiple NPZ shards."""
    if not shards:
        return {}
    merged: dict[str, list[np.ndarray]] = {}
    for shard in shards:
        data = np.load(shard, allow_pickle=True)
        for key in data.files:
            merged.setdefault(key, []).append(data[key])
    return {k: np.concatenate(v, axis=0) for k, v in merged.items()}


def split_by_case(
    windows: dict[str, np.ndarray],
    *,
    val_cases: set[str],
    test_cases: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split window arrays by case_name metadata."""
    cases = windows.get("case_name")
    if cases is None:
        raise ValueError("windows missing case_name for split")

    train_idx, val_idx, test_idx = [], [], []
    for i, name in enumerate(cases):
        s = str(name)
        if s in test_cases:
            test_idx.append(i)
        elif s in val_cases:
            val_idx.append(i)
        else:
            train_idx.append(i)

    def _take(idxs: list[int]) -> dict[str, np.ndarray]:
        if not idxs:
            return {}
        return {k: v[idxs] for k, v in windows.items()}

    return _take(train_idx), _take(val_idx), _take(test_idx)
