"""NN-1 online detector: ring buffer + z-score + SlipTCN/GRU."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sim.slip_nn_features import FEATURE_DIM
from sim.slip_nn_model import DEFAULT_WINDOW, build_slip_model

# Teacher-leak / case-id channels zeroed when drop_leak_features=True.
IDX_SLIP_RULE_S2 = 17
IDX_PHASE_EXTEND = 24
IDX_FRICTION_SCALE = 25
LEAK_FEATURE_INDICES = (IDX_SLIP_RULE_S2, IDX_PHASE_EXTEND, IDX_FRICTION_SCALE)


@dataclass
class SlipNnReading:
    p_slip: float
    slip_active: bool  # grip trigger (includes optional latch)
    n_valid_steps: int
    slip_now: bool = False  # raw p > τ this step (for false-trigger metrics)


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any] | Path) -> "NormStats":
        if isinstance(manifest, Path):
            manifest = json.loads(manifest.read_text())
        norm = manifest.get("norm") or {}
        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        std = np.where(std < 1e-8, 1.0, std)
        if mean.shape != (FEATURE_DIM,) or std.shape != (FEATURE_DIM,):
            raise ValueError(f"norm dim mismatch: mean={mean.shape} std={std.shape}")
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x.astype(np.float32) - self.mean) / self.std).astype(np.float32)


class SlipNeuralDetector:
    """Maintain T-step feature window; emit p_slip each update."""

    def __init__(
        self,
        model_path: str | Path,
        norm: NormStats | dict[str, Any] | Path,
        *,
        threshold: float = 0.5,
        device: str = "cpu",
        window_steps: int = DEFAULT_WINDOW,
        arch: str | None = None,
        latch: bool | None = None,
        drop_leak_features: bool | None = None,
        confirm_steps: int = 1,
    ):
        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.window_steps = int(window_steps)
        self.norm = norm if isinstance(norm, NormStats) else NormStats.from_manifest(norm)
        self.confirm_steps = max(1, int(confirm_steps))

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
            arch = arch or ckpt.get("arch", "tcn")
            feature_dim = int(ckpt.get("feature_dim", FEATURE_DIM))
            if latch is None:
                latch = bool(ckpt.get("deploy_latch", False))
            if drop_leak_features is None:
                drop_leak_features = bool(ckpt.get("drop_leak_features", False))
            self.confirm_steps = max(1, int(ckpt.get("confirm_steps", self.confirm_steps)))
        else:
            state = ckpt
            arch = arch or "tcn"
            feature_dim = FEATURE_DIM
            if latch is None:
                latch = False
            if drop_leak_features is None:
                drop_leak_features = False

        self.latch = bool(latch)
        self.drop_leak_features = bool(drop_leak_features)
        self.arch = arch
        self.model = build_slip_model(arch, feature_dim=feature_dim)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self._buf: list[np.ndarray] = []
        self._latched = False
        self._high_run = 0

    def reset_extend(self) -> None:
        self._buf.clear()
        self._latched = False
        self._high_run = 0

    def reset(self) -> None:
        self.reset_extend()

    @property
    def n_valid_steps(self) -> int:
        return len(self._buf)

    def update(self, features: np.ndarray) -> SlipNnReading:
        """features: (D,) raw (unnormalized) current-step vector."""
        feat = np.asarray(features, dtype=np.float32).reshape(-1)
        if feat.shape[0] != FEATURE_DIM:
            raise ValueError(f"expected features ({FEATURE_DIM},), got {feat.shape}")

        if self.drop_leak_features:
            feat = feat.copy()
            for idx in LEAK_FEATURE_INDICES:
                feat[idx] = 0.0

        self._buf.append(self.norm.transform(feat))
        if len(self._buf) > self.window_steps:
            self._buf.pop(0)

        n = len(self._buf)
        if n < self.window_steps:
            pad = [self._buf[0]] * (self.window_steps - n)
            window = np.stack(pad + self._buf, axis=0)
        else:
            window = np.stack(self._buf, axis=0)

        x = torch.from_numpy(window).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = float(self.model.predict_proba(x).item())
        raw_high = p > self.threshold
        if raw_high:
            self._high_run += 1
        else:
            self._high_run = 0
        fired = self._high_run >= self.confirm_steps
        if self.latch and fired:
            self._latched = True
        active = bool(self._latched if self.latch else fired)
        return SlipNnReading(
            p_slip=p,
            slip_active=active,
            n_valid_steps=n,
            slip_now=fired,
        )


def load_detector_from_dir(
    model_dir: Path,
    *,
    threshold: float | None = None,
    device: str = "cpu",
) -> SlipNeuralDetector:
    """Load ``slip_tcn_v1.pt`` + sibling / data manifest norm."""
    model_dir = Path(model_dir)
    pt = model_dir / "slip_tcn_v1.pt"
    if not pt.exists():
        pts = sorted(model_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No checkpoint in {model_dir}")
        pt = pts[0]
    meta_path = model_dir / "train_meta.json"
    norm: NormStats | Path
    arch = None
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        arch = meta.get("arch")
        if "norm" in meta:
            norm = NormStats(
                mean=np.asarray(meta["norm"]["mean"], dtype=np.float32),
                std=np.asarray(meta["norm"]["std"], dtype=np.float32),
            )
        else:
            norm = Path(meta.get("manifest", "data/slip_nn/manifest.json"))
    else:
        norm = Path("data/slip_nn/manifest.json")
    if threshold is None:
        threshold = float(meta.get("default_threshold", 0.5))
    return SlipNeuralDetector(pt, norm, threshold=threshold, device=device, arch=arch)
