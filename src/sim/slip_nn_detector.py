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


@dataclass
class SlipNnReading:
    p_slip: float
    slip_active: bool
    n_valid_steps: int


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
    ):
        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.window_steps = int(window_steps)
        self.norm = norm if isinstance(norm, NormStats) else NormStats.from_manifest(norm)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
            arch = arch or ckpt.get("arch", "tcn")
            feature_dim = int(ckpt.get("feature_dim", FEATURE_DIM))
        else:
            state = ckpt
            arch = arch or "tcn"
            feature_dim = FEATURE_DIM

        self.arch = arch
        self.model = build_slip_model(arch, feature_dim=feature_dim)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self._buf: list[np.ndarray] = []

    def reset_extend(self) -> None:
        self._buf.clear()

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
        self._buf.append(self.norm.transform(feat))
        if len(self._buf) > self.window_steps:
            self._buf.pop(0)

        n = len(self._buf)
        if n < self.window_steps:
            # Pad left with first frame (or zeros) until full window.
            pad = [self._buf[0]] * (self.window_steps - n)
            window = np.stack(pad + self._buf, axis=0)
        else:
            window = np.stack(self._buf, axis=0)

        x = torch.from_numpy(window).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = float(self.model.predict_proba(x).item())
        return SlipNnReading(
            p_slip=p,
            slip_active=p > self.threshold,
            n_valid_steps=n,
        )


def load_detector_from_dir(
    model_dir: Path,
    *,
    threshold: float = 0.5,
    device: str = "cpu",
) -> SlipNeuralDetector:
    """Load ``slip_tcn_v1.pt`` + sibling / data manifest norm."""
    model_dir = Path(model_dir)
    pt = model_dir / "slip_tcn_v1.pt"
    if not pt.exists():
        # allow any single .pt
        pts = sorted(model_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No checkpoint in {model_dir}")
        pt = pts[0]
    meta_path = model_dir / "train_meta.json"
    norm: NormStats | Path
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if "norm" in meta:
            norm = NormStats(
                mean=np.asarray(meta["norm"]["mean"], dtype=np.float32),
                std=np.asarray(meta["norm"]["std"], dtype=np.float32),
            )
        else:
            norm = Path(meta.get("manifest", "data/slip_nn/manifest.json"))
    else:
        norm = Path("data/slip_nn/manifest.json")
    arch = None
    if meta_path.exists():
        arch = json.loads(meta_path.read_text()).get("arch")
    return SlipNeuralDetector(pt, norm, threshold=threshold, device=device, arch=arch)
