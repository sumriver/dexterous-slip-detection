"""Shared NN-1 dataset / metric helpers used by train + offline eval."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sim.slip_nn_features import FEATURE_DIM


class SlipWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool = False,
        force_noise_idx: tuple[int, ...] = (7, 8, 9, 10, 13, 14, 15),
    ):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        self.augment = augment
        self.force_noise_idx = force_noise_idx

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = (self.x[idx] - self.mean) / self.std
        if self.augment:
            x = x.copy()
            for d in self.force_noise_idx:
                if d < x.shape[1]:
                    sigma = 0.05 * (np.abs(x[:, d]) + 1e-3)
                    x[:, d] += np.random.randn(x.shape[0]).astype(np.float32) * sigma
            t = x.shape[0]
            for ti in range(1, t):
                if random.random() < 0.05:
                    x[ti] = x[ti - 1]
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def load_split(data_dir: Path, split: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / split / "windows.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run scripts/export_slip_dataset.py")
    raw = np.load(path, allow_pickle=True)
    if label not in raw:
        raise KeyError(f"{label} not in {path} keys={raw.files}")
    x = raw["X"].astype(np.float32)
    if x.shape[-1] != FEATURE_DIM:
        raise ValueError(f"feature dim {x.shape[-1]} != {FEATURE_DIM}")
    return x, raw[label].astype(np.float32)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5
) -> dict[str, float]:
    y_hat = (y_prob >= thr).astype(np.float32)
    tp = float(np.sum((y_hat == 1) & (y_true == 1)))
    fp = float(np.sum((y_hat == 1) & (y_true == 0)))
    fn = float(np.sum((y_hat == 0) & (y_true == 1)))
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"precision": prec, "recall": rec, "f1": f1, "threshold": thr}


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thr: float = 0.5,
) -> dict[str, float]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logit = model(xb)
        p = torch.sigmoid(logit).cpu().numpy()
        probs.append(p)
        labels.append(yb.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    out = classification_metrics(y_true, y_prob, thr)
    out["n"] = float(len(y_true))
    out["pos_rate"] = float(y_true.mean())
    return out

