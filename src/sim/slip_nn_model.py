"""NN-1 slip classifiers: 1D-TCN baseline and optional GRU."""

from __future__ import annotations

import torch
import torch.nn as nn

from sim.slip_nn_features import FEATURE_DIM

DEFAULT_WINDOW = 40
DEFAULT_HIDDEN = 64
DEFAULT_MLP = 32


def count_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


class SlipTCN(nn.Module):
    """1D-TCN + MLP head → logits (use BCEWithLogitsLoss) or probs via sigmoid."""

    def __init__(
        self,
        *,
        feature_dim: int = FEATURE_DIM,
        hidden: int = DEFAULT_HIDDEN,
        mlp_dim: int = DEFAULT_MLP,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv1d(feature_dim, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → logits (B,)."""
        # (B, D, T)
        h = x.transpose(1, 2)
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        h = self.act(self.fc1(h))
        return self.fc2(h).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class SlipGRU(nn.Module):
    """Optional GRU baseline with same I/O as SlipTCN."""

    def __init__(
        self,
        *,
        feature_dim: int = FEATURE_DIM,
        hidden: int = DEFAULT_HIDDEN,
        mlp_dim: int = DEFAULT_MLP,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.gru = nn.GRU(feature_dim, hidden, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → logits (B,)."""
        out, _ = self.gru(x)
        h = out[:, -1, :]
        h = self.act(self.fc1(h))
        return self.fc2(h).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def build_slip_model(arch: str = "tcn", **kwargs) -> nn.Module:
    arch = arch.lower()
    if arch == "tcn":
        return SlipTCN(**kwargs)
    if arch == "gru":
        return SlipGRU(**kwargs)
    raise ValueError(f"Unknown arch: {arch} (expected tcn|gru)")
