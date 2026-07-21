"""NN-Policy-1: frozen detect backbone + grip policy head (tier-A MLP)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from sim.slip_nn_features import FEATURE_DIM
from sim.slip_nn_model import DEFAULT_HIDDEN, DEFAULT_MLP, SlipTCNMulti, count_params

# Default policy MLP width (tier A: 34 → 64 → 64 → 1).
DEFAULT_POLICY_WIDTH = 64


class SlipPolicyHead(nn.Module):
    """Maps backbone embedding (+ p_slip, grip_extra) → target grip in [0, max_grip].

    Tier A default: two-layer MLP with width 64 (+ LayerNorm), ~8–10k params.
    """

    def __init__(
        self,
        *,
        hidden: int = DEFAULT_MLP,
        width: int = DEFAULT_POLICY_WIDTH,
        max_grip: float = 0.25,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_grip = float(max_grip)
        self.width = int(width)
        in_dim = hidden + 2  # h + p_slip + grip_extra
        self.fc1 = nn.Linear(in_dim, width)
        self.ln1 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.fc3 = nn.Linear(width, 1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor, p_slip: torch.Tensor, grip_extra: torch.Tensor) -> torch.Tensor:
        """h:(B,H) p_slip:(B,) grip_extra:(B,) → grip:(B,)"""
        x = torch.cat([h, p_slip.unsqueeze(-1), grip_extra.unsqueeze(-1)], dim=-1)
        x = self.drop(self.act(self.ln1(self.fc1(x))))
        x = self.drop(self.act(self.ln2(self.fc2(x))))
        return torch.sigmoid(self.fc3(x).squeeze(-1)) * self.max_grip


class SlipDetectAndPolicy(nn.Module):
    """Shared TCN-multi backbone; slip head frozen by default; trainable policy head."""

    def __init__(
        self,
        *,
        feature_dim: int = FEATURE_DIM,
        hidden: int = DEFAULT_HIDDEN,
        mlp_dim: int = DEFAULT_MLP,
        policy_width: int = DEFAULT_POLICY_WIDTH,
        max_grip: float = 0.25,
        residual: bool = False,
        policy_dropout: float = 0.0,
    ):
        super().__init__()
        self.max_grip = float(max_grip)
        self.residual = bool(residual)
        self.policy_width = int(policy_width)
        self.backbone = SlipTCNMulti(
            feature_dim=feature_dim,
            hidden=hidden,
            mlp_dim=mlp_dim,
            max_grip=max_grip,
        )
        self.policy = SlipPolicyHead(
            hidden=mlp_dim,
            width=policy_width,
            max_grip=max_grip,
            dropout=policy_dropout,
        )

    def freeze_detect(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_detect(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(x)

    def forward_policy(
        self,
        x: torch.Tensor,
        grip_extra: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (p_slip, g_ref_nn2, g_policy) with shapes (B,)."""
        h = self.encode(x)
        slip_logit = self.backbone.fc_slip(h).squeeze(-1)
        p_slip = torch.sigmoid(slip_logit)
        g_ref = torch.sigmoid(self.backbone.fc_grip(h).squeeze(-1)) * self.max_grip
        g_pol = self.policy(h, p_slip, grip_extra)
        if self.residual:
            g_pol = torch.clamp(g_ref + g_pol, 0.0, self.max_grip)
        return p_slip, g_ref, g_pol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Slip logits only (detector-compatible)."""
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def load_backbone_from_multitask_ckpt(
    model: SlipDetectAndPolicy,
    ckpt_path: str | Path,
    map_location: str = "cpu",
) -> None:
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.backbone.load_state_dict(state, strict=True)


def policy_param_count(model: SlipDetectAndPolicy) -> dict[str, int]:
    return {
        "policy_trainable": count_params(model.policy),
        "backbone_trainable": int(
            sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        ),
        "total": int(sum(p.numel() for p in model.parameters())),
        "policy_width": int(getattr(model, "policy_width", DEFAULT_POLICY_WIDTH)),
    }
