"""NN-Policy-2: frozen detect backbone + grip+wrist policy head (P2-A)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from sim.antislip_control import Policy2Action
from sim.slip_nn_features import FEATURE_DIM
from sim.slip_nn_model import DEFAULT_HIDDEN, DEFAULT_MLP, SlipTCNMulti, count_params
from sim.slip_nn_policy import DEFAULT_POLICY_WIDTH, load_backbone_from_multitask_ckpt

DEFAULT_WRIST_MAX = 0.25


class SlipPolicy2Head(nn.Module):
    """Maps (h, p_slip, grip_extra) → (g*, Δr, Δp, Δy)."""

    def __init__(
        self,
        *,
        hidden: int = DEFAULT_MLP,
        width: int = DEFAULT_POLICY_WIDTH,
        max_grip: float = 0.25,
        max_wrist: float = DEFAULT_WRIST_MAX,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_grip = float(max_grip)
        self.max_wrist = float(max_wrist)
        self.width = int(width)
        in_dim = hidden + 2
        self.fc1 = nn.Linear(in_dim, width)
        self.ln1 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.fc_grip = nn.Linear(width, 1)
        self.fc_wrist = nn.Linear(width, 3)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self, h: torch.Tensor, p_slip: torch.Tensor, grip_extra: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return grip:(B,) wrist:(B,3)."""
        x = torch.cat([h, p_slip.unsqueeze(-1), grip_extra.unsqueeze(-1)], dim=-1)
        x = self.drop(self.act(self.ln1(self.fc1(x))))
        x = self.drop(self.act(self.ln2(self.fc2(x))))
        grip = torch.sigmoid(self.fc_grip(x).squeeze(-1)) * self.max_grip
        wrist = torch.tanh(self.fc_wrist(x)) * self.max_wrist
        return grip, wrist


class SlipDetectAndPolicy2(nn.Module):
    """Shared TCN-multi backbone + P2-A policy head (grip + wrist residual)."""

    def __init__(
        self,
        *,
        feature_dim: int = FEATURE_DIM,
        hidden: int = DEFAULT_HIDDEN,
        mlp_dim: int = DEFAULT_MLP,
        policy_width: int = DEFAULT_POLICY_WIDTH,
        max_grip: float = 0.25,
        max_wrist: float = DEFAULT_WRIST_MAX,
        policy_dropout: float = 0.0,
    ):
        super().__init__()
        self.max_grip = float(max_grip)
        self.max_wrist = float(max_wrist)
        self.policy_width = int(policy_width)
        self.backbone = SlipTCNMulti(
            feature_dim=feature_dim,
            hidden=hidden,
            mlp_dim=mlp_dim,
            max_grip=max_grip,
        )
        self.policy = SlipPolicy2Head(
            hidden=mlp_dim,
            width=policy_width,
            max_grip=max_grip,
            max_wrist=max_wrist,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (p_slip, g_ref, g_policy, wrist_delta) with wrist:(B,3)."""
        h = self.encode(x)
        slip_logit = self.backbone.fc_slip(h).squeeze(-1)
        p_slip = torch.sigmoid(slip_logit)
        g_ref = torch.sigmoid(self.backbone.fc_grip(h).squeeze(-1)) * self.max_grip
        g_pol, wrist = self.policy(h, p_slip, grip_extra)
        return p_slip, g_ref, g_pol, wrist

    def predict_action(
        self, x: torch.Tensor, grip_extra: torch.Tensor
    ) -> tuple[torch.Tensor, Policy2Action]:
        """Batch-size-1 helper → Policy2Action."""
        p, _, g, w = self.forward_policy(x, grip_extra)
        act = Policy2Action(
            grip=float(g[0].item()),
            wrist_delta=tuple(float(v) for v in w[0].detach().cpu().tolist()),
        )
        return p, act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def load_policy2_backbone(
    model: SlipDetectAndPolicy2,
    ckpt_path: str | Path,
    map_location: str = "cpu",
) -> None:
    load_backbone_from_multitask_ckpt(model, ckpt_path, map_location=map_location)  # type: ignore[arg-type]


def policy2_param_count(model: SlipDetectAndPolicy2) -> dict[str, int]:
    return {
        "policy_trainable": count_params(model.policy),
        "backbone_trainable": int(
            sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        ),
        "total": int(sum(p.numel() for p in model.parameters())),
        "policy_width": int(model.policy_width),
        "max_wrist": int(model.max_wrist * 1000),
    }
