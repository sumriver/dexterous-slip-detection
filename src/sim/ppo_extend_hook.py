"""PPO adapter for ``ExtendActionHook`` (Step 2).

Loads an SB3 PPO zip trained on ``KetchupExtendRLEnv`` and maps
``ExtendStepInfo`` → ``Policy2Action``. Still gated by
``extend_hook_mode`` in ``replay_spider_task``.

``apply_actions=False`` → propose+record only (safe inspect; no override).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sim.antislip_control import Policy2Action
from sim.extend_action_hook import ExtendStepDecision, ExtendStepInfo
from sim.slip_nn_features import FEATURE_DIM

# Must match KetchupExtendRLEnv observation layout.
PPO_OBS_DIM = FEATURE_DIM + 1 + 3 + 1 + 1 + 1  # 33


def build_ppo_obs_from_info(info: ExtendStepInfo) -> np.ndarray:
    """Build the same 33-D vector used by ``KetchupExtendRLEnv._obs``."""
    if info.features is None:
        feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    else:
        feat = np.asarray(info.features, dtype=np.float32).reshape(-1)
        if feat.shape[0] != FEATURE_DIM:
            raise ValueError(f"features dim {feat.shape[0]} != {FEATURE_DIM}")
    t_frac = float(info.step_i / max(1, info.n_extend - 1))
    extra = np.array(
        [
            float(info.grip_extra),
            float(info.wrist_cmd[0]),
            float(info.wrist_cmd[1]),
            float(info.wrist_cmd[2]),
            t_frac,
            float(info.friction_scale),
            float(info.mass_scale),
        ],
        dtype=np.float32,
    )
    return np.concatenate([feat, extra], axis=0)


def policy2_from_ppo_action(
    action: np.ndarray,
    *,
    g_max: float = 0.25,
    d_max: float = 0.25,
) -> Policy2Action:
    """Map PPO Box[-1,1]^4 → Policy2Action (same as Gym env)."""
    a = np.asarray(action, dtype=np.float64).reshape(4)
    a = np.clip(a, -1.0, 1.0)
    grip = float((a[0] + 1.0) * 0.5 * g_max)
    wrist = tuple(float(x) * d_max for x in a[1:])
    return Policy2Action(grip=grip, wrist_delta=wrist)


@dataclass
class PPOExtendHook:
    """``ExtendActionHook`` backed by a Stable-Baselines3 PPO policy."""

    model: Any  # stable_baselines3.PPO
    g_max: float = 0.25
    d_max: float = 0.25
    apply_actions: bool = True
    deterministic: bool = True
    max_records: int = 5000
    records: list[dict] = field(default_factory=list)

    @classmethod
    def from_zip(
        cls,
        path: Path | str,
        *,
        g_max: float = 0.25,
        d_max: float = 0.25,
        apply_actions: bool = True,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> "PPOExtendHook":
        from stable_baselines3 import PPO

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"PPO zip missing: {p}")
        model = PPO.load(str(p), device=device)
        return cls(
            model=model,
            g_max=g_max,
            d_max=d_max,
            apply_actions=apply_actions,
            deterministic=deterministic,
        )

    def reset_records(self) -> None:
        self.records.clear()

    def __call__(self, info: ExtendStepInfo) -> ExtendStepDecision:
        obs = build_ppo_obs_from_info(info)
        if obs.shape[0] != PPO_OBS_DIM:
            raise ValueError(f"obs dim {obs.shape[0]} != {PPO_OBS_DIM}")
        raw, _ = self.model.predict(obs, deterministic=self.deterministic)
        action = policy2_from_ppo_action(raw, g_max=self.g_max, d_max=self.d_max)
        if len(self.records) < self.max_records:
            self.records.append(
                {
                    "step_i": info.step_i,
                    "p_slip": info.p_slip,
                    "soft_fire": info.soft_fire,
                    "slip_active": info.slip_active,
                    "apply": self.apply_actions,
                    "raw": [float(x) for x in np.asarray(raw).reshape(-1)],
                    "grip": float(action.grip),
                    "wrist": list(action.wrist_delta),
                    "dz_cm": (info.object_z - info.z_extend_start) * 100.0,
                    "case_name": info.case_name,
                }
            )
        if not self.apply_actions:
            return ExtendStepDecision(action=None, note="ppo_propose_only")
        return ExtendStepDecision(action=action, note="ppo")
