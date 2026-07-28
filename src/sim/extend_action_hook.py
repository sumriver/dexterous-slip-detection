"""Optional per-step extend action hook for closed-loop PPO / inspect.

Default is off: ``replay_spider_task`` behavior unchanged when hook is None
or ``mode="off"``.

Modes (used in later steps):
  - off:        ignore hook
  - always:     ask hook every extend step (Gym-like)
  - on_detect:  ask hook only when slip soft/hard fires (BC-parity closed-loop)
  - inspect:    ask hook every step for logging; apply only if it returns an action
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from sim.antislip_control import Policy2Action

ExtendHookMode = Literal["off", "always", "on_detect", "inspect"]


@dataclass
class ExtendStepInfo:
    """Read-only snapshot offered to the hook each extend step."""

    step_i: int
    n_extend: int
    features: np.ndarray | None
    p_slip: float | None
    slip_now: bool
    slip_active: bool
    soft_fire: bool
    grip_extra: float
    wrist_cmd: tuple[float, float, float]
    object_z: float
    z_extend_start: float
    friction_scale: float
    mass_scale: float
    case_name: str = ""


@dataclass
class ExtendStepDecision:
    """Hook output. ``action is None`` → do not override controllers this step."""

    action: Policy2Action | None = None
    note: str = ""


# Hook contract: inspect info → optional action override.
ExtendActionHook = Callable[[ExtendStepInfo], ExtendStepDecision]


@dataclass
class RecordingExtendHook:
    """Inspect helper: records every call; never overrides action (safe default)."""

    records: list[dict] = field(default_factory=list)
    max_records: int = 5000

    def __call__(self, info: ExtendStepInfo) -> ExtendStepDecision:
        if len(self.records) < self.max_records:
            self.records.append(
                {
                    "step_i": info.step_i,
                    "p_slip": info.p_slip,
                    "slip_now": info.slip_now,
                    "slip_active": info.slip_active,
                    "soft_fire": info.soft_fire,
                    "grip_extra": info.grip_extra,
                    "wrist_cmd": list(info.wrist_cmd),
                    "object_z": info.object_z,
                    "dz_cm": (info.object_z - info.z_extend_start) * 100.0,
                    "friction_scale": info.friction_scale,
                    "mass_scale": info.mass_scale,
                    "case_name": info.case_name,
                }
            )
        return ExtendStepDecision(action=None, note="record_only")


def should_query_hook(
    mode: ExtendHookMode,
    *,
    soft_fire: bool,
    slip_active: bool,
) -> bool:
    if mode == "off":
        return False
    if mode in ("always", "inspect"):
        return True
    if mode == "on_detect":
        return bool(soft_fire or slip_active)
    return False
