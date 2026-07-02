"""Anti-slip response: increase finger closure when center-divergence slip is detected."""

from __future__ import annotations

import mujoco
import numpy as np

# XHAND right-hand actuators: 0–5 arm, 6–17 fingers (see scene.xml).
FINGER_ACTUATOR_INDICES: tuple[int, ...] = tuple(range(6, 18))


def apply_grip_boost(
    ctrl: np.ndarray,
    model: mujoco.MjModel,
    grip_extra: float,
    *,
    finger_indices: tuple[int, ...] = FINGER_ACTUATOR_INDICES,
) -> np.ndarray:
    """Add grip_extra to finger actuators (closing direction = +), clipped to ctrlrange."""
    out = ctrl.copy()
    for aid in finger_indices:
        if aid >= model.nu:
            continue
        lo, hi = model.actuator_ctrlrange[aid]
        out[aid] = float(np.clip(out[aid] + grip_extra, lo, hi))
    return out


class GripBoostController:
    """Ramp finger closure while slip is active during extend phase."""

    def __init__(
        self,
        *,
        step_boost: float = 0.015,
        max_extra: float = 0.25,
        decay: float = 0.0,
    ):
        self.step_boost = step_boost
        self.max_extra = max_extra
        self.decay = decay
        self.grip_extra = 0.0

    def reset(self) -> None:
        self.grip_extra = 0.0

    def on_slip(self) -> None:
        self.grip_extra = min(self.max_extra, self.grip_extra + self.step_boost)

    def on_no_slip(self) -> None:
        if self.decay > 0:
            self.grip_extra = max(0.0, self.grip_extra - self.decay)

    def apply(self, ctrl: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
        if self.grip_extra <= 0:
            return ctrl
        return apply_grip_boost(ctrl, model, self.grip_extra)
