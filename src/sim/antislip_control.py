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


class NormalTangentGripController:
    """Modulate grip from the horizontal normal/tangential net-force imbalance.

    Hypothesis (horizontal anti-slip): in a stable grasp the horizontal net
    normal force and net tangential (friction) force are equal and opposite.
    When the normal net exceeds the tangential net (|F_n| > |F_t|), friction
    can no longer balance the inward normal push, producing a net force that
    tends to eject the object. This controller then REDUCES grip (releases
    finger closure) to shrink the un-cancelled normal push; when the balance
    is restored it slowly re-grips.

    grip_delta is added to the finger actuators (negative => release).
    """

    def __init__(
        self,
        *,
        release_step: float = 0.01,
        restore_step: float = 0.003,
        max_release: float = 0.15,
        trigger_ratio: float = 1.05,
        min_normal: float = 3.0,
    ):
        self.release_step = release_step
        self.restore_step = restore_step
        self.max_release = max_release
        self.trigger_ratio = trigger_ratio
        self.min_normal = min_normal
        self.grip_delta = 0.0
        self.releasing = False

    def reset(self) -> None:
        self.grip_delta = 0.0
        self.releasing = False

    def update(self, fn_mag: float, ft_mag: float) -> float:
        """Update grip_delta from horizontal normal/tangential magnitudes."""
        if fn_mag >= self.min_normal and fn_mag > self.trigger_ratio * ft_mag:
            self.grip_delta = max(-self.max_release, self.grip_delta - self.release_step)
            self.releasing = True
        else:
            self.grip_delta = min(0.0, self.grip_delta + self.restore_step)
            self.releasing = False
        return self.grip_delta

    def apply(self, ctrl: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
        if self.grip_delta == 0.0:
            return ctrl
        return apply_grip_boost(ctrl, model, self.grip_delta)


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
