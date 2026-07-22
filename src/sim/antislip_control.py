"""Anti-slip response: increase finger closure when center-divergence slip is detected."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# XHAND right-hand actuators: 0–5 arm, 6–17 fingers (see scene.xml).
FINGER_ACTUATOR_INDICES: tuple[int, ...] = tuple(range(6, 18))
WRIST_ACTUATOR_INDICES: tuple[int, ...] = (3, 4, 5)  # roll, pitch, yaw


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


def apply_wrist_residual(
    ctrl: np.ndarray,
    model: mujoco.MjModel,
    wrist_delta: np.ndarray,
    *,
    wrist_indices: tuple[int, ...] = WRIST_ACTUATOR_INDICES,
) -> np.ndarray:
    """Add wrist residual to roll/pitch/yaw actuators, clipped to ctrlrange."""
    out = ctrl.copy()
    delta = np.asarray(wrist_delta, dtype=np.float64).reshape(-1)
    for i, aid in enumerate(wrist_indices):
        if aid >= model.nu or i >= delta.shape[0]:
            continue
        lo, hi = model.actuator_ctrlrange[aid]
        out[aid] = float(np.clip(out[aid] + float(delta[i]), lo, hi))
    return out


@dataclass
class Policy2Action:
    """P2-A open-loop action: target grip + wrist residual (rad)."""

    grip: float = 0.0
    wrist_delta: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def as_vector(self) -> np.ndarray:
        return np.array([self.grip, *self.wrist_delta], dtype=np.float64)

    @classmethod
    def from_vector(cls, v: np.ndarray | list[float]) -> "Policy2Action":
        a = np.asarray(v, dtype=np.float64).reshape(-1)
        if a.shape[0] != 4:
            raise ValueError(f"Policy2Action expects 4-D, got {a.shape}")
        return cls(grip=float(a[0]), wrist_delta=(float(a[1]), float(a[2]), float(a[3])))


class Policy2OpenLoopController:
    """Rate-limited open-loop executor for Policy-2 teacher search / replay.

    Ramps ``grip_extra`` toward ``action.grip`` (ratchet-up only by default) and
    ``wrist_cmd`` toward ``action.wrist_delta``, then applies both on ``ctrl_ref``.
    """

    def __init__(
        self,
        action: Policy2Action,
        *,
        g_max: float = 0.25,
        d_max: float = 0.25,
        rate_g: float = 0.02,
        rate_w: float = 0.02,
        grip_ratchet: bool = True,
    ):
        self.g_max = float(g_max)
        self.d_max = float(d_max)
        self.rate_g = float(rate_g)
        self.rate_w = float(rate_w)
        self.grip_ratchet = bool(grip_ratchet)
        self.set_action(action)
        self.grip_extra = 0.0
        self.wrist_cmd = np.zeros(3, dtype=np.float64)

    def set_action(self, action: Policy2Action) -> None:
        g = float(np.clip(action.grip, 0.0, self.g_max))
        w = np.clip(np.asarray(action.wrist_delta, dtype=np.float64), -self.d_max, self.d_max)
        self.target_grip = g
        self.target_wrist = w.reshape(3).copy()

    def reset(self) -> None:
        self.grip_extra = 0.0
        self.wrist_cmd[:] = 0.0

    def _step_toward(self, cur: float, tgt: float, rate: float) -> float:
        if abs(tgt - cur) <= rate:
            return tgt
        return cur + rate if tgt > cur else cur - rate

    def apply(self, ctrl: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
        if self.grip_ratchet:
            # Only increase grip toward target (never release via this path).
            desire = max(self.grip_extra, self.target_grip)
            self.grip_extra = self._step_toward(self.grip_extra, desire, self.rate_g)
            self.grip_extra = min(self.g_max, self.grip_extra)
        else:
            self.grip_extra = self._step_toward(self.grip_extra, self.target_grip, self.rate_g)
            self.grip_extra = float(np.clip(self.grip_extra, 0.0, self.g_max))
        for i in range(3):
            self.wrist_cmd[i] = self._step_toward(
                float(self.wrist_cmd[i]), float(self.target_wrist[i]), self.rate_w
            )
            self.wrist_cmd[i] = float(np.clip(self.wrist_cmd[i], -self.d_max, self.d_max))
        out = apply_wrist_residual(ctrl, model, self.wrist_cmd)
        if self.grip_extra > 0:
            out = apply_grip_boost(out, model, self.grip_extra)
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

    def set_grip(self, value: float) -> None:
        """Ratchet grip_extra up to ``value`` (used by NN Δgrip head)."""
        self.grip_extra = min(self.max_extra, max(self.grip_extra, float(value)))

    def on_no_slip(self) -> None:
        if self.decay > 0:
            self.grip_extra = max(0.0, self.grip_extra - self.decay)

    def apply(self, ctrl: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
        if self.grip_extra <= 0:
            return ctrl
        return apply_grip_boost(ctrl, model, self.grip_extra)
