"""Scheme-2 slip signals: vertical support force and sliding-window integral.

Uses only contact forces + world gravity direction (real-robot compatible).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class VerticalSupportReading:
    n_contacts: int
    support_z: float          # Σ f_on_obj · ĝ  [N]
    support_normal_z: float   # Σ (f_n · ĝ)
    support_tangent_z: float  # Σ (f_t · ĝ)
    support_ratio: float      # support_z / (m·g)
    mg: float


def gravity_up(model: mujoco.MjModel) -> np.ndarray:
    g = np.asarray(model.opt.gravity, dtype=float)
    if np.linalg.norm(g) < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return -g / np.linalg.norm(g)


def decompose_contact_force_on_object(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact_index: int,
    object_geom_ids: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f_on_obj, f_normal, f_tangent) in world frame.

    MuJoCo stores contact.frame row-major with rows = contact axes expressed
    in world (row 0 = normal). A vector given in contact coordinates therefore
    maps to world via ``frame.T @ v`` (columns of frame.T are the axes).
    """
    contact = data.contact[contact_index]
    wrench = np.zeros(6)
    mujoco.mj_contactForce(model, data, contact_index, wrench)
    frame = np.array(contact.frame, dtype=float).reshape(3, 3)
    f_local = wrench[:3]
    f_world = frame.T @ f_local
    n_hat = frame[0, :]
    f_normal = f_local[0] * n_hat
    f_tangent = f_world - f_normal

    g1, g2 = contact.geom1, contact.geom2
    if g1 in object_geom_ids:
        f_world = -f_world
        f_normal = -f_normal
        f_tangent = -f_tangent
    return f_world, f_normal, f_tangent


def measure_vertical_support(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geom_ids: set[int],
    object_geom_ids: set[int],
    object_body: str = "right_object",
    g_hat: np.ndarray | None = None,
) -> VerticalSupportReading:
    """Sum hand→object contact forces projected onto vertical (anti-gravity)."""
    g_hat = g_hat if g_hat is not None else gravity_up(model)
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    mg = float(model.body_mass[oid] * np.linalg.norm(model.opt.gravity))

    support_z = 0.0
    support_nz = 0.0
    support_tz = 0.0
    n_con = 0
    f_sum = np.zeros(3)

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        hand_hit = g1 in hand_geom_ids or g2 in hand_geom_ids
        obj_hit = g1 in object_geom_ids or g2 in object_geom_ids
        if not (hand_hit and obj_hit):
            continue

        f_w, f_n, f_t = decompose_contact_force_on_object(model, data, i, object_geom_ids)
        f_sum += f_w
        # Upward-only contributions (anti-gravity support capacity)
        support_z += max(0.0, float(np.dot(f_w, g_hat)))
        support_nz += max(0.0, float(np.dot(f_n, g_hat)))
        support_tz += max(0.0, float(np.dot(f_t, g_hat)))
        n_con += 1

    # Net upward support (matches grasp_validate.support_z convention)
    support_net_z = float(max(0.0, f_sum[2]))
    if support_net_z > support_z:
        support_z = support_net_z

    ratio = support_z / mg if mg > 1e-9 else float("nan")
    return VerticalSupportReading(
        n_contacts=n_con,
        support_z=support_z,
        support_normal_z=support_nz,
        support_tangent_z=support_tz,
        support_ratio=ratio,
        mg=mg,
    )


class VerticalSupportWindow:
    """Sliding-window integral of vertical support S(t)."""

    def __init__(self, window_s: float, sim_dt: float):
        self.window_steps = max(1, int(round(window_s / sim_dt)))
        self.sim_dt = sim_dt
        self._buf: list[float] = []

    def reset(self) -> None:
        self._buf.clear()

    def push(self, support_z: float) -> float:
        self._buf.append(support_z * self.sim_dt)
        if len(self._buf) > self.window_steps:
            self._buf.pop(0)
        return float(sum(self._buf))

    @property
    def integral(self) -> float:
        return float(sum(self._buf))


@dataclass
class SupportAvgReading:
    """Instant support vs trailing moving average."""

    support_z: float          # raw instantaneous S
    support_smooth: float     # short-window smoothed S
    support_avg: float        # long-window mean of S_smooth
    ratio_to_avg: float       # S_smooth / S_avg
    slip: bool
    n_samples: int


class VerticalSupportMovingAverage:
    """Slip when short-window smoothed S drops below fraction of long-window S_avg.

    Pipeline: S_raw → S_smooth (e.g. 200ms MA) → S_avg (e.g. 2s MA of S_smooth).
    Replaces noisy single-point S_0 baseline.
    """

    def __init__(
        self,
        window_s: float,
        sim_dt: float,
        *,
        smooth_window_s: float = 0.2,
        slip_ratio: float = 0.7,
        min_samples: int | None = None,
    ):
        self.avg_window_steps = max(1, int(round(window_s / sim_dt)))
        self.smooth_window_steps = max(1, int(round(smooth_window_s / sim_dt)))
        self.slip_ratio = slip_ratio
        self.min_samples = (
            min_samples if min_samples is not None else max(5, self.avg_window_steps // 5)
        )
        self._smooth_buf: list[float] = []
        self._avg_buf: list[float] = []

    def reset(self) -> None:
        self._smooth_buf.clear()
        self._avg_buf.clear()

    def update(self, support_z: float) -> SupportAvgReading:
        self._smooth_buf.append(support_z)
        if len(self._smooth_buf) > self.smooth_window_steps:
            self._smooth_buf.pop(0)
        s_smooth = float(np.mean(self._smooth_buf))

        self._avg_buf.append(s_smooth)
        if len(self._avg_buf) > self.avg_window_steps:
            self._avg_buf.pop(0)

        s_avg = float(np.mean(self._avg_buf))
        n = len(self._avg_buf)

        if n < self.min_samples or s_avg < 1e-6:
            return SupportAvgReading(
                support_z=support_z,
                support_smooth=s_smooth,
                support_avg=s_avg,
                ratio_to_avg=1.0,
                slip=False,
                n_samples=n,
            )

        ratio = s_smooth / s_avg
        slip = ratio < self.slip_ratio
        return SupportAvgReading(
            support_z=support_z,
            support_smooth=s_smooth,
            support_avg=s_avg,
            ratio_to_avg=ratio,
            slip=slip,
            n_samples=n,
        )


@dataclass
class AntislipSupportReading:
    """Support reading for closed-loop anti-slip (extend phase)."""

    support_z: float
    support_smooth: float
    support_avg: float
    ratio_to_avg: float
    peak_smooth: float
    slip_now: bool          # instantaneous slip (ratio or peak drop)
    slip_active: bool       # latched: keep boosting grip for rest of extend
    n_samples: int


class VerticalSupportAntislipDetector:
    """Scheme-2 closed-loop slip detector for extend phase.

    Combines S_smooth/S_avg with peak-tracking on S_smooth during extend.
    Once slip fires, latch stays active so brief force recoveries do not
    stop grip ramping (real slip onset is earlier than S/S_avg collapse).
    """

    def __init__(
        self,
        window_s: float,
        sim_dt: float,
        *,
        smooth_window_s: float = 0.2,
        slip_ratio: float = 0.7,
        peak_slip_ratio: float = 0.95,
        min_peak_support: float = 100.0,
        min_samples: int | None = None,
    ):
        self._avg = VerticalSupportMovingAverage(
            window_s,
            sim_dt,
            smooth_window_s=smooth_window_s,
            slip_ratio=slip_ratio,
            min_samples=min_samples,
        )
        self.peak_slip_ratio = peak_slip_ratio
        self.min_peak_support = min_peak_support
        self._peak_smooth = 0.0
        self._latched = False

    def reset_peak(self) -> None:
        self._peak_smooth = 0.0
        self._latched = False

    def update(self, support_z: float) -> AntislipSupportReading:
        reading = self._avg.update(support_z)
        self._peak_smooth = max(self._peak_smooth, reading.support_smooth)

        peak_slip = (
            self._peak_smooth >= self.min_peak_support
            and reading.support_smooth < self.peak_slip_ratio * self._peak_smooth
        )
        slip_now = reading.slip or peak_slip
        if slip_now:
            self._latched = True

        return AntislipSupportReading(
            support_z=reading.support_z,
            support_smooth=reading.support_smooth,
            support_avg=reading.support_avg,
            ratio_to_avg=reading.ratio_to_avg,
            peak_smooth=self._peak_smooth,
            slip_now=slip_now,
            slip_active=self._latched,
            n_samples=reading.n_samples,
        )
