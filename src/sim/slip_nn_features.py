"""NN-0 feature vector builder (DS-SLIP-NN-001 §3)."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from sim.slip_center_detect import CenterDivergenceDetector, world_to_hand
from sim.slip_vertical_support import (
    VerticalSupportAntislipDetector,
    VerticalSupportWindow,
    gravity_up,
    measure_vertical_support,
)


FEATURE_NAMES: tuple[str, ...] = (
    # Scheme 1 — geometry (8)
    "n_contacts",
    "sep",
    "sep_delta",
    "v_div_x",
    "v_div_y",
    "v_div_z",
    "sep_rate",
    "contact_force_norm",
    # Scheme 2 — mechanics (10)
    "S_raw",
    "S_smooth",
    "S_avg",
    "S_ratio",
    "S_peak_ratio",
    "S_n",
    "S_t",
    "S_integral_0.5s",
    "mg_ratio",
    "slip_rule_s2",
    # Proprioception / context (8)
    "object_dz_extend",
    "object_dz_traj_end",
    "object_dz_start",
    "wrist_tz",
    "wrist_tz_rate",
    "grip_extra",
    "phase_extend",
    "friction_scale",
)

FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class SlipFeatureLabels:
    y_scheme1: bool
    y_scheme2: bool
    y_gt: bool
    y_fused: bool
    grip_extra: float
    slip_speed_m_s: float


@dataclass
class SlipFeatureReading:
    features: np.ndarray
    labels: SlipFeatureLabels


@dataclass
class _StepContext:
    phase: str
    wrist_tz: float
    grip_extra: float
    friction_scale: float
    object_z: float
    object_z_traj_end: float
    object_z_extend_start: float | None
    object_z_start: float
    in_trajectory: bool = False


class SlipFeatureBuilder:
    """Stateful per-step feature + teacher label builder for NN-0."""

    def __init__(
        self,
        *,
        sim_dt: float = 0.01,
        avg_window_s: float = 2.0,
        smooth_window_s: float = 0.2,
        integral_window_s: float = 0.5,
        slip_ratio: float = 0.7,
        peak_slip_ratio: float = 0.95,
        min_peak_support: float = 100.0,
        separation_threshold_m: float = 0.008,
        motion_threshold_m_s: float = 0.015,
        hand_body: str = "right_hand_link",
        object_body: str = "right_object",
        kinematic_epsilon_m_s: float = 0.02,
    ):
        self.sim_dt = sim_dt
        self.hand_body = hand_body
        self.object_body = object_body
        self.kinematic_epsilon_m_s = kinematic_epsilon_m_s

        self._center = CenterDivergenceDetector(
            separation_threshold_m=separation_threshold_m,
            motion_threshold_m_s=motion_threshold_m_s,
            hand_body=hand_body,
            sim_dt=sim_dt,
        )
        self._support = VerticalSupportAntislipDetector(
            avg_window_s,
            sim_dt,
            smooth_window_s=smooth_window_s,
            slip_ratio=slip_ratio,
            peak_slip_ratio=peak_slip_ratio,
            min_peak_support=min_peak_support,
        )
        self._integral = VerticalSupportWindow(integral_window_s, sim_dt)

        self._prev_sep: float | None = None
        self._prev_wrist_tz: float | None = None
        self._prev_obj_h: np.ndarray | None = None
        self._prev_contact_h: np.ndarray | None = None
        self._object_z_start = 0.0
        self._object_z_traj_end = 0.0
        self._object_z_extend_start: float | None = None

    def reset_trajectory(self, object_z_start: float) -> None:
        self._center.reset()
        self._support.reset_peak()
        self._integral.reset()
        self._prev_sep = None
        self._prev_wrist_tz = None
        self._prev_obj_h = None
        self._prev_contact_h = None
        self._object_z_start = object_z_start
        self._object_z_traj_end = object_z_start
        self._object_z_extend_start = None

    def mark_trajectory_end(self, object_z: float) -> None:
        self._object_z_traj_end = object_z

    def reset_extend(self, object_z: float) -> None:
        self._center.reset()
        self._support.reset_peak()
        self._integral.reset()
        self._prev_sep = None
        self._prev_obj_h = None
        self._prev_contact_h = None
        self._object_z_extend_start = object_z

    def build(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        hand_geoms: set[int],
        object_geoms: set[int],
        object_id: int,
        ctx: _StepContext,
    ) -> SlipFeatureReading:
        from sim.slip_kinematic_gt import compute_slip_gt

        g_hat = gravity_up(model)
        center = self._center.update(model, data, hand_geoms, object_geoms, object_id)
        vs = measure_vertical_support(
            model, data, hand_geoms, object_geoms, self.object_body, g_hat=g_hat
        )
        support = self._support.update(vs.support_z)
        s_integral = self._integral.push(vs.support_z)

        v_div = np.zeros(3, dtype=float)
        sep_rate = 0.0
        prev_obj_h = self._prev_obj_h
        prev_contact_h = self._prev_contact_h
        if not np.isnan(center.separation_m):
            obj_h = world_to_hand(model, data, center.object_center_world, self.hand_body)
            if center.n_contacts > 0:
                contact_h = world_to_hand(
                    model, data, center.contact_center_world, self.hand_body
                )
                if prev_obj_h is not None and prev_contact_h is not None:
                    v_obj = (obj_h - prev_obj_h) / self.sim_dt
                    v_contact = (contact_h - prev_contact_h) / self.sim_dt
                    v_div = v_obj - v_contact
                self._prev_obj_h = obj_h.copy()
                self._prev_contact_h = contact_h.copy()

            if self._prev_sep is not None:
                sep_rate = (center.separation_m - self._prev_sep) / self.sim_dt
            self._prev_sep = center.separation_m

        kin = compute_slip_gt(
            model,
            data,
            object_id,
            hand_geoms,
            object_geoms,
            hand_body=self.hand_body,
            sim_dt=self.sim_dt,
            epsilon_m_s=self.kinematic_epsilon_m_s,
            prev_obj_h=prev_obj_h,
            prev_contact_h=prev_contact_h,
        )

        contact_force_norm = 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if not (
                (g1 in hand_geoms or g2 in hand_geoms)
                and (g1 in object_geoms or g2 in object_geoms)
            ):
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            frame = np.array(contact.frame, dtype=float).reshape(3, 3)
            contact_force_norm += float(np.linalg.norm(frame.T @ wrench[:3]))

        peak_ratio = (
            support.support_smooth / support.peak_smooth
            if support.peak_smooth > 1e-6
            else 1.0
        )

        wrist_tz_rate = 0.0
        if self._prev_wrist_tz is not None:
            wrist_tz_rate = (ctx.wrist_tz - self._prev_wrist_tz) / self.sim_dt
        self._prev_wrist_tz = ctx.wrist_tz

        dz_extend = 0.0
        if self._object_z_extend_start is not None:
            dz_extend = ctx.object_z - self._object_z_extend_start

        # During trajectory the end height is unknown; zero until trajectory completes.
        dz_traj_end = 0.0 if ctx.in_trajectory else ctx.object_z - self._object_z_traj_end

        phase_extend = 1.0 if ctx.phase.startswith("extend") else 0.0
        y_fused = center.slip or support.slip_active

        features = np.array(
            [
                float(center.n_contacts),
                float(center.separation_m) if not np.isnan(center.separation_m) else 0.0,
                float(center.separation_delta_m),
                float(v_div[0]),
                float(v_div[1]),
                float(v_div[2]),
                float(sep_rate),
                contact_force_norm,
                float(vs.support_z),
                float(support.support_smooth),
                float(support.support_avg),
                float(support.ratio_to_avg),
                float(peak_ratio),
                float(vs.support_normal_z),
                float(vs.support_tangent_z),
                float(s_integral),
                float(vs.support_ratio) if not np.isnan(vs.support_ratio) else 0.0,
                float(support.slip_active),
                float(dz_extend),
                float(dz_traj_end),
                float(ctx.object_z - self._object_z_start),
                float(ctx.wrist_tz),
                float(wrist_tz_rate),
                float(ctx.grip_extra),
                phase_extend,
                float(ctx.friction_scale),
            ],
            dtype=np.float32,
        )
        assert features.shape == (FEATURE_DIM,)

        labels = SlipFeatureLabels(
            y_scheme1=center.slip,
            y_scheme2=support.slip_active,
            y_gt=kin.slip,
            y_fused=y_fused,
            grip_extra=ctx.grip_extra,
            slip_speed_m_s=kin.slip_speed_m_s,
        )
        return SlipFeatureReading(features=features, labels=labels)


def make_step_context(
    *,
    phase: str,
    wrist_tz: float,
    grip_extra: float,
    friction_scale: float,
    object_z: float,
    object_z_traj_end: float,
    object_z_extend_start: float | None,
    object_z_start: float,
    in_trajectory: bool = False,
) -> _StepContext:
    return _StepContext(
        phase=phase,
        wrist_tz=wrist_tz,
        grip_extra=grip_extra,
        friction_scale=friction_scale,
        object_z=object_z,
        object_z_traj_end=object_z_traj_end,
        object_z_extend_start=object_z_extend_start,
        object_z_start=object_z_start,
        in_trajectory=in_trajectory,
    )
