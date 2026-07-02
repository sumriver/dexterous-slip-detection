"""Scheme-1 slip detection: contact-center vs object-center divergence in hand frame.

Uses only quantities available on real hardware:
  - contact positions + forces (tactile / mj_contact)
  - object center pose (point cloud on robot; body xpos in sim)
  - hand frame from proprioception (FK)
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


def _extract_hand_object_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geoms: set[int],
    object_geoms: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    forces: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        hand_hit = g1 in hand_geoms or g2 in hand_geoms
        obj_hit = g1 in object_geoms or g2 in object_geoms
        if not (hand_hit and obj_hit):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        frame = np.array(contact.frame, dtype=float).reshape(3, 3)
        forces.append(frame @ wrench[:3])
        positions.append(contact.pos.copy())
    if not forces:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.array(forces), np.array(positions)


@dataclass
class CenterSlipReading:
    slip: bool
    n_contacts: int
    separation_m: float
    separation_delta_m: float
    motion_divergence_m_s: float
    contact_center_world: np.ndarray
    object_center_world: np.ndarray


def world_to_hand(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    point_world: np.ndarray,
    hand_body: str = "right_hand_link",
) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
    if bid < 0:
        return point_world.copy()
    hand_pos = data.xpos[bid]
    hand_rot = data.xmat[bid].reshape(3, 3)
    return hand_rot.T @ (point_world - hand_pos)


def contact_center_weighted(forces: np.ndarray, positions: np.ndarray) -> np.ndarray | None:
    """Force-magnitude weighted contact centroid (world frame)."""
    if len(forces) == 0:
        return None
    weights = np.linalg.norm(forces, axis=1)
    weights = np.maximum(weights, 1e-6)
    return np.average(positions, axis=0, weights=weights)


class CenterDivergenceDetector:
    """Detect slip when object center and contact center diverge in the hand frame."""

    def __init__(
        self,
        *,
        separation_threshold_m: float = 0.008,
        motion_threshold_m_s: float = 0.015,
        min_contacts: int = 2,
        hand_body: str = "right_hand_link",
        sim_dt: float = 0.01,
    ):
        self.separation_threshold_m = separation_threshold_m
        self.motion_threshold_m_s = motion_threshold_m_s
        self.min_contacts = min_contacts
        self.hand_body = hand_body
        self.sim_dt = sim_dt
        self._sep_ref: float | None = None
        self._prev_obj_h: np.ndarray | None = None
        self._prev_contact_h: np.ndarray | None = None

    def reset(self) -> None:
        self._sep_ref = None
        self._prev_obj_h = None
        self._prev_contact_h = None

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        hand_geoms: set[int],
        object_geoms: set[int],
        object_id: int,
    ) -> CenterSlipReading:
        forces, positions = _extract_hand_object_contacts(model, data, hand_geoms, object_geoms)
        n_con = len(forces)
        obj_w = data.xpos[object_id].copy()
        contact_w = contact_center_weighted(forces, positions)

        if contact_w is None or n_con < self.min_contacts:
            return CenterSlipReading(
                slip=False,
                n_contacts=n_con,
                separation_m=float("nan"),
                separation_delta_m=0.0,
                motion_divergence_m_s=0.0,
                contact_center_world=np.zeros(3),
                object_center_world=obj_w,
            )

        obj_h = world_to_hand(model, data, obj_w, self.hand_body)
        contact_h = world_to_hand(model, data, contact_w, self.hand_body)
        separation = float(np.linalg.norm(obj_h - contact_h))

        if self._sep_ref is None:
            self._sep_ref = separation
            self._prev_obj_h = obj_h.copy()
            self._prev_contact_h = contact_h.copy()
            return CenterSlipReading(
                slip=False,
                n_contacts=n_con,
                separation_m=separation,
                separation_delta_m=0.0,
                motion_divergence_m_s=0.0,
                contact_center_world=contact_w,
                object_center_world=obj_w,
            )

        sep_delta = abs(separation - self._sep_ref)
        motion_div = 0.0
        if self._prev_obj_h is not None and self._prev_contact_h is not None:
            v_obj = (obj_h - self._prev_obj_h) / self.sim_dt
            v_contact = (contact_h - self._prev_contact_h) / self.sim_dt
            motion_div = float(np.linalg.norm(v_obj - v_contact))

        self._prev_obj_h = obj_h.copy()
        self._prev_contact_h = contact_h.copy()

        slip = (
            sep_delta >= self.separation_threshold_m
            or motion_div >= self.motion_threshold_m_s
        )

        return CenterSlipReading(
            slip=slip,
            n_contacts=n_con,
            separation_m=separation,
            separation_delta_m=sep_delta,
            motion_divergence_m_s=motion_div,
            contact_center_world=contact_w,
            object_center_world=obj_w,
        )
