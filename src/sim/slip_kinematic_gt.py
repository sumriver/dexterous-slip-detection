"""Kinematic slip ground truth for NN-0 (simulation-only labels).

Slip speed is ‖v_obj^H − v_contact^H‖ when contacts exist (object motion relative to
force-weighted contact centroid in the palm frame). Falls back to object-vs-hand
linear velocity when contacts are absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from sim.slip_center_detect import contact_center_weighted, world_to_hand


@dataclass(frozen=True)
class KinematicSlipReading:
    slip: bool
    slip_speed_m_s: float
    rel_vel_hand: np.ndarray


def compute_slip_gt(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_id: int,
    hand_geoms: set[int],
    object_geoms: set[int],
    *,
    hand_body: str = "right_hand_link",
    sim_dt: float = 0.01,
    epsilon_m_s: float = 0.02,
    prev_obj_h: np.ndarray | None = None,
    prev_contact_h: np.ndarray | None = None,
) -> KinematicSlipReading:
    """True when object moves relative to contact reference faster than epsilon."""
    obj_w = data.xpos[object_id].copy()
    obj_h = world_to_hand(model, data, obj_w, hand_body)

    forces: list[np.ndarray] = []
    positions: list[np.ndarray] = []
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
        forces.append(frame.T @ wrench[:3])
        positions.append(contact.pos.copy())

    rel_vel_hand = np.zeros(3, dtype=float)
    if (
        forces
        and prev_obj_h is not None
        and prev_contact_h is not None
        and len(forces) >= 1
    ):
        contact_w = contact_center_weighted(np.array(forces), np.array(positions))
        if contact_w is not None:
            contact_h = world_to_hand(model, data, contact_w, hand_body)
            v_obj = (obj_h - prev_obj_h) / sim_dt
            v_contact = (contact_h - prev_contact_h) / sim_dt
            rel_vel_hand = v_obj - v_contact
    else:
        obj_vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, object_id, obj_vel, 0)
        hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
        if hand_bid >= 0:
            hand_vel = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, hand_bid, hand_vel, 0
            )
            hand_rot = data.xmat[hand_bid].reshape(3, 3)
            rel_vel_hand = hand_rot.T @ (obj_vel[:3] - hand_vel[:3])
        else:
            rel_vel_hand = obj_vel[:3].copy()

    slip_speed = float(np.linalg.norm(rel_vel_hand))
    return KinematicSlipReading(
        slip=slip_speed > epsilon_m_s,
        slip_speed_m_s=slip_speed,
        rel_vel_hand=rel_vel_hand,
    )
