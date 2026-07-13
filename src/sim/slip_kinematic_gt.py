"""Kinematic slip ground truth for NN-0 (simulation-only labels).

Uses object velocity relative to the hand frame — not available on real hardware,
but useful as a training label mixed with teacher labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from sim.slip_center_detect import world_to_hand


@dataclass(frozen=True)
class KinematicSlipReading:
    slip: bool
    slip_speed_m_s: float
    rel_vel_hand: np.ndarray


def compute_slip_gt(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_id: int,
    *,
    hand_body: str = "right_hand_link",
    epsilon_m_s: float = 0.02,
) -> KinematicSlipReading:
    """True when object center moves relative to hand faster than ``epsilon_m_s``."""
    obj_vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, object_id, obj_vel, 0)
    obj_v_world = obj_vel[:3].copy()

    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
    if hand_bid < 0:
        rel_vel_hand = obj_v_world
    else:
        hand_vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, hand_bid, hand_vel, 0)
        hand_v_world = hand_vel[:3]
        hand_rot = data.xmat[hand_bid].reshape(3, 3)
        rel_vel_hand = hand_rot.T @ (obj_v_world - hand_v_world)

    slip_speed = float(np.linalg.norm(rel_vel_hand))
    return KinematicSlipReading(
        slip=slip_speed > epsilon_m_s,
        slip_speed_m_s=slip_speed,
        rel_vel_hand=rel_vel_hand,
    )
