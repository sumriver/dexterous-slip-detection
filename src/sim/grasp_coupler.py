"""Kinematic grasp coupling: keep bottle fixed relative to hand after contact."""

from __future__ import annotations

import mujoco
import numpy as np


def _mat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3) -> quaternion [w, x, y, z]."""
    m = rot
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array(
            [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
        )
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class GraspCoupler:
    """Maintain bottle pose in hand frame after successful finger contact."""

    def __init__(self, hand_body: str = "rh_forearm"):
        self.hand_body = hand_body
        self.offset_pos: np.ndarray | None = None
        self.offset_quat: np.ndarray | None = None
        self.active = False

    def capture(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.hand_body)
        bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
        r_h = data.xmat[hand_id].reshape(3, 3)
        p_h = data.xpos[hand_id]
        r_b = data.xmat[bottle_id].reshape(3, 3)
        p_b = data.xpos[bottle_id]

        r_off = r_h.T @ r_b
        self.offset_pos = r_h.T @ (p_b - p_h)
        self.offset_quat = _mat_to_quat(r_off)
        self.active = True

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not self.active or self.offset_pos is None or self.offset_quat is None:
            return

        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.hand_body)
        bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")
        qadr = model.jnt_qposadr[bottle_joint]
        vadr = model.jnt_dofadr[bottle_joint]

        r_h = data.xmat[hand_id].reshape(3, 3)
        p_h = data.xpos[hand_id]
        r_off = _quat_to_mat(self.offset_quat)
        r_b = r_h @ r_off
        p_b = p_h + r_h @ self.offset_pos

        data.qpos[qadr : qadr + 3] = p_b
        data.qpos[qadr + 3 : qadr + 7] = _mat_to_quat(r_b)
        data.qvel[vadr : vadr + 6] = 0.0
