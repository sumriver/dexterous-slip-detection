"""Gymnasium environment: XHAND learns to grasp and lift horizontal bottle via RL."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_utils import get_hand_geom_ids
from scene_loader import load_xhand_scene
from sim.grasp_physics import measure_bottle_grasp
from sim.xhand_grasp_controller import lateral_grasp_quat


def _multiply_quat(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    half = angle * 0.5
    return np.array(
        [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
        dtype=np.float64,
    )


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / (np.linalg.norm(q) + 1e-9)


def _bottle_tilt_deg(data: mujoco.MjData, bottle_id: int) -> float:
    xmat = data.xmat[bottle_id].reshape(3, 3)
    axis = xmat[:, 2] / (np.linalg.norm(xmat[:, 2]) + 1e-9)
    cos_a = float(np.clip(np.dot(axis, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_a)))


class XHandGraspEnv(gym.Env):
    """RL env: policy controls finger actuators + hand base deltas (physics-only bottle)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    BOTTLE_ANCHOR = np.array([0.55, 0.0, 0.022])
    LIFT_TARGET = 0.20
    # Validated lateral pre-grasp region (see preview_xhand_grasp_pose.py)
    HAND_POS_INIT = np.array([0.55, -0.14, 0.070])
    HAND_POS_LO = np.array([0.52, -0.22, 0.035])
    HAND_POS_HI = np.array([0.58, -0.06, 0.14])

    def __init__(
        self,
        *,
        frame_skip: int = 20,
        max_episode_steps: int = 500,
        randomize_reset: bool = True,
        fix_orientation: bool = True,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.randomize_reset = randomize_reset
        self.fix_orientation = fix_orientation
        self.render_mode = render_mode

        self.model, self.data = load_xhand_scene()
        self.hand_geom_ids = get_hand_geom_ids(self.model)
        self.hand_free_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
        self.hand_free_adr = self.model.jnt_qposadr[self.hand_free_id]
        self.hand_free_dof = self.model.jnt_dofadr[self.hand_free_id]
        self.bottle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
        self.base_quat = lateral_grasp_quat()

        # 12 finger + 3 hand position deltas + (optional) 3 orientation deltas
        self.n_finger = self.model.nu
        self.n_pos = 3
        self.n_ori = 0 if fix_orientation else 3
        action_dim = self.n_finger + self.n_pos + self.n_ori
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # bottle_rel(3) + bottle_quat(4) + hand_rel(3) + hand_quat(4) + fingers(12) + contacts(1) + rel(3) + tilt(1)
        self._obs_dim = 3 + 4 + 3 + 4 + self.n_finger + 1 + 3 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

        self._finger_lo = np.zeros(self.n_finger)
        self._finger_hi = np.zeros(self.n_finger)
        for i in range(self.n_finger):
            jid = self.model.actuator_trnid[i, 0]
            self._finger_lo[i] = self.model.jnt_range[jid, 0]
            self._finger_hi[i] = self.model.jnt_range[jid, 1]

        self._renderer: mujoco.Renderer | None = None
        self._step_count = 0
        self._initial_bottle_z = 0.022
        self._episode_max_z = 0.022
        self._prev_ctrl = np.zeros(self.n_finger, dtype=np.float64)
        self._ctrl_alpha = 0.35  # low-pass on finger targets to avoid contact explosions

    def _map_finger_action(self, a: np.ndarray) -> np.ndarray:
        t = (a + 1.0) * 0.5
        return self._finger_lo + t * (self._finger_hi - self._finger_lo)

    def _get_obs(self) -> np.ndarray:
        d = self.data
        hand_pos = d.qpos[self.hand_free_adr : self.hand_free_adr + 3].copy()
        hand_quat = d.qpos[self.hand_free_adr + 3 : self.hand_free_adr + 7].copy()
        bottle_pos = d.xpos[self.bottle_id].copy()
        bottle_quat = d.xquat[self.bottle_id].copy()
        metrics = measure_bottle_grasp(self.model, d, self.hand_geom_ids)
        finger = d.ctrl.copy()
        finger_norm = (finger - self._finger_lo) / (self._finger_hi - self._finger_lo + 1e-9) * 2 - 1
        rel = bottle_pos - hand_pos
        tilt = _bottle_tilt_deg(d, self.bottle_id) / 90.0
        obs = np.concatenate(
            [
                bottle_pos - self.BOTTLE_ANCHOR,
                bottle_quat,
                hand_pos - self.HAND_POS_INIT,
                hand_quat,
                finger_norm,
                [metrics.n_contacts / 10.0],
                rel,
                [tilt],
            ]
        ).astype(np.float32)
        return obs

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        d = self.data
        metrics = measure_bottle_grasp(self.model, d, self.hand_geom_ids)
        bottle_pos = d.xpos[self.bottle_id]
        bottle_z = float(bottle_pos[2])
        xy_err = float(np.linalg.norm(bottle_pos[:2] - self.BOTTLE_ANCHOR[:2]))

        r_contact = min(metrics.n_contacts, 8) * 0.05
        r_lift = max(0.0, bottle_z - self._initial_bottle_z) * 80.0
        r_support = min(metrics.support_force_z, 3.0) * 0.15
        r_xy = -xy_err * 2.0
        r_action = -0.002 * float(np.linalg.norm(action))
        r_alive = 0.01

        reward = r_contact + r_lift + r_support + r_xy + r_action + r_alive

        lifted = bottle_z > self._initial_bottle_z + 0.08
        if lifted:
            reward += 2.0
        if bottle_z > self._initial_bottle_z + self.LIFT_TARGET - 0.02:
            reward += 10.0

        info = {
            "n_contacts": float(metrics.n_contacts),
            "bottle_z": bottle_z,
            "support_z": metrics.support_force_z,
            "xy_err": xy_err,
            "lifted": float(lifted),
        }
        return reward, info

    def _apply_action(self, action: np.ndarray) -> None:
        finger_a = action[: self.n_finger]
        pos_a = action[self.n_finger : self.n_finger + 3]
        ori_a = action[self.n_finger + 3 : self.n_finger + 3 + self.n_ori]

        adr = self.hand_free_adr
        pos = self.data.qpos[adr : adr + 3].copy()
        quat = self.data.qpos[adr + 3 : adr + 7].copy()

        pos += pos_a * np.array([0.0006, 0.0009, 0.0006], dtype=np.float64)
        pos = np.clip(pos, self.HAND_POS_LO, self.HAND_POS_HI)

        if self.fix_orientation:
            quat = self.base_quat.copy()
        else:
            dq = _quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), float(ori_a[0]) * 0.02)
            dq = _multiply_quat(_quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), float(ori_a[1]) * 0.02), dq)
            dq = _multiply_quat(_quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), float(ori_a[2]) * 0.02), dq)
            quat = _normalize_quat(_multiply_quat(dq, quat))

        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = quat
        self.data.qvel[self.hand_free_dof : self.hand_free_dof + 6] = 0.0
        target_ctrl = self._map_finger_action(finger_a)
        self._prev_ctrl = (1.0 - self._ctrl_alpha) * self._prev_ctrl + self._ctrl_alpha * target_ctrl
        self.data.ctrl[:] = self._prev_ctrl

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if self.randomize_reset and self.np_random is not None:
            pos = self.HAND_POS_INIT + self.np_random.uniform(-0.02, 0.02, size=3)
            pos[1] = np.clip(pos[1], self.HAND_POS_LO[1], self.HAND_POS_HI[1])
            pos[2] = np.clip(pos[2], self.HAND_POS_LO[2], self.HAND_POS_HI[2])
            finger_open = self._finger_lo + self.np_random.uniform(0.0, 0.15, size=self.n_finger) * (
                self._finger_hi - self._finger_lo
            )
        else:
            pos = self.HAND_POS_INIT.copy()
            finger_open = self._finger_lo.copy()

        adr = self.hand_free_adr
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = self.base_quat
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = finger_open
        self._prev_ctrl = finger_open.copy()
        for i in range(self.n_finger):
            self.data.qpos[adr + 7 + i] = finger_open[i]

        mujoco.mj_forward(self.model, self.data)
        self._initial_bottle_z = float(self.data.xpos[self.bottle_id][2])
        self._episode_max_z = self._initial_bottle_z
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        self._apply_action(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qacc).all() or not np.isfinite(self.data.qpos).all():
                reward = -10.0
                info = {
                    "n_contacts": 0.0,
                    "bottle_z": float(self._initial_bottle_z),
                    "support_z": 0.0,
                    "xy_err": 999.0,
                    "lifted": 0.0,
                    "episode_max_z": self._episode_max_z,
                    "unstable": 1.0,
                }
                return self._get_obs(), reward, True, False, info

        self._step_count += 1
        reward, info = self._compute_reward(action)
        bottle_pos = self.data.xpos[self.bottle_id]
        self._episode_max_z = max(self._episode_max_z, float(bottle_pos[2]))

        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        bottle_z = float(bottle_pos[2])
        xy_err = float(np.linalg.norm(bottle_pos[:2] - self.BOTTLE_ANCHOR[:2]))
        if bottle_z > self._initial_bottle_z + self.LIFT_TARGET - 0.01 and info["n_contacts"] >= 2:
            terminated = True
            reward += 20.0
        if xy_err > 0.25 or bottle_pos[2] > 2.0:
            terminated = True
            reward -= 5.0

        info["episode_max_z"] = self._episode_max_z
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.lookat[:] = np.array([0.55, 0.0, 0.08])
        cam.distance = 0.65
        cam.azimuth = 135
        cam.elevation = -8
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
