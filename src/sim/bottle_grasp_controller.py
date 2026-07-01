"""Phase 1 bottle grasp — physics-only finger control, no kinematic bottle coupling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import mujoco
import numpy as np


class Phase(Enum):
    SETTLE = auto()
    APPROACH = auto()
    GRASP = auto()
    HOLD = auto()
    LIFT = auto()
    FLIP = auto()
    DONE = auto()


@dataclass
class PhaseConfig:
    duration_s: float
    finger_key: str | None = None
    finger_key_end: str | None = None


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta_0)
    w0 = np.sin((1.0 - t) * theta_0) / sin_theta
    w1 = np.sin(t * theta_0) / sin_theta
    return w0 * q0 + w1 * q1


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    half = angle * 0.5
    return np.array(
        [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
        dtype=float,
    )


def multiply_quat(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


@dataclass
class TrajectoryTargets:
    pos: np.ndarray
    quat: np.ndarray
    finger_ctrl: np.ndarray


def finger_qpos_to_actuator_ctrl(finger_qpos: np.ndarray) -> np.ndarray:
    """Map 24 hand joint qpos values to 20 position actuator targets."""
    q = finger_qpos
    return np.array(
        [
            q[0],
            q[1],  # WRJ2, WRJ1
            q[19],
            q[20],
            q[21],
            q[22],
            q[23],  # THJ5..THJ1
            q[2],
            q[3],
            q[4] + q[5],  # FFJ4, FFJ3, FFJ0 tendon
            q[6],
            q[7],
            q[8] + q[9],  # MF
            q[10],
            q[11],
            q[12] + q[13],  # RF
            q[14],
            q[15],
            q[16],
            q[17] + q[18],  # LF
        ],
        dtype=float,
    )


class BottleGraspController:
    """Arm path is planned (kinematic); fingers and bottle obey contact physics."""

    LIFT_HEIGHT_M = 0.20
    FLIP_DEG = 90.0
    LIFT_SPEED_M_S = 0.02

    APPROACH_POS = np.array([0.55, -0.38, 0.91])
    GRASP_POS = np.array([0.55, -0.30, 0.91])
    # Palm faces +Y toward bottle (was wrong: previous quat pointed fingers away)
    BASE_QUAT = np.array([0.5, -0.5, 0.5, 0.5], dtype=float)

    PHASES: list[tuple[Phase, PhaseConfig]] = [
        (Phase.SETTLE, PhaseConfig(1.0, "open hand", "open hand")),
        (Phase.APPROACH, PhaseConfig(4.0, "open hand", "pre grasp")),
        (Phase.GRASP, PhaseConfig(4.0, "pre grasp", "grasp soft")),
        (Phase.HOLD, PhaseConfig(2.5, "grasp soft", "grasp soft")),
        (Phase.LIFT, PhaseConfig(5.0, "grasp soft", "grasp soft")),
        (Phase.FLIP, PhaseConfig(4.0, "grasp soft", "grasp soft")),
        (Phase.DONE, PhaseConfig(1.0, "grasp soft", "grasp soft")),
    ]

    def __init__(self, model: mujoco.MjModel, timestep: float):
        self.model = model
        self.timestep = timestep
        hand_free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
        self.hand_free_adr = model.jnt_qposadr[hand_free_id]
        self.hand_free_dof = model.jnt_dofadr[hand_free_id]
        self._phase_index = 0
        self._phase_time = 0.0
        self._total_time = 0.0
        self._finger_cache: dict[str, np.ndarray] = {}
        self._lift_start_z: float | None = None
        self._cylinder_grasp_ctrl = self._build_cylinder_grasp_ctrl(model)
        self.lift_enabled = False
        self.flip_enabled = False

    def try_enable_lift(self, n_contacts: int, bottle_xy: np.ndarray) -> None:
        stable = float(np.linalg.norm(bottle_xy - np.array([0.55, 0.0]))) < 0.08
        if n_contacts >= 3 and stable:
            self.lift_enabled = True

    def try_enable_flip(self, n_contacts: int, bottle_z: float, initial_z: float) -> None:
        if self.lift_enabled and n_contacts >= 2 and bottle_z > initial_z + 0.03:
            self.flip_enabled = True

    def _build_cylinder_grasp_ctrl(self, model: mujoco.MjModel) -> np.ndarray:
        """Cylinder side-pinch: start from grasp hard keyframe, boost finger closure."""
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "grasp soft")
        base = finger_qpos_to_actuator_ctrl(model.key_qpos[key_id, 7:31].copy())
        # Stronger curl on index/middle/ring tendons for 44 mm diameter bottle
        boost = np.zeros(model.nu)
        for act_id in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id) or ""
            if name in ("rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0"):
                boost[act_id] = 0.15
        return base + boost

    def _finger_ctrl_from_key(self, key_name: str) -> np.ndarray:
        if key_name not in self._finger_cache:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
            if key_id < 0:
                raise ValueError(f"Keyframe not found: {key_name}")
            self._finger_cache[key_name] = finger_qpos_to_actuator_ctrl(self.model.key_qpos[key_id, 7:31])
        return self._finger_cache[key_name]

    @property
    def phase(self) -> Phase:
        return self.PHASES[self._phase_index][0]

    @property
    def total_time(self) -> float:
        return self._total_time

    def advance(self) -> None:
        self._phase_time += self.timestep
        self._total_time += self.timestep
        _, cfg = self.PHASES[self._phase_index]
        if self._phase_time >= cfg.duration_s and self._phase_index < len(self.PHASES) - 1:
            if self.phase == Phase.LIFT:
                self._lift_start_z = None
            self._phase_index += 1
            self._phase_time = 0.0

    def _phase_progress(self) -> float:
        _, cfg = self.PHASES[self._phase_index]
        return smoothstep(self._phase_time / max(cfg.duration_s, 1e-6))

    def targets(self) -> TrajectoryTargets:
        phase = self.phase
        t = self._phase_progress()
        _, cfg = self.PHASES[self._phase_index]

        pos = self.APPROACH_POS.copy()
        quat = self.BASE_QUAT.copy()

        if phase == Phase.SETTLE:
            pos = self.APPROACH_POS.copy()
        elif phase == Phase.APPROACH:
            pos = self.APPROACH_POS + t * (self.GRASP_POS - self.APPROACH_POS)
        elif phase in (Phase.GRASP, Phase.HOLD):
            pos = self.GRASP_POS.copy()
        elif phase == Phase.LIFT:
            if not self.lift_enabled:
                pos = self.GRASP_POS.copy()
            else:
                if self._lift_start_z is None:
                    self._lift_start_z = float(self.GRASP_POS[2])
                dz = min(self.LIFT_HEIGHT_M, self.LIFT_SPEED_M_S * self._phase_time)
                pos = self.GRASP_POS.copy()
                pos[2] = self._lift_start_z + dz
        elif phase in (Phase.FLIP, Phase.DONE):
            if not self.flip_enabled:
                pos = self.GRASP_POS.copy()
                quat = self.BASE_QUAT.copy()
            else:
                lift_z = self.GRASP_POS[2] + self.LIFT_HEIGHT_M
                pos = np.array([self.GRASP_POS[0], self.GRASP_POS[1], lift_z])
                flip_quat = multiply_quat(
                    quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), -np.deg2rad(self.FLIP_DEG)),
                    self.BASE_QUAT,
                )
                quat = flip_quat if phase == Phase.DONE else slerp(self.BASE_QUAT, flip_quat, t)

        start_f = self._finger_ctrl_from_key(cfg.finger_key or "open hand")
        end_f = self._finger_ctrl_from_key(cfg.finger_key_end or cfg.finger_key or "open hand")
        grasp_target = self._cylinder_grasp_ctrl

        if phase.value >= Phase.HOLD.value:
            finger = grasp_target.copy()
        elif phase == Phase.GRASP:
            finger = start_f + t * (grasp_target - start_f)
        else:
            finger = start_f + t * (end_f - start_f)

        return TrajectoryTargets(pos=pos, quat=quat, finger_ctrl=finger)

    def apply_arm_pose(self, data: mujoco.MjData) -> None:
        """Move arm along planned path with per-step displacement cap (avoid interpenetration snap)."""
        tgt = self.targets()
        adr = self.hand_free_adr
        vadr = self.hand_free_dof

        max_step = 0.0015  # 1.5 mm per step @ 1 kHz
        current_pos = data.qpos[adr : adr + 3].copy()
        delta = tgt.pos - current_pos
        dist = float(np.linalg.norm(delta))
        if dist > max_step:
            current_pos = current_pos + delta * (max_step / dist)
        else:
            current_pos = tgt.pos.copy()

        # Slerp rotation with capped step
        current_quat = data.qpos[adr + 3 : adr + 7].copy()
        target_quat = tgt.quat / np.linalg.norm(tgt.quat)
        # small rotation blend per step
        blend_t = min(1.0, 0.004 / max(np.linalg.norm(target_quat - current_quat), 1e-6))
        new_quat = slerp(current_quat, target_quat, blend_t)
        new_quat = new_quat / np.linalg.norm(new_quat)

        data.qpos[adr : adr + 3] = current_pos
        data.qpos[adr + 3 : adr + 7] = new_quat
        data.qvel[vadr : vadr + 6] = 0.0

    def apply_finger_actuators(self, data: mujoco.MjData) -> None:
        data.ctrl[:] = self.targets().finger_ctrl

    def apply(self, data: mujoco.MjData) -> None:
        self.apply_arm_pose(data)
        self.apply_finger_actuators(data)
        mujoco.mj_forward(self.model, data)
