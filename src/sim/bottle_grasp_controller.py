"""Phase 1 bottle grasp simulation: approach, grasp, lift 20 cm, flip 90°."""

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
    finger_qpos: np.ndarray


class BottleGraspController:
    """Kinematic hand-base trajectory with position-controlled fingers."""

    LIFT_HEIGHT_M = 0.20
    FLIP_DEG = 90.0

    # Hand approaches from -Y; grasp at bottle mid-height (z ≈ 0.89 m)
    APPROACH_POS = np.array([0.58, -0.12, 0.87])
    GRASP_POS = np.array([0.58, -0.04, 0.87])
    LIFT_POS = GRASP_POS + np.array([0.0, 0.0, LIFT_HEIGHT_M])

    BASE_QUAT = np.array([0.707107, 0.0, 0.707107, 0.0], dtype=float)

    PHASES: list[tuple[Phase, PhaseConfig]] = [
        (Phase.SETTLE, PhaseConfig(0.8, "open hand", "open hand")),
        (Phase.APPROACH, PhaseConfig(2.0, "open hand", "pre grasp")),
        (Phase.GRASP, PhaseConfig(1.5, "pre grasp", "grasp sphere")),
        (Phase.HOLD, PhaseConfig(1.0, "grasp sphere", "grasp sphere")),
        (Phase.LIFT, PhaseConfig(2.5, "grasp sphere", "grasp sphere")),
        (Phase.FLIP, PhaseConfig(3.0, "grasp sphere", "grasp sphere")),
        (Phase.DONE, PhaseConfig(1.0, "grasp sphere", "grasp sphere")),
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

    def _finger_qpos(self, key_name: str) -> np.ndarray:
        if key_name not in self._finger_cache:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
            if key_id < 0:
                raise ValueError(f"Keyframe not found: {key_name}")
            self._finger_cache[key_name] = self.model.key_qpos[key_id, 7:31].copy()
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
            pos = self.GRASP_POS + t * np.array([0.0, 0.0, self.LIFT_HEIGHT_M])
        elif phase in (Phase.FLIP, Phase.DONE):
            lift_pos = self.LIFT_POS.copy()
            flip_quat = multiply_quat(
                quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), -np.deg2rad(self.FLIP_DEG)),
                self.BASE_QUAT,
            )
            pos = lift_pos
            quat = flip_quat if phase == Phase.DONE else slerp(self.BASE_QUAT, flip_quat, t)

        start_f = self._finger_qpos(cfg.finger_key or "open hand")
        end_f = self._finger_qpos(cfg.finger_key_end or cfg.finger_key or "open hand")
        if phase.value >= Phase.GRASP.value:
            finger = end_f.copy()
        else:
            finger = start_f + t * (end_f - start_f)

        return TrajectoryTargets(pos=pos, quat=quat, finger_qpos=finger)

    def _set_actuator_ctrl(self, data: mujoco.MjData, finger_qpos: np.ndarray) -> None:
        """Map 24-DOF finger qpos to 20 actuators (incl. tendon combos)."""
        # finger_qpos layout: WRJ2, WRJ1, FFJ4..LFJ0 joints (24 values)
        joint_values = {i: finger_qpos[i] for i in range(24)}

        joint_index_by_name: dict[str, int] = {
            "rh_WRJ2": 0,
            "rh_WRJ1": 1,
            "rh_THJ5": 19,
            "rh_THJ4": 20,
            "rh_THJ3": 21,
            "rh_THJ2": 22,
            "rh_THJ1": 23,
            "rh_FFJ4": 2,
            "rh_FFJ3": 3,
            "rh_MFJ4": 6,
            "rh_MFJ3": 7,
            "rh_RFJ4": 10,
            "rh_RFJ3": 11,
            "rh_LFJ5": 14,
            "rh_LFJ4": 15,
            "rh_LFJ3": 16,
        }
        tendon_index_by_name = {
            "rh_FFJ0": (4, 5),
            "rh_MFJ0": (8, 9),
            "rh_RFJ0": (12, 13),
            "rh_LFJ0": (17, 18),
        }

        data.ctrl[:] = 0.0
        for act_id in range(self.model.nu):
            act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id) or ""
            short = act_name.replace("rh_A_", "rh_")
            if short in tendon_index_by_name:
                i, j = tendon_index_by_name[short]
                data.ctrl[act_id] = finger_qpos[i] + finger_qpos[j]
            elif short in joint_index_by_name:
                data.ctrl[act_id] = finger_qpos[joint_index_by_name[short]]

    def apply(self, data: mujoco.MjData) -> None:
        tgt = self.targets()
        adr = self.hand_free_adr
        vadr = self.hand_free_dof

        data.qpos[adr : adr + 3] = tgt.pos
        data.qpos[adr + 3 : adr + 7] = tgt.quat
        data.qvel[vadr : vadr + 6] = 0.0

        # Kinematic finger closure from GRASP onward (Shadow Hand cylinder contact tuning)
        if self.phase.value >= Phase.GRASP.value:
            data.qpos[7:31] = tgt.finger_qpos
            data.qvel[6:30] = 0.0

        self._set_actuator_ctrl(data, tgt.finger_qpos)
        mujoco.mj_forward(self.model, data)
