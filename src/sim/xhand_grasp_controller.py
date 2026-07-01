"""XHAND bottle grasp controller — physics-only finger actuation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import mujoco
import numpy as np

from sim.bottle_grasp_controller import Phase, PhaseConfig, TrajectoryTargets, multiply_quat, quat_from_axis_angle, slerp, smoothstep


@dataclass
class FingerPreset:
    name: str
    ctrl: np.ndarray


class XHandGraspController:
    """Planned arm pose + position-actuated 12-DoF fingers."""

    LIFT_HEIGHT_M = 0.20
    FLIP_DEG = 90.0
    LIFT_SPEED_M_S = 0.02

    # Palm on -Y side; rotate ~-70° about X so fingers reach toward +Y (bottle at y=0)
    APPROACH_POS = np.array([0.55, -0.16, 0.92])
    GRASP_POS = np.array([0.55, -0.105, 0.895])
    BASE_QUAT = np.array([np.cos(-0.6), np.sin(-0.6), 0.0, 0.0], dtype=float)

    PHASES: list[tuple[Phase, PhaseConfig]] = [
        (Phase.SETTLE, PhaseConfig(1.0, "open hand", "open hand")),
        (Phase.APPROACH, PhaseConfig(4.0, "open hand", "pre grasp")),
        (Phase.GRASP, PhaseConfig(4.0, "pre grasp", "grasp soft")),
        (Phase.HOLD, PhaseConfig(2.5, "grasp soft", "grasp soft")),
        (Phase.LIFT, PhaseConfig(5.0, "grasp soft", "grasp soft")),
        (Phase.FLIP, PhaseConfig(4.0, "grasp soft", "grasp soft")),
        (Phase.DONE, PhaseConfig(1.0, "grasp soft", "grasp soft")),
    ]

    # 12 actuators: thumb(3), index(3), mid(2), ring(2), pinky(2)
    CYLINDER_GRASP_BOOST = np.array(
        [0.08, 0.06, 0.08, 0.03, 0.10, 0.08, 0.10, 0.08, 0.08, 0.06, 0.06, 0.05],
        dtype=float,
    )

    def reset_hand_pose(self, data: mujoco.MjData) -> None:
        """Place hand at approach pose with open fingers before simulation starts."""
        adr = self.hand_free_adr
        data.qpos[adr : adr + 3] = self.APPROACH_POS
        data.qpos[adr + 3 : adr + 7] = self.BASE_QUAT / np.linalg.norm(self.BASE_QUAT)
        open_ctrl = self._finger_ctrl_from_key("open hand")
        data.ctrl[:] = open_ctrl
        hand_joint_adr = adr + 7
        for i in range(12):
            data.qpos[hand_joint_adr + i] = open_ctrl[i]
        data.qvel[:] = 0.0

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
        if n_contacts >= 2 and stable:
            self.lift_enabled = True

    def try_enable_flip(self, n_contacts: int, bottle_z: float, initial_z: float) -> None:
        if self.lift_enabled and n_contacts >= 2 and bottle_z > initial_z + 0.03:
            self.flip_enabled = True

    def _finger_ctrl_from_key(self, key_name: str) -> np.ndarray:
        if key_name not in self._finger_cache:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
            if key_id < 0:
                raise ValueError(f"Keyframe not found: {key_name}")
            self._finger_cache[key_name] = self.model.key_ctrl[key_id].copy()
        return self._finger_cache[key_name]

    def _build_cylinder_grasp_ctrl(self, model: mujoco.MjModel) -> np.ndarray:
        base = self._finger_ctrl_from_key("grasp soft")
        boosted = base + self.CYLINDER_GRASP_BOOST
        for act_id in range(model.nu):
            joint_id = model.actuator_trnid[act_id, 0]
            lo, hi = model.jnt_range[joint_id]
            boosted[act_id] = float(np.clip(boosted[act_id], lo, hi))
        return boosted

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
        tgt = self.targets()
        adr = self.hand_free_adr
        vadr = self.hand_free_dof

        max_step = 0.0008  # slower approach to reduce impact impulses
        current_pos = data.qpos[adr : adr + 3].copy()
        delta = tgt.pos - current_pos
        dist = float(np.linalg.norm(delta))
        if dist > max_step:
            current_pos = current_pos + delta * (max_step / dist)
        else:
            current_pos = tgt.pos.copy()

        current_quat = data.qpos[adr + 3 : adr + 7].copy()
        target_quat = tgt.quat / np.linalg.norm(tgt.quat)
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
