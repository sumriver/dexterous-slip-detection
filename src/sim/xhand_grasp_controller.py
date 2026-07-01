"""XHAND horizontal-bottle grasp: lateral tripod (thumb vs four fingers) → lift → stand → hold."""

from __future__ import annotations

import mujoco
import numpy as np

from sim.bottle_grasp_controller import Phase, PhaseConfig, TrajectoryTargets, multiply_quat, quat_from_axis_angle, slerp, smoothstep


def lateral_grasp_quat() -> np.ndarray:
    """Four fingers parallel to bottle axis (+X); thumb opposes on +Y side."""
    qy = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.pi / 2)
    q = multiply_quat(quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.8), qy)
    return multiply_quat(quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), -0.26), q)


class XHandGraspController:
    """Side pinch on horizontal cylinder — no top-down penetration."""

    LIFT_HEIGHT_M = 0.20
    STAND_DEG = 90.0
    LIFT_SPEED_M_S = 0.025
    BOTTLE_RADIUS_M = 0.022

    # Approach from -Y: fingers on -Y, thumb on +Y (validated by geom distance)
    APPROACH_FAR = np.array([0.55, -0.20, 0.10])
    APPROACH_SIDE = np.array([0.55, -0.14, 0.070])
    GRASP_POS = np.array([0.55, -0.112, 0.070])
    BASE_QUAT = lateral_grasp_quat()

    PHASES: list[tuple[Phase, PhaseConfig]] = [
        (Phase.SETTLE, PhaseConfig(1.5, "open hand", "open hand")),
        (Phase.APPROACH, PhaseConfig(7.0, "open hand", "open hand")),
        (Phase.GRASP, PhaseConfig(8.0, "open hand", "grasp lateral")),
        (Phase.HOLD, PhaseConfig(3.0, "grasp lateral", "grasp lateral")),
        (Phase.LIFT, PhaseConfig(9.0, "grasp lateral", "grasp lateral")),
        (Phase.FLIP, PhaseConfig(6.0, "grasp lateral", "grasp lateral")),
        (Phase.DONE, PhaseConfig(4.0, "grasp lateral", "grasp lateral")),
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
        self._stand_quat = multiply_quat(
            quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.deg2rad(self.STAND_DEG)),
            self.BASE_QUAT,
        )
        self.lift_enabled = False
        self.stand_enabled = False

    def reset_hand_pose(self, data: mujoco.MjData) -> None:
        adr = self.hand_free_adr
        data.qpos[adr : adr + 3] = self.APPROACH_FAR
        data.qpos[adr + 3 : adr + 7] = self.BASE_QUAT / np.linalg.norm(self.BASE_QUAT)
        open_ctrl = self._finger_ctrl_from_key("open hand")
        data.ctrl[:] = open_ctrl
        for i in range(12):
            data.qpos[adr + 7 + i] = open_ctrl[i]
        data.qvel[:] = 0.0

    def try_enable_lift(self, n_contacts: int, bottle_xy: np.ndarray) -> None:
        if n_contacts >= 3:
            self.lift_enabled = True

    def try_enable_stand(self, n_contacts: int, bottle_z: float, initial_z: float) -> None:
        if self.lift_enabled and n_contacts >= 2 and bottle_z > initial_z + 0.10:
            self.stand_enabled = True

    def _finger_ctrl_from_key(self, key_name: str) -> np.ndarray:
        if key_name not in self._finger_cache:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
            if key_id < 0:
                raise ValueError(f"Keyframe not found: {key_name}")
            self._finger_cache[key_name] = self.model.key_ctrl[key_id].copy()
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

    def _lifted_pos(self) -> np.ndarray:
        z0 = self._lift_start_z if self._lift_start_z is not None else float(self.GRASP_POS[2])
        return np.array([self.GRASP_POS[0], self.GRASP_POS[1], z0 + self.LIFT_HEIGHT_M])

    def targets(self) -> TrajectoryTargets:
        phase = self.phase
        t = self._phase_progress()
        _, cfg = self.PHASES[self._phase_index]

        pos = self.APPROACH_FAR.copy()
        quat = self.BASE_QUAT.copy()

        if phase == Phase.SETTLE:
            pos = self.APPROACH_FAR.copy()
        elif phase == Phase.APPROACH:
            if t < 0.4:
                u = smoothstep(t / 0.4)
                pos = self.APPROACH_FAR + u * (self.APPROACH_SIDE - self.APPROACH_FAR)
            else:
                u = smoothstep((t - 0.4) / 0.6)
                pos = self.APPROACH_SIDE + u * (self.GRASP_POS - self.APPROACH_SIDE)
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
        elif phase == Phase.FLIP:
            pos = self._lifted_pos()
            if self.stand_enabled:
                quat = slerp(self.BASE_QUAT, self._stand_quat, t)
            else:
                quat = self.BASE_QUAT.copy()
        elif phase == Phase.DONE:
            pos = self._lifted_pos()
            quat = self._stand_quat.copy() if self.stand_enabled else self.BASE_QUAT.copy()

        start_f = self._finger_ctrl_from_key(cfg.finger_key or "open hand")
        end_f = self._finger_ctrl_from_key(cfg.finger_key_end or cfg.finger_key or "open hand")

        if phase.value >= Phase.HOLD.value:
            finger = end_f.copy()
        elif phase == Phase.GRASP:
            finger = start_f + t * (end_f - start_f)
        else:
            finger = start_f + t * (end_f - start_f)

        return TrajectoryTargets(pos=pos, quat=quat, finger_ctrl=finger)

    def apply_arm_pose(self, data: mujoco.MjData) -> None:
        tgt = self.targets()
        adr = self.hand_free_adr
        vadr = self.hand_free_dof

        max_step = 0.00035
        current_pos = data.qpos[adr : adr + 3].copy()
        delta = tgt.pos - current_pos
        dist = float(np.linalg.norm(delta))
        if dist > max_step:
            current_pos = current_pos + delta * (max_step / dist)
        else:
            current_pos = tgt.pos.copy()

        current_quat = data.qpos[adr + 3 : adr + 7].copy()
        target_quat = tgt.quat / np.linalg.norm(tgt.quat)
        blend_t = min(1.0, 0.003 / max(np.linalg.norm(target_quat - current_quat), 1e-6))
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
