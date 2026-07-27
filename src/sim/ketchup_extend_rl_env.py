"""Gymnasium env: learn Policy2Action during ketchup extend (lift) only.

Episode = 200 mimic+lift steps after a cached post-trajectory snapshot.
Action = (grip, Δroll, Δpitch, Δyaw). Observation = slip features + proprio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from energy_flow import SlipDetector
from sim.antislip_control import Policy2Action, Policy2OpenLoopController
from sim.slip_nn_features import FEATURE_DIM, SlipFeatureBuilder, make_step_context
from sim.spider_ketchup import DEFAULT_WORKSPACE
from sim.spider_replay import (
    SpiderTaskConfig,
    build_extend_mimic_lift_controls,
    get_object_geom_ids,
    get_spider_hand_collision_geom_ids,
    load_trajectory_arrays,
    upsample_controls,
)
from sim.spider_scene_modify import apply_object_physics

ROOT = Path(__file__).resolve().parents[2]
SPIDER = ROOT / "third_party" / "spider"

EXTEND_S = 2.0
MIMIC_S = 1.0
LIFT_M = 0.10
SIM_DT = 0.01
REF_DT = 0.02
EXTEND_STEPS = int(round(EXTEND_S / SIM_DT))  # 200
DZ_PASS_M = 0.06
DROP_FAIL_M = 0.03


@dataclass(frozen=True)
class ExtendCase:
    name: str
    mass_scale: float = 1.0
    friction_scale: float = 1.0


DEFAULT_TRAIN_CASES: tuple[ExtendCase, ...] = (
    ExtendCase("friction_div2", friction_scale=0.50),
    ExtendCase("baseline", friction_scale=1.0),
    ExtendCase("friction_s045", friction_scale=0.45),
    ExtendCase("friction_s040", friction_scale=0.40),  # envelope — sparse
)


def _default_cfg() -> SpiderTaskConfig:
    return SpiderTaskConfig(
        dataset_dir=SPIDER / "example_datasets",
        dataset_name="arcticv2",
        robot_type="xhand",
        embodiment_type="right",
        task="s01-ketchup_use_01",
        workspace_root=DEFAULT_WORKSPACE,
    )


def _n_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geoms: set[int],
    object_geoms: set[int],
) -> int:
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in hand_geoms and g2 in object_geoms) or (g2 in hand_geoms and g1 in object_geoms):
            n += 1
    return n


class KetchupExtendRLEnv(gym.Env):
    """Per-step Policy-2 RL on the extend segment only."""

    metadata = {"render_modes": [], "render_fps": 100}

    def __init__(
        self,
        *,
        cases: tuple[ExtendCase, ...] | None = None,
        case_probs: tuple[float, ...] | None = None,
        g_max: float = 0.25,
        d_max: float = 0.25,
        rate_g: float = 0.02,
        rate_w: float = 0.02,
        grip_ratchet: bool = False,
        lambda_grip: float = 0.05,
        lambda_wrist: float = 0.02,
        seed: int | None = None,
    ):
        super().__init__()
        self.cases = cases or DEFAULT_TRAIN_CASES
        if case_probs is None:
            # Prefer div2; s040 rare (no open-loop teacher — exploration only)
            raw = []
            for c in self.cases:
                if c.name == "friction_div2":
                    raw.append(0.55)
                elif c.name == "friction_s040":
                    raw.append(0.10)
                else:
                    raw.append(0.175)
            s = sum(raw)
            self.case_probs = tuple(x / s for x in raw)
        else:
            self.case_probs = case_probs
        self.g_max = float(g_max)
        self.d_max = float(d_max)
        self.rate_g = float(rate_g)
        self.rate_w = float(rate_w)
        self.grip_ratchet = bool(grip_ratchet)
        self.lambda_grip = float(lambda_grip)
        self.lambda_wrist = float(lambda_wrist)
        self._rng = np.random.default_rng(seed)

        self.cfg = _default_cfg()
        if not self.cfg.scene_path.exists():
            raise FileNotFoundError(f"Missing scene: {self.cfg.scene_path}")

        # obs: features(26) + grip_extra + wrist(3) + t_frac + friction + mass = 33
        self._obs_dim = FEATURE_DIM + 1 + 3 + 1 + 1 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        # action in [-1,1]^4 → mapped to grip∈[0,g_max], wrist∈[-d_max,d_max]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._active: ExtendCase | None = None
        self._extend_ctrl: np.ndarray | None = None
        self._step_i = 0
        self._z0 = 0.0
        self._z_traj_end = 0.0
        self._z_start = 0.0
        self._contact_steps = 0
        self._max_grip = 0.0
        self._ctrl: Policy2OpenLoopController | None = None
        self._fb: SlipFeatureBuilder | None = None
        self._hand_geoms: set[int] = set()
        self._object_geoms: set[int] = set()
        self._object_id = -1
        self._detector = SlipDetector(window_size=30, threshold=0.15)

        # Pre-build snapshots for all cases (one traj replay each)
        for case in self.cases:
            self._ensure_snapshot(case)

    def _ensure_snapshot(self, case: ExtendCase) -> None:
        if case.name in self._snapshots:
            return
        model = mujoco.MjModel.from_xml_path(str(self.cfg.scene_path))
        apply_object_physics(
            model,
            mass_scale=case.mass_scale,
            friction_scale=case.friction_scale,
            object_body="right_object",
        )
        model.opt.timestep = SIM_DT
        data = mujoco.MjData(model)
        qpos_ref, qvel_ref, ctrl_ref = load_trajectory_arrays(
            self.cfg.trajectory_path, model, self.cfg.data_type
        )
        qpos_ref, qvel_ref, ctrl_ref = upsample_controls(
            qpos_ref, qvel_ref, ctrl_ref, SIM_DT, REF_DT
        )
        hand_geoms = get_spider_hand_collision_geom_ids(model)
        object_geoms = get_object_geom_ids(model, "right_object")
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")

        data.qpos[:] = qpos_ref[0]
        data.qvel[:] = qvel_ref[0]
        data.ctrl[:] = ctrl_ref[0]
        mujoco.mj_forward(model, data)
        z_start = float(data.xpos[object_id][2])

        # Match replay_spider_task trajectory phase: ctrl + mj_step (physics).
        for ctrl in ctrl_ref:
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

        z_traj_end = float(data.xpos[object_id][2])
        extend_ctrl = build_extend_mimic_lift_controls(
            ctrl_ref,
            sim_dt=SIM_DT,
            extend_s=EXTEND_S,
            mimic_s=MIMIC_S,
            lift_m=LIFT_M,
            arm_tz_index=2,
        )
        self._snapshots[case.name] = {
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "ctrl": data.ctrl.copy(),
            "time": float(data.time),
            "z_start": z_start,
            "z_traj_end": z_traj_end,
            "extend_ctrl": extend_ctrl.copy(),
            "mass_scale": case.mass_scale,
            "friction_scale": case.friction_scale,
        }
        del data
        del model

    def _build_model_for(self, case: ExtendCase) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.scene_path))
        apply_object_physics(
            self.model,
            mass_scale=case.mass_scale,
            friction_scale=case.friction_scale,
            object_body="right_object",
        )
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        self._hand_geoms = get_spider_hand_collision_geom_ids(self.model)
        self._object_geoms = get_object_geom_ids(self.model, "right_object")
        self._object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_object")

    def _map_action(self, action: np.ndarray) -> Policy2Action:
        a = np.asarray(action, dtype=np.float64).reshape(4)
        a = np.clip(a, -1.0, 1.0)
        grip = float((a[0] + 1.0) * 0.5 * self.g_max)  # [-1,1] → [0,g_max]
        wrist = tuple(float(x) * self.d_max for x in a[1:])
        return Policy2Action(grip=grip, wrist_delta=wrist)

    def _obs(self) -> np.ndarray:
        assert self.model is not None and self.data is not None and self._ctrl is not None
        assert self._fb is not None and self._active is not None and self._extend_ctrl is not None
        ctrl_row = self._extend_ctrl[min(self._step_i, len(self._extend_ctrl) - 1)]
        ctx = make_step_context(
            phase="extend",
            wrist_tz=float(ctrl_row[2]),
            grip_extra=float(self._ctrl.grip_extra),
            friction_scale=self._active.friction_scale,
            object_z=float(self.data.xpos[self._object_id][2]),
            object_z_traj_end=self._z_traj_end,
            object_z_extend_start=self._z0,
            object_z_start=self._z_start,
            in_trajectory=False,
        )
        feat = self._fb.build(
            self.model,
            self.data,
            self._hand_geoms,
            self._object_geoms,
            self._object_id,
            ctx,
        ).features.astype(np.float32)
        t_frac = np.float32(self._step_i / max(1, EXTEND_STEPS - 1))
        extra = np.array(
            [
                float(self._ctrl.grip_extra),
                float(self._ctrl.wrist_cmd[0]),
                float(self._ctrl.wrist_cmd[1]),
                float(self._ctrl.wrist_cmd[2]),
                float(t_frac),
                float(self._active.friction_scale),
                float(self._active.mass_scale),
            ],
            dtype=np.float32,
        )
        return np.concatenate([feat, extra], axis=0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options and "case" in options:
            name = str(options["case"])
            case = next(c for c in self.cases if c.name == name)
        else:
            idx = int(self._rng.choice(len(self.cases), p=self.case_probs))
            case = self.cases[idx]

        self._ensure_snapshot(case)
        snap = self._snapshots[case.name]
        if self._active is None or self._active.name != case.name or self.model is None:
            self._build_model_for(case)
        assert self.model is not None and self.data is not None

        self.data.qpos[:] = snap["qpos"]
        self.data.qvel[:] = snap["qvel"]
        self.data.ctrl[:] = snap["ctrl"]
        self.data.time = snap["time"]
        mujoco.mj_forward(self.model, self.data)

        self._active = case
        self._extend_ctrl = snap["extend_ctrl"]
        self._z0 = float(self.data.xpos[self._object_id][2])
        self._z_traj_end = float(snap["z_traj_end"])
        self._z_start = float(snap["z_start"])
        self._step_i = 0
        self._contact_steps = 0
        self._max_grip = 0.0
        self._ctrl = Policy2OpenLoopController(
            Policy2Action(),
            g_max=self.g_max,
            d_max=self.d_max,
            rate_g=self.rate_g,
            rate_w=self.rate_w,
            grip_ratchet=self.grip_ratchet,
        )
        self._fb = SlipFeatureBuilder(sim_dt=SIM_DT)
        self._fb.reset_trajectory(self._z_start)
        self._fb.mark_trajectory_end(self._z_traj_end)
        self._fb.reset_extend(self._z0)
        self._detector = SlipDetector(window_size=30, threshold=0.15)

        return self._obs(), {"case": case.name}

    def step(self, action: np.ndarray):
        assert (
            self.model is not None
            and self.data is not None
            and self._ctrl is not None
            and self._extend_ctrl is not None
            and self._active is not None
        )
        act = self._map_action(action)
        self._ctrl.set_action(act)
        ctrl_row = self._extend_ctrl[self._step_i]
        applied = self._ctrl.apply(ctrl_row, self.model)
        self.data.ctrl[:] = applied
        mujoco.mj_step(self.model, self.data)

        n_con = _n_contacts(self.model, self.data, self._hand_geoms, self._object_geoms)
        if n_con > 0:
            self._contact_steps += 1
        self._max_grip = max(self._max_grip, float(self._ctrl.grip_extra))

        z = float(self.data.xpos[self._object_id][2])
        dz = z - self._z0
        drop = max(0.0, self._z0 - z)

        # Dense reward shaped like PASS − λ·grip − μ·wrist
        r = 0.0
        r += 0.01 if n_con > 0 else -0.02
        r += 0.5 * float(np.clip(dz / DZ_PASS_M, -1.0, 1.5))
        r -= self.lambda_grip * float(self._ctrl.grip_extra)
        r -= self.lambda_wrist * float(np.linalg.norm(self._ctrl.wrist_cmd))

        self._step_i += 1
        terminated = False
        truncated = self._step_i >= EXTEND_STEPS
        info: dict[str, Any] = {
            "case": self._active.name,
            "extend_dz_cm": dz * 100.0,
            "extend_contact_steps": self._contact_steps,
            "max_grip": self._max_grip,
            "drop_cm": drop * 100.0,
        }

        if truncated:
            gate_ok = dz >= DZ_PASS_M and self._contact_steps >= EXTEND_STEPS
            if gate_ok:
                r += 5.0
            elif dz >= 0.03 and self._contact_steps >= 100:
                r += 1.0
            if drop > DROP_FAIL_M or self._contact_steps < 30:
                r -= 5.0
            info["gate_ok"] = bool(gate_ok)
            info["pass"] = bool(gate_ok)

        obs = self._obs() if not truncated else np.zeros(self._obs_dim, dtype=np.float32)
        return obs, float(r), terminated, truncated, info
