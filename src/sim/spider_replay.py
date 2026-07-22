"""Replay SPIDER precomputed trajectories in MuJoCo (CPU) with optional energy-flow logging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from energy_flow import SlipDetector, compute_applied_power, compute_mass_estimate
from energy_flow.state import compute_retained_power
from sim.grasp_validate import GraspPhysicsReport, format_grasp_report, validate_grasp_physics
from sim.spider_scene_modify import apply_object_physics
from sim.video_recorder import VideoRecorder, make_spider_ketchup_camera
from sim.antislip_control import GripBoostController, Policy2Action, Policy2OpenLoopController
from sim.slip_vertical_support import (
    VerticalSupportAntislipDetector,
    gravity_up,
    measure_vertical_support,
)
from sim.slip_nn_features import SlipFeatureBuilder, make_step_context
from sim.slip_dataset_logger import SlipDatasetLogger, SlipDatasetMeta
from sim.slip_nn_detector import SlipNeuralDetector


@dataclass
class SpiderTaskConfig:
    dataset_dir: Path
    dataset_name: str = "oakinkv2"
    robot_type: str = "xhand"
    embodiment_type: str = "right"
    task: str = "pick_spoon_bowl"
    data_id: int = 0
    data_type: str = "mjwp_fast"
    workspace_root: Path | None = None

    @property
    def task_dir(self) -> Path:
        if self.workspace_root is not None:
            return self.workspace_root
        return (
            self.dataset_dir
            / "processed"
            / self.dataset_name
            / self.robot_type
            / self.embodiment_type
            / self.task
        )

    @property
    def scene_path(self) -> Path:
        return self.task_dir / "scene.xml"

    @property
    def trajectory_path(self) -> Path:
        if self.workspace_root is not None:
            return self.task_dir / f"trajectory_{self.data_type}.npz"
        return self.task_dir / str(self.data_id) / f"trajectory_{self.data_type}.npz"


@dataclass
class ReplayResult:
    steps: int
    contact_steps: int
    slip_events: int
    object_z_start: float
    object_z_end: float
    object_dz: float
    post_lift_dz: float = 0.0
    post_lift_contact_steps: int = 0
    post_extend_s: float = 0.0
    post_extend_object_dz: float = 0.0
    post_extend_contact_steps: int = 0
    object_z_after_trajectory: float = 0.0
    grasp_physics_ok: bool = False
    grasp_report: GraspPhysicsReport | None = None
    log_path: Path | None = None
    video_path: Path | None = None
    physics_meta: dict | None = None
    center_slip_events: int = 0
    support_slip_events: int = 0
    nn_slip_events: int = 0
    antislip_max_grip: float = 0.0
    antislip_scheme: int = 0


@dataclass
class _StepMetrics:
    n_contacts: int
    mass_estimate: float
    slipped: bool


def _log_energy_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geoms: set[int],
    object_geoms: set[int],
    object_id: int,
    detector: SlipDetector,
    slip_counter: list[int],
) -> _StepMetrics:
    forces, _, velocities = extract_hand_object_contacts(model, data, hand_geoms, object_geoms)
    n_con = len(forces)
    mass_est = float("nan")
    slipped = False
    if n_con > 0:
        applied = compute_applied_power(forces, velocities)
        total_force = np.sum(forces, axis=0)
        obj_vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, object_id, obj_vel, 0)
        retained = compute_retained_power(total_force, obj_vel[:3])
        mass_est = compute_mass_estimate(applied, retained)
        slipped = detector.update(mass_est)
        if slipped:
            slip_counter[0] += 1
    return _StepMetrics(n_con, mass_est, slipped)


@dataclass
class _EnergyLog:
    step: list[int] = field(default_factory=list)
    sim_time: list[float] = field(default_factory=list)
    n_contacts: list[int] = field(default_factory=list)
    mass_estimate: list[float] = field(default_factory=list)
    slip: list[bool] = field(default_factory=list)
    object_z: list[float] = field(default_factory=list)


def get_spider_hand_collision_geom_ids(model: mujoco.MjModel) -> set[int]:
    ids: set[int] = set()
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("collision_hand_right_"):
            ids.add(gid)
    return ids


def get_object_geom_ids(model: mujoco.MjModel, object_body: str = "right_object") -> set[int]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if bid < 0:
        return set()
    return {gid for gid in range(model.ngeom) if model.geom_bodyid[gid] == bid}


def extract_hand_object_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geom_ids: set[int],
    object_geom_ids: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forces/positions/velocities for hand↔object contacts (world frame)."""
    forces: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        hand_hit = g1 in hand_geom_ids or g2 in hand_geom_ids
        obj_hit = g1 in object_geom_ids or g2 in object_geom_ids
        if not (hand_hit and obj_hit):
            continue

        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        frame = np.array(contact.frame, dtype=float).reshape(3, 3)
        forces.append(frame @ wrench[:3])
        positions.append(contact.pos.copy())

        hand_gid = g1 if g1 in hand_geom_ids else g2
        body_id = model.geom_bodyid[hand_gid]
        body_vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, body_vel, 0)
        velocities.append(body_vel[:3].copy())

    if not forces:
        empty = np.zeros((0, 3))
        return empty, empty, empty
    return np.array(forces), np.array(positions), np.array(velocities)


def load_trajectory_arrays(
    traj_path: Path,
    model: mujoco.MjModel,
    data_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (qpos, qvel, ctrl) each shape (T, dof)."""
    raw = np.load(traj_path)
    if data_type == "mjwp_fast":
        attempt = 0
        if "rew_mean" in raw:
            attempt = int(np.argmax(raw["rew_mean"].sum(axis=1)))
        elif "succeeded" in raw:
            succ = raw["succeeded"].reshape(-1)
            attempt = int(np.argmax(succ)) if succ.any() else 0
        qpos = raw["qpos"][attempt]
        qvel = raw["qvel"][attempt]
        ctrl = raw["ctrl"][attempt]
    else:
        qpos = raw["qpos"].reshape(-1, model.nq)
        qvel = raw["qvel"].reshape(-1, model.nv)
        if "ctrl" in raw:
            ctrl = raw["ctrl"].reshape(-1, model.nu)
        else:
            ctrl = qpos[:, : model.nu]
    return qpos, qvel, ctrl


def upsample_controls(
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    sim_dt: float = 0.01,
    ref_dt: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    repeat = max(1, int(round(ref_dt / sim_dt)))
    qpos_u = qpos[:, None, :].repeat(repeat, axis=1).reshape(-1, qpos.shape[1])
    qvel_u = qvel[:, None, :].repeat(repeat, axis=1).reshape(-1, qvel.shape[1])
    ctrl_u = ctrl[:, None, :].repeat(repeat, axis=1).reshape(-1, ctrl.shape[1])
    return qpos_u, qvel_u, ctrl_u


def build_extend_mimic_lift_controls(
    ctrl_ref: np.ndarray,
    *,
    sim_dt: float,
    extend_s: float,
    mimic_s: float,
    lift_m: float,
    arm_tz_index: int = 2,
) -> np.ndarray:
    """Cycle the last ``mimic_s`` of controls while ramping wrist tz by ``lift_m`` over ``extend_s``."""
    mimic_steps = max(1, int(round(mimic_s / sim_dt)))
    extend_steps = max(1, int(round(extend_s / sim_dt)))
    tail = ctrl_ref[-min(mimic_steps, len(ctrl_ref)) :].copy()
    tz0 = float(ctrl_ref[-1, arm_tz_index])
    out = np.zeros((extend_steps, ctrl_ref.shape[1]))
    for i in range(extend_steps):
        alpha = (i + 1) / extend_steps
        out[i] = tail[i % len(tail)]
        out[i, arm_tz_index] = tz0 + lift_m * alpha
    return out


def find_grasp_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_ref: np.ndarray,
    qvel_ref: np.ndarray,
    ctrl_ref: np.ndarray,
    hand_geoms: set[int],
    object_geoms: set[int],
    object_body: str,
    search_frac: float = 0.8,
) -> tuple[int, GraspPhysicsReport]:
    """Best frame for lift attempt; prefers passing physics validation."""
    data.qpos[:] = qpos_ref[0]
    data.qvel[:] = qvel_ref[0]
    data.ctrl[:] = ctrl_ref[0]
    mujoco.mj_forward(model, data)

    limit = max(1, int(search_frac * len(ctrl_ref)))
    best_idx = 0
    best_score = -1e9
    best_report = validate_grasp_physics(model, data, object_body, hand_geoms, object_geoms)

    for i in range(limit):
        data.ctrl[:] = ctrl_ref[i]
        mujoco.mj_step(model, data)
        report = validate_grasp_physics(model, data, object_body, hand_geoms, object_geoms)
        if report.n_floor_object_contacts > 0:
            continue
        if report.n_hand_object_contacts < 2:
            continue
        score = report.n_hand_object_contacts * 10 + report.support_force_z
        if report.ok:
            score += 1000
        if score > best_score:
            best_score = score
            best_idx = i
            best_report = report

    return best_idx, best_report


def replay_spider_task(
    cfg: SpiderTaskConfig,
    out_dir: Path,
    *,
    sim_dt: float = 0.01,
    ref_dt: float = 0.02,
    save_video: bool = True,
    video_fps: int = 30,
    object_body: str = "right_object",
    post_lift_m: float = 0.0,
    post_extend_s: float = 0.0,
    post_mimic_s: float = 1.0,
    post_hold_steps: int = 60,
    post_lift_steps: int = 150,
    arm_tz_index: int = 2,
    mass_scale: float = 1.0,
    friction_scale: float = 1.0,
    log_energy: bool = True,
    antislip: bool = False,
    antislip_scheme: int = 2,
    antislip_sep_threshold_m: float = 0.008,
    antislip_grip_step: float = 0.015,
    antislip_grip_max: float = 0.25,
    antislip_slip_ratio: float = 0.7,
    antislip_smooth_window_s: float = 0.2,
    antislip_avg_window_s: float = 2.0,
    antislip_peak_slip_ratio: float = 0.95,
    antislip_min_peak_support: float = 100.0,
    antislip_nn: bool = False,
    nn_detector: SlipNeuralDetector | None = None,
    policy2_controller: Policy2OpenLoopController | None = None,
    dataset_logger: SlipDatasetLogger | None = None,
    feature_builder: SlipFeatureBuilder | None = None,
    dataset_case_name: str = "",
) -> ReplayResult:
    if not cfg.scene_path.exists():
        raise FileNotFoundError(f"Missing scene: {cfg.scene_path}")
    if not cfg.trajectory_path.exists() or cfg.trajectory_path.stat().st_size < 1000:
        raise FileNotFoundError(
            f"Missing trajectory (run setup_spider.sh + git lfs checkout): {cfg.trajectory_path}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(cfg.scene_path))
    physics_meta = apply_object_physics(
        model, mass_scale=mass_scale, friction_scale=friction_scale, object_body=object_body
    )
    model.opt.timestep = sim_dt
    data = mujoco.MjData(model)

    qpos_ref, qvel_ref, ctrl_ref = load_trajectory_arrays(cfg.trajectory_path, model, cfg.data_type)
    qpos_ref, qvel_ref, ctrl_ref = upsample_controls(qpos_ref, qvel_ref, ctrl_ref, sim_dt, ref_dt)

    hand_geoms = get_spider_hand_collision_geom_ids(model)
    object_geoms = get_object_geom_ids(model, object_body)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    g_hat = gravity_up(model)

    support_detector: VerticalSupportAntislipDetector | None = None
    grip_controller: GripBoostController | None = None
    nn_policy2_ctrl: Policy2OpenLoopController | None = None
    use_policy2 = policy2_controller is not None
    use_nn = bool(antislip_nn and nn_detector is not None and not use_policy2)
    use_nn_policy2 = bool(
        use_nn
        and nn_detector is not None
        and (
            getattr(nn_detector, "arch", "") == "detect_and_policy2"
            or str(getattr(nn_detector, "policy_mode", "")) == "p2a"
        )
    )
    use_scheme2 = bool(antislip and antislip_scheme == 2 and not use_nn and not use_policy2)
    if use_scheme2:
        support_detector = VerticalSupportAntislipDetector(
            antislip_avg_window_s,
            sim_dt,
            smooth_window_s=antislip_smooth_window_s,
            slip_ratio=antislip_slip_ratio,
            peak_slip_ratio=antislip_peak_slip_ratio,
            min_peak_support=antislip_min_peak_support,
        )
        grip_controller = GripBoostController(
            step_boost=antislip_grip_step,
            max_extra=antislip_grip_max,
        )
    elif use_nn:
        if feature_builder is None:
            feature_builder = SlipFeatureBuilder(sim_dt=sim_dt)
        if use_nn_policy2 and nn_detector is not None:
            d_max = float(getattr(nn_detector, "max_wrist", 0.25))
            nn_policy2_ctrl = Policy2OpenLoopController(
                Policy2Action(),
                g_max=float(antislip_grip_max),
                d_max=d_max,
            )
        else:
            grip_controller = GripBoostController(
                step_boost=antislip_grip_step,
                max_extra=antislip_grip_max,
            )
    elif use_policy2 and policy2_controller is not None:
        policy2_controller.reset()

    lift_start: int | None = None
    grasp_report: GraspPhysicsReport | None = None
    use_reset_lift = post_lift_m > 0 and post_extend_s <= 0
    if use_reset_lift:
        lift_start, _ = find_grasp_frame(
            model, data, qpos_ref, qvel_ref, ctrl_ref, hand_geoms, object_geoms, object_body
        )

    data.qpos[:] = qpos_ref[0]
    data.qvel[:] = qvel_ref[0]
    data.ctrl[:] = ctrl_ref[0]
    mujoco.mj_forward(model, data)
    z_start = float(data.xpos[object_id][2])

    detector = SlipDetector(window_size=30, threshold=0.15)
    log = _EnergyLog()
    slip_counter = [0]
    contact_steps = 0
    phase: list[str] = []
    log_every = 5 if log_energy else 0

    video_recorder: VideoRecorder | None = None
    if save_video:
        model.vis.global_.offwidth = 1280
        model.vis.global_.offheight = 720
        tag_preview = f"{cfg.dataset_name}_{cfg.robot_type}_{cfg.embodiment_type}_{cfg.task}"
        video_recorder = VideoRecorder(
            model,
            out_dir / f"{tag_preview}_replay.mp4",
            width=1280,
            height=720,
            fps=video_fps,
            timestep=sim_dt,
        )
        video_recorder.camera = make_spider_ketchup_camera(model, object_body)

    global_step = 0
    object_z_extend_start: float | None = None
    object_z_traj_end = z_start

    if feature_builder is None and dataset_logger is not None:
        feature_builder = SlipFeatureBuilder(sim_dt=sim_dt)
    if feature_builder is not None:
        feature_builder.reset_trajectory(z_start)

    def _maybe_log_dataset(phase_name: str, wrist_tz: float, grip_extra: float) -> None:
        if dataset_logger is None or feature_builder is None:
            return
        ctx = make_step_context(
            phase=phase_name,
            wrist_tz=wrist_tz,
            grip_extra=grip_extra,
            friction_scale=friction_scale,
            object_z=float(data.xpos[object_id][2]),
            object_z_traj_end=object_z_traj_end,
            object_z_extend_start=object_z_extend_start,
            object_z_start=z_start,
            in_trajectory=phase_name == "trajectory",
        )
        reading = feature_builder.build(
            model, data, hand_geoms, object_geoms, object_id, ctx
        )
        dataset_logger.append(
            reading.features,
            reading.labels,
            SlipDatasetMeta(
                step=global_step,
                sim_time=float(data.time),
                phase=phase_name,
                friction_scale=friction_scale,
                mass_scale=mass_scale,
                case_name=dataset_case_name,
                object_z=float(data.xpos[object_id][2]),
            ),
        )

    def _maybe_warm_nn_features(phase_name: str, wrist_tz: float, grip_extra: float) -> None:
        """Advance SlipFeatureBuilder during trajectory when NN antislip is enabled."""
        if not use_nn or feature_builder is None or dataset_logger is not None:
            return
        if phase_name != "trajectory":
            return
        ctx = make_step_context(
            phase=phase_name,
            wrist_tz=wrist_tz,
            grip_extra=grip_extra,
            friction_scale=friction_scale,
            object_z=float(data.xpos[object_id][2]),
            object_z_traj_end=object_z_traj_end,
            object_z_extend_start=object_z_extend_start,
            object_z_start=z_start,
            in_trajectory=True,
        )
        feature_builder.build(model, data, hand_geoms, object_geoms, object_id, ctx)

    def record_frame() -> None:
        if video_recorder is not None:
            video_recorder.maybe_capture(data, global_step)

    def run_ctrl(ctrl: np.ndarray, n_steps: int, phase_name: str, step_log_every: int = 5) -> None:
        nonlocal global_step, contact_steps
        for _ in range(n_steps):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            m = _log_energy_step(model, data, hand_geoms, object_geoms, object_id, detector, slip_counter)
            if m.n_contacts > 0:
                contact_steps += 1
            if step_log_every > 0 and global_step % step_log_every == 0:
                log.step.append(global_step)
                log.sim_time.append(float(data.time))
                log.n_contacts.append(m.n_contacts)
                log.mass_estimate.append(m.mass_estimate)
                log.slip.append(m.slipped)
                log.object_z.append(float(data.xpos[object_id][2]))
                phase.append(phase_name)
            record_frame()
            global_step += 1

    for ctrl in ctrl_ref:
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        if support_detector is not None:
            vs = measure_vertical_support(
                model, data, hand_geoms, object_geoms, object_body, g_hat=g_hat
            )
            support_detector.update(vs.support_z)
        m = _log_energy_step(model, data, hand_geoms, object_geoms, object_id, detector, slip_counter)
        if m.n_contacts > 0:
            contact_steps += 1
        _maybe_log_dataset("trajectory", float(ctrl[arm_tz_index]), 0.0)
        _maybe_warm_nn_features("trajectory", float(ctrl[arm_tz_index]), 0.0)
        if log_every > 0 and global_step % log_every == 0:
            log.step.append(global_step)
            log.sim_time.append(float(data.time))
            log.n_contacts.append(m.n_contacts)
            log.mass_estimate.append(m.mass_estimate)
            log.slip.append(m.slipped)
            log.object_z.append(float(data.xpos[object_id][2]))
            phase.append("trajectory")
        record_frame()
        global_step += 1

    z_after_traj = float(data.xpos[object_id][2])
    object_z_traj_end = z_after_traj
    if feature_builder is not None:
        feature_builder.mark_trajectory_end(z_after_traj)
    post_lift_dz = 0.0
    post_lift_contact_steps = 0
    post_extend_object_dz = 0.0
    post_extend_contact_steps = 0
    center_slip_events = 0
    support_slip_events = 0
    nn_slip_events = 0
    antislip_max_grip = 0.0
    if use_nn:
        active_antislip_scheme = 3
    elif antislip:
        active_antislip_scheme = antislip_scheme
    else:
        active_antislip_scheme = 0

    if post_extend_s > 0:
        z_extend_start = float(data.xpos[object_id][2])
        object_z_extend_start = z_extend_start
        if feature_builder is not None:
            feature_builder.reset_extend(z_extend_start)
        if use_nn and nn_detector is not None:
            nn_detector.reset_extend()
        extend_ctrl = build_extend_mimic_lift_controls(
            ctrl_ref,
            sim_dt=sim_dt,
            extend_s=post_extend_s,
            mimic_s=post_mimic_s,
            lift_m=post_lift_m,
            arm_tz_index=arm_tz_index,
        )
        if grip_controller is not None:
            grip_controller.reset()
        if use_policy2 and policy2_controller is not None:
            policy2_controller.reset()
        if nn_policy2_ctrl is not None:
            nn_policy2_ctrl.reset()
        if support_detector is not None:
            support_detector.reset_peak()

        for ctrl in extend_ctrl:
            phase_name = "extend_mimic_lift"
            applied_ctrl = ctrl
            if use_policy2 and policy2_controller is not None:
                applied_ctrl = policy2_controller.apply(ctrl, model)
                antislip_max_grip = max(antislip_max_grip, float(policy2_controller.grip_extra))
                phase_name = "extend_policy2"
            elif support_detector is not None and grip_controller is not None:
                vs = measure_vertical_support(
                    model, data, hand_geoms, object_geoms, object_body, g_hat=g_hat
                )
                reading = support_detector.update(vs.support_z)
                if reading.slip_active:
                    support_slip_events += 1
                    grip_controller.on_slip()
                    phase_name = "extend_antislip"
                applied_ctrl = grip_controller.apply(ctrl, model)
                antislip_max_grip = max(antislip_max_grip, grip_controller.grip_extra)
            elif (
                use_nn
                and nn_detector is not None
                and feature_builder is not None
                and (grip_controller is not None or nn_policy2_ctrl is not None)
            ):
                grip_extra_feat = (
                    float(nn_policy2_ctrl.grip_extra)
                    if nn_policy2_ctrl is not None
                    else float(grip_controller.grip_extra)  # type: ignore[union-attr]
                )
                ctx = make_step_context(
                    phase="extend",
                    wrist_tz=float(ctrl[arm_tz_index]),
                    grip_extra=grip_extra_feat,
                    friction_scale=friction_scale,
                    object_z=float(data.xpos[object_id][2]),
                    object_z_traj_end=object_z_traj_end,
                    object_z_extend_start=object_z_extend_start,
                    object_z_start=z_start,
                    in_trajectory=False,
                )
                feat_reading = feature_builder.build(
                    model, data, hand_geoms, object_geoms, object_id, ctx
                )
                nn_reading = nn_detector.update(feat_reading.features)
                # Soft preempt: apply grip before hard confirm (does not increment nn_slip_events).
                soft_thr = float(getattr(nn_detector, "soft_threshold", 1.01))
                if nn_reading.p_slip >= soft_thr:
                    scale = float(getattr(nn_detector, "soft_grip_scale", 1.0))
                    if hasattr(nn_detector, "resolve_grip"):
                        soft_g = nn_detector.resolve_grip(nn_reading)
                    elif getattr(nn_detector, "use_grip_head", False):
                        soft_g = nn_reading.delta_grip
                    else:
                        soft_g = None
                    if soft_g is not None:
                        if nn_policy2_ctrl is not None:
                            wrist = nn_detector.resolve_wrist(nn_reading)
                            nn_policy2_ctrl.set_action(
                                Policy2Action(
                                    grip=float(soft_g) * scale,
                                    wrist_delta=wrist,
                                )
                            )
                        elif grip_controller is not None:
                            grip_controller.set_grip(float(soft_g) * scale)
                if nn_reading.slip_now:
                    nn_slip_events += 1
                if nn_reading.slip_active:
                    if hasattr(nn_detector, "resolve_grip"):
                        hard_g = nn_detector.resolve_grip(nn_reading)
                    elif (
                        getattr(nn_detector, "use_grip_head", False)
                        and nn_reading.delta_grip is not None
                    ):
                        hard_g = nn_reading.delta_grip
                    else:
                        hard_g = None
                    if hard_g is not None:
                        if nn_policy2_ctrl is not None:
                            wrist = nn_detector.resolve_wrist(nn_reading)
                            nn_policy2_ctrl.set_action(
                                Policy2Action(grip=float(hard_g), wrist_delta=wrist)
                            )
                        elif grip_controller is not None:
                            grip_controller.set_grip(float(hard_g))
                    elif grip_controller is not None:
                        grip_controller.on_slip()
                    phase_name = (
                        "extend_antislip_nn_p2"
                        if nn_policy2_ctrl is not None
                        else "extend_antislip_nn"
                    )
                if nn_policy2_ctrl is not None:
                    applied_ctrl = nn_policy2_ctrl.apply(ctrl, model)
                    antislip_max_grip = max(
                        antislip_max_grip, float(nn_policy2_ctrl.grip_extra)
                    )
                elif grip_controller is not None:
                    applied_ctrl = grip_controller.apply(ctrl, model)
                    antislip_max_grip = max(antislip_max_grip, grip_controller.grip_extra)

            data.ctrl[:] = applied_ctrl
            mujoco.mj_step(model, data)
            m = _log_energy_step(
                model, data, hand_geoms, object_geoms, object_id, detector, slip_counter
            )
            if m.n_contacts > 0:
                contact_steps += 1
                post_extend_contact_steps += 1
            if grip_controller is not None:
                grip_extra = grip_controller.grip_extra
            elif nn_policy2_ctrl is not None:
                grip_extra = float(nn_policy2_ctrl.grip_extra)
            elif use_policy2 and policy2_controller is not None:
                grip_extra = float(policy2_controller.grip_extra)
            else:
                grip_extra = 0.0
            _maybe_log_dataset(phase_name, float(ctrl[arm_tz_index]), grip_extra)
            if log_every > 0 and global_step % log_every == 0:
                log.step.append(global_step)
                log.sim_time.append(float(data.time))
                log.n_contacts.append(m.n_contacts)
                log.mass_estimate.append(m.mass_estimate)
                log.slip.append(m.slipped)
                log.object_z.append(float(data.xpos[object_id][2]))
                phase.append(phase_name)
            record_frame()
            global_step += 1
        post_extend_object_dz = float(data.xpos[object_id][2]) - z_extend_start
        post_lift_dz = post_extend_object_dz

    elif use_reset_lift and lift_start is not None:
        # Branch: reset to best grasp pose, validate, then physics-only lift
        data.qpos[:] = qpos_ref[0]
        data.qvel[:] = qvel_ref[0]
        for c in ctrl_ref[: lift_start + 1]:
            data.ctrl[:] = c
            mujoco.mj_step(model, data)

        hold_ctrl = data.ctrl.copy()
        grasp_report = validate_grasp_physics(
            model, data, object_body, hand_geoms, object_geoms
        )

        if grasp_report.ok:
            run_ctrl(hold_ctrl, post_hold_steps, "hold")
            z_lift_start = float(data.xpos[object_id][2])
            tz0 = float(hold_ctrl[arm_tz_index])
            lift_contact = 0
            for i in range(post_lift_steps):
                alpha = (i + 1) / post_lift_steps
                lift_ctrl = hold_ctrl.copy()
                lift_ctrl[arm_tz_index] = tz0 + post_lift_m * alpha
                data.ctrl[:] = lift_ctrl
                mujoco.mj_step(model, data)
                m = _log_energy_step(
                    model, data, hand_geoms, object_geoms, object_id, detector, slip_counter
                )
                if m.n_contacts > 0:
                    contact_steps += 1
                    lift_contact += 1
                if log_every > 0 and global_step % log_every == 0:
                    log.step.append(global_step)
                    log.sim_time.append(float(data.time))
                    log.n_contacts.append(m.n_contacts)
                    log.mass_estimate.append(m.mass_estimate)
                    log.slip.append(m.slipped)
                    log.object_z.append(float(data.xpos[object_id][2]))
                    phase.append("lift")
                record_frame()
                global_step += 1

            post_lift_dz = float(data.xpos[object_id][2]) - z_lift_start
            post_lift_contact_steps = lift_contact

            hold_up = hold_ctrl.copy()
            hold_up[arm_tz_index] = tz0 + post_lift_m
            run_ctrl(hold_up, 40, "hold_up")
        else:
            phase.append("lift_skipped")

    z_end = float(data.xpos[object_id][2])
    slip_events = slip_counter[0]
    tag = f"{cfg.dataset_name}_{cfg.robot_type}_{cfg.embodiment_type}_{cfg.task}"
    log_path = None
    if log_energy:
        log_path = out_dir / f"{tag}_energy.json"
        log_path.write_text(
            json.dumps(
                {
                    "task": cfg.task,
                    "dataset": cfg.dataset_name,
                    "data_type": cfg.data_type,
                    "physics": physics_meta,
                    "steps": global_step,
                    "contact_steps": contact_steps,
                    "slip_events": slip_events,
                    "object_z_start": z_start,
                    "object_z_after_trajectory": z_after_traj,
                    "object_z_end": z_end,
                    "object_dz": z_end - z_start,
                    "post_lift_m": post_lift_m,
                    "post_lift_dz": post_lift_dz,
                    "post_lift_contact_steps": post_lift_contact_steps,
                    "post_extend_s": post_extend_s,
                    "post_mimic_s": post_mimic_s,
                    "post_extend_object_dz": post_extend_object_dz,
                    "post_extend_contact_steps": post_extend_contact_steps,
                    "center_slip_events": center_slip_events,
                    "support_slip_events": support_slip_events,
                    "nn_slip_events": nn_slip_events,
                    "antislip": antislip or use_nn,
                    "antislip_scheme": active_antislip_scheme,
                    "antislip_max_grip": antislip_max_grip,
                    "lift_start_frame": lift_start,
                    "grasp_physics_ok": grasp_report.ok if grasp_report else None,
                    "grasp_fail_reasons": grasp_report.reasons if grasp_report else [],
                    "series": {
                        "step": log.step,
                        "sim_time": log.sim_time,
                        "n_contacts": log.n_contacts,
                        "mass_estimate": log.mass_estimate,
                        "slip": log.slip,
                        "object_z": log.object_z,
                        "phase": phase,
                    },
                },
                indent=2,
            )
        )

    video_path = None
    if video_recorder is not None:
        suffix = ""
        if post_extend_s > 0:
            suffix = f"_extend{post_extend_s:.0f}s_lift{int(post_lift_m * 100)}cm"
        elif post_lift_m > 0:
            suffix = f"_lift{int(post_lift_m * 100)}cm"
        final_path = out_dir / f"{tag}_replay{suffix}.mp4"
        video_recorder.output_path = final_path
        video_path = video_recorder.save()
        video_recorder.close()

    return ReplayResult(
        steps=global_step,
        contact_steps=contact_steps,
        slip_events=slip_events,
        object_z_start=z_start,
        object_z_end=z_end,
        object_dz=z_end - z_start,
        post_lift_dz=post_lift_dz,
        post_lift_contact_steps=post_lift_contact_steps,
        post_extend_s=post_extend_s,
        post_extend_object_dz=post_extend_object_dz,
        post_extend_contact_steps=post_extend_contact_steps,
        object_z_after_trajectory=z_after_traj,
        grasp_physics_ok=grasp_report.ok if grasp_report else False,
        grasp_report=grasp_report,
        log_path=log_path,
        video_path=video_path,
        physics_meta=physics_meta,
        center_slip_events=center_slip_events,
        support_slip_events=support_slip_events,
        nn_slip_events=nn_slip_events,
        antislip_max_grip=antislip_max_grip,
        antislip_scheme=active_antislip_scheme,
    )
