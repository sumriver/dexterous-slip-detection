"""Replay SPIDER precomputed trajectories in MuJoCo (CPU) with optional energy-flow logging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from energy_flow import SlipDetector, compute_applied_power, compute_mass_estimate
from energy_flow.state import compute_retained_power


@dataclass
class SpiderTaskConfig:
    dataset_dir: Path
    dataset_name: str = "oakinkv2"
    robot_type: str = "xhand"
    embodiment_type: str = "right"
    task: str = "pick_spoon_bowl"
    data_id: int = 0
    data_type: str = "mjwp_fast"

    @property
    def task_dir(self) -> Path:
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
    log_path: Path | None = None
    video_path: Path | None = None


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


def find_grasp_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_ref: np.ndarray,
    qvel_ref: np.ndarray,
    ctrl_ref: np.ndarray,
    hand_geoms: set[int],
    object_geoms: set[int],
    object_id: int,
    search_frac: float = 0.8,
    min_contacts: int = 2,
) -> int:
    """Frame index with strong grasp before place-in-bowl phase (first search_frac of traj)."""
    obj_q = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint")
    ]
    data.qpos[:] = qpos_ref[0]
    data.qvel[:] = qvel_ref[0]
    data.ctrl[:] = ctrl_ref[0]
    mujoco.mj_forward(model, data)

    contacts: list[int] = []
    obj_z: list[float] = []
    limit = max(1, int(search_frac * len(ctrl_ref)))
    for i in range(limit):
        data.ctrl[:] = ctrl_ref[i]
        mujoco.mj_step(model, data)
        forces, _, _ = extract_hand_object_contacts(model, data, hand_geoms, object_geoms)
        contacts.append(len(forces))
        obj_z.append(float(data.qpos[obj_q + 2]))

    contacts_a = np.array(contacts)
    obj_z_a = np.array(obj_z)
    valid = np.where(contacts_a >= min_contacts)[0]
    if len(valid) == 0:
        return int(np.argmax(contacts_a))
    return int(valid[np.argmax(obj_z_a[valid])])


def _apply_grasp_sync_lift(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_id: int,
    z_grasp: float,
    xy_grasp: np.ndarray,
    quat_grasp: np.ndarray,
    lift_alpha: float,
    lift_m: float,
) -> None:
    """Keep object pose from grasp; raise Z with arm during confirmed hold."""
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint")
    obj_q = model.jnt_qposadr[jnt]
    obj_v = model.jnt_dofadr[jnt]
    data.qpos[obj_q : obj_q + 2] = xy_grasp
    data.qpos[obj_q + 2] = z_grasp + lift_m * lift_alpha
    data.qpos[obj_q + 3 : obj_q + 7] = quat_grasp
    data.qvel[obj_v : obj_v + 6] = 0
    mujoco.mj_forward(model, data)


def replay_spider_task(
    cfg: SpiderTaskConfig,
    out_dir: Path,
    *,
    sim_dt: float = 0.01,
    ref_dt: float = 0.02,
    save_video: bool = True,
    video_fps: int = 50,
    object_body: str = "right_object",
    post_lift_m: float = 0.0,
    post_hold_steps: int = 60,
    post_lift_steps: int = 150,
    arm_tz_index: int = 2,
    lift_mode: str = "grasp_sync",
) -> ReplayResult:
    if not cfg.scene_path.exists():
        raise FileNotFoundError(f"Missing scene: {cfg.scene_path}")
    if not cfg.trajectory_path.exists() or cfg.trajectory_path.stat().st_size < 1000:
        raise FileNotFoundError(
            f"Missing trajectory (run setup_spider.sh + git lfs checkout): {cfg.trajectory_path}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(cfg.scene_path))
    model.opt.timestep = sim_dt
    data = mujoco.MjData(model)

    qpos_ref, qvel_ref, ctrl_ref = load_trajectory_arrays(cfg.trajectory_path, model, cfg.data_type)
    qpos_ref, qvel_ref, ctrl_ref = upsample_controls(qpos_ref, qvel_ref, ctrl_ref, sim_dt, ref_dt)

    hand_geoms = get_spider_hand_collision_geom_ids(model)
    object_geoms = get_object_geom_ids(model, object_body)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    obj_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint")
    obj_qadr = model.jnt_qposadr[obj_jnt]

    lift_start: int | None = None
    if post_lift_m > 0:
        lift_start = find_grasp_frame(
            model, data, qpos_ref, qvel_ref, ctrl_ref, hand_geoms, object_geoms, object_id
        )
        ctrl_ref = ctrl_ref[: lift_start + 1]
        qpos_ref = qpos_ref[: lift_start + 1]
        qvel_ref = qvel_ref[: lift_start + 1]

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

    renderer = None
    frames: list[np.ndarray] = []
    if save_video:
        model.vis.global_.offwidth = 720
        model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(model, height=480, width=720)

    global_step = 0

    def record_frame() -> None:
        if renderer is not None and global_step % 2 == 0:
            renderer.update_scene(data, "front")
            frames.append(renderer.render().copy())

    def run_ctrl(ctrl: np.ndarray, n_steps: int, phase_name: str, log_every: int = 5) -> None:
        nonlocal global_step, contact_steps
        for _ in range(n_steps):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            m = _log_energy_step(model, data, hand_geoms, object_geoms, object_id, detector, slip_counter)
            if m.n_contacts > 0:
                contact_steps += 1
            if global_step % log_every == 0:
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
        m = _log_energy_step(model, data, hand_geoms, object_geoms, object_id, detector, slip_counter)
        if m.n_contacts > 0:
            contact_steps += 1
        if global_step % 5 == 0:
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
    post_lift_dz = 0.0
    post_lift_contact_steps = 0

    if post_lift_m > 0:
        hold_ctrl = data.ctrl.copy()
        m_hold = _log_energy_step(model, data, hand_geoms, object_geoms, object_id, detector, slip_counter)
        if m_hold.n_contacts < 2:
            raise RuntimeError(
                f"Grasp too weak at lift start ({m_hold.n_contacts} contacts). "
                "Cannot lift — check trajectory or grasp frame."
            )

        xy_grasp = data.qpos[obj_qadr : obj_qadr + 2].copy()
        z_grasp = float(data.qpos[obj_qadr + 2])
        quat_grasp = data.qpos[obj_qadr + 3 : obj_qadr + 7].copy()

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
            if lift_mode == "grasp_sync":
                _apply_grasp_sync_lift(
                    model, data, object_id, z_grasp, xy_grasp, quat_grasp, alpha, post_lift_m
                )
            m = _log_energy_step(model, data, hand_geoms, object_geoms, object_id, detector, slip_counter)
            if m.n_contacts > 0:
                contact_steps += 1
                lift_contact += 1
            if global_step % 5 == 0:
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
        for _ in range(40):
            data.ctrl[:] = hold_up
            mujoco.mj_step(model, data)
            if lift_mode == "grasp_sync":
                _apply_grasp_sync_lift(
                    model, data, object_id, z_grasp, xy_grasp, quat_grasp, 1.0, post_lift_m
                )
            m = _log_energy_step(model, data, hand_geoms, object_geoms, object_id, detector, slip_counter)
            if m.n_contacts > 0:
                contact_steps += 1
            if global_step % 5 == 0:
                log.step.append(global_step)
                log.sim_time.append(float(data.time))
                log.n_contacts.append(m.n_contacts)
                log.mass_estimate.append(m.mass_estimate)
                log.slip.append(m.slipped)
                log.object_z.append(float(data.xpos[object_id][2]))
                phase.append("hold_up")
            record_frame()
            global_step += 1

    z_end = float(data.xpos[object_id][2])
    slip_events = slip_counter[0]
    tag = f"{cfg.dataset_name}_{cfg.robot_type}_{cfg.embodiment_type}_{cfg.task}"
    log_path = out_dir / f"{tag}_energy.json"
    log_path.write_text(
        json.dumps(
            {
                "task": cfg.task,
                "dataset": cfg.dataset_name,
                "data_type": cfg.data_type,
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
                "lift_start_frame": lift_start,
                "lift_mode": lift_mode if post_lift_m > 0 else None,
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
    if renderer is not None and frames:
        suffix = f"_lift{int(post_lift_m * 100)}cm" if post_lift_m > 0 else ""
        video_path = out_dir / f"{tag}_replay{suffix}.mp4"
        iio.imwrite(video_path, np.stack(frames), fps=video_fps, codec="libx264", pixelformat="yuv420p")
        renderer.close()

    return ReplayResult(
        steps=global_step,
        contact_steps=contact_steps,
        slip_events=slip_events,
        object_z_start=z_start,
        object_z_end=z_end,
        object_dz=z_end - z_start,
        post_lift_dz=post_lift_dz,
        post_lift_contact_steps=post_lift_contact_steps,
        log_path=log_path,
        video_path=video_path,
    )
