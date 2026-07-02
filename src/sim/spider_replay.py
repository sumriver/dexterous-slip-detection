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
    log_path: Path | None = None
    video_path: Path | None = None


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


def replay_spider_task(
    cfg: SpiderTaskConfig,
    out_dir: Path,
    *,
    sim_dt: float = 0.01,
    ref_dt: float = 0.02,
    save_video: bool = True,
    video_fps: int = 50,
    object_body: str = "right_object",
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

    data.qpos[:] = qpos_ref[0]
    data.qvel[:] = qvel_ref[0]
    data.ctrl[:] = ctrl_ref[0]
    mujoco.mj_forward(model, data)
    z_start = float(data.xpos[object_id][2])

    detector = SlipDetector(window_size=30, threshold=0.15)
    log = _EnergyLog()
    slip_events = 0
    contact_steps = 0

    renderer = None
    frames: list[np.ndarray] = []
    if save_video:
        model.vis.global_.offwidth = 720
        model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(model, height=480, width=720)

    for step, ctrl in enumerate(ctrl_ref):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        forces, _, velocities = extract_hand_object_contacts(model, data, hand_geoms, object_geoms)
        n_con = len(forces)
        mass_est = float("nan")
        slipped = False

        if n_con > 0:
            contact_steps += 1
            applied = compute_applied_power(forces, velocities)
            total_force = np.sum(forces, axis=0)
            obj_vel = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, object_id, obj_vel, 0
            )
            retained = compute_retained_power(total_force, obj_vel[:3])
            mass_est = compute_mass_estimate(applied, retained)
            slipped = detector.update(mass_est)
            if slipped:
                slip_events += 1

        if step % 5 == 0:
            log.step.append(step)
            log.sim_time.append(float(data.time))
            log.n_contacts.append(n_con)
            log.mass_estimate.append(float(mass_est))
            log.slip.append(bool(slipped))
            log.object_z.append(float(data.xpos[object_id][2]))

        if renderer is not None and step % 2 == 0:
            renderer.update_scene(data, "front")
            frames.append(renderer.render().copy())

    z_end = float(data.xpos[object_id][2])
    tag = f"{cfg.dataset_name}_{cfg.robot_type}_{cfg.embodiment_type}_{cfg.task}"
    log_path = out_dir / f"{tag}_energy.json"
    log_path.write_text(
        json.dumps(
            {
                "task": cfg.task,
                "dataset": cfg.dataset_name,
                "data_type": cfg.data_type,
                "steps": len(ctrl_ref),
                "contact_steps": contact_steps,
                "slip_events": slip_events,
                "object_z_start": z_start,
                "object_z_end": z_end,
                "object_dz": z_end - z_start,
                "series": {
                    "step": log.step,
                    "sim_time": log.sim_time,
                    "n_contacts": log.n_contacts,
                    "mass_estimate": log.mass_estimate,
                    "slip": log.slip,
                    "object_z": log.object_z,
                },
            },
            indent=2,
        )
    )

    video_path = None
    if renderer is not None and frames:
        video_path = out_dir / f"{tag}_replay.mp4"
        iio.imwrite(video_path, np.stack(frames), fps=video_fps, codec="libx264", pixelformat="yuv420p")
        renderer.close()

    return ReplayResult(
        steps=len(ctrl_ref),
        contact_steps=contact_steps,
        slip_events=slip_events,
        object_z_start=z_start,
        object_z_end=z_end,
        object_dz=z_end - z_start,
        log_path=log_path,
        video_path=video_path,
    )
