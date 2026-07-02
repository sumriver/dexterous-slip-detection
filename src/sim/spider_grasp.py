"""SPIDER-style XHAND scene helpers (collision geoms + metrics)."""

from __future__ import annotations

import mujoco
import numpy as np

from sim.grasp_physics import measure_bottle_grasp, required_support_force


def get_spider_collision_geom_ids(model: mujoco.MjModel) -> set[int]:
    ids: set[int] = set()
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("collision_hand_right_"):
            ids.add(gid)
    return ids


def settle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: np.ndarray,
    steps: int = 2000,
) -> bool:
    """Apply ctrl via position actuators and step. Returns False if unstable."""
    data.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qacc).all():
            return False
    return True


def count_bottle_floor_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    bottle_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_geom")
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    n = 0
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if (g1 == bottle_geom and g2 == floor_geom) or (g2 == bottle_geom and g1 == floor_geom):
            n += 1
    return n


def evaluate_lift_hold(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: np.ndarray,
    hand_geoms: set[int],
    lift_dz: float = 0.05,
    lift_steps: int = 120,
    settle_steps: int = 2000,
) -> tuple[float, float, int, float]:
    """Settle grasp, ramp tz by lift_dz. Returns (bottle_dz, support_end, contacts, bottle_z0)."""
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    mujoco.mj_resetData(model, data)
    if not settle(model, data, ctrl, steps=settle_steps):
        return 0.0, 0.0, 0, 0.0

    z0 = float(data.xpos[bottle_id][2])
    tz0 = float(ctrl[2])
    trial = ctrl.copy()
    for i in range(lift_steps):
        alpha = (i + 1) / lift_steps
        trial[2] = tz0 + lift_dz * alpha
        data.ctrl[:] = trial
        for _ in range(8):
            mujoco.mj_step(model, data)

    m = measure_bottle_grasp(model, data, hand_geoms)
    dz = float(data.xpos[bottle_id][2]) - z0
    return dz, m.support_force_z, m.n_contacts, z0


def evaluate_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: np.ndarray,
    hand_geoms: set[int],
    mg: float,
    settle_steps: int = 2000,
    lift_probe: bool = False,
) -> tuple[float, dict[str, float]]:
    """Score grasp: prioritize support force >= mg, then contacts."""
    mujoco.mj_resetData(model, data)
    if not settle(model, data, ctrl, steps=settle_steps):
        return -1e6, {"stable": 0.0}

    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    z0 = float(data.xpos[bottle_id][2])
    m = measure_bottle_grasp(model, data, hand_geoms)
    floor_contacts = count_bottle_floor_contacts(model, data)

    score = m.support_force_z * 10.0 + m.n_contacts * 1.0
    if m.support_force_z >= mg:
        score += 50.0
    elif m.support_force_z >= 0.5 * mg:
        score += 10.0
    if m.n_contacts >= 3:
        score += 5.0
    if floor_contacts == 0 and z0 > 0.028:
        score += 15.0
    else:
        score -= floor_contacts * 3.0
    xy_err = float(np.linalg.norm(data.xpos[bottle_id][:2] - np.array([0.55, 0.0])))
    score -= xy_err * 10.0

    info: dict[str, float] = {
        "stable": 1.0,
        "n_contacts": float(m.n_contacts),
        "support_z": m.support_force_z,
        "bottle_z": z0,
        "floor_contacts": float(floor_contacts),
        "xy_err": xy_err,
        "lift_probe_dz": 0.0,
        "lift_probe_support": 0.0,
    }

    if lift_probe:
        dz_p, sup_p, nc_p, _ = evaluate_lift_hold(
            model, data, ctrl, hand_geoms, lift_dz=0.05, settle_steps=settle_steps
        )
        info["lift_probe_dz"] = dz_p
        info["lift_probe_support"] = sup_p
        info["lift_probe_contacts"] = float(nc_p)
        score += dz_p * 200.0
        score += sup_p * 5.0
        score += nc_p * 2.0
        if dz_p >= 0.03:
            score += 30.0

    return score, info


def simulate_lift(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl_grasp: np.ndarray,
    lift_dz: float = 0.20,
    steps: int = 500,
) -> tuple[float, float, int]:
    """Ramp tz ctrl while holding grasp. Returns (bottle_dz, final_support, final_contacts)."""
    dz, sup, nc, _ = evaluate_lift_hold(
        model, data, ctrl_grasp, get_spider_collision_geom_ids(model),
        lift_dz=lift_dz, lift_steps=steps, settle_steps=2500,
    )
    return dz, sup, nc
