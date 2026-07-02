"""Runtime tweaks to SPIDER ketchup object mass and contact friction."""

from __future__ import annotations

import mujoco


def _object_pair_indices(model: mujoco.MjModel, object_body: str = "right_object") -> list[int]:
    """Pair indices where at least one geom belongs to object_body."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if body_id < 0:
        return []
    object_geoms = {gid for gid in range(model.ngeom) if model.geom_bodyid[gid] == body_id}
    pairs: list[int] = []
    for pid in range(model.npair):
        if model.pair_geom1[pid] in object_geoms or model.pair_geom2[pid] in object_geoms:
            pairs.append(pid)
    return pairs


def scale_object_mass(model: mujoco.MjModel, scale: float, object_body: str = "right_object") -> float:
    """Scale object body mass and inertia in-place. Returns original mass (kg)."""
    if scale == 1.0:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        return float(model.body_mass[body_id])

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if body_id < 0:
        raise ValueError(f"Body not found: {object_body}")
    original = float(model.body_mass[body_id])
    model.body_mass[body_id] = original * scale
    model.body_inertia[body_id] = model.body_inertia[body_id] * scale
    return original


def scale_object_friction(
    model: mujoco.MjModel,
    scale: float,
    object_body: str = "right_object",
) -> int:
    """Scale tangential friction on all pairs involving the object. Returns pair count."""
    if scale == 1.0:
        return len(_object_pair_indices(model, object_body))

    count = 0
    for pid in _object_pair_indices(model, object_body):
        model.pair_friction[pid, :3] *= scale
        count += 1
    return count


def apply_object_physics(
    model: mujoco.MjModel,
    *,
    mass_scale: float = 1.0,
    friction_scale: float = 1.0,
    object_body: str = "right_object",
) -> dict[str, float | int]:
    """Apply mass and friction scaling; return summary for logging."""
    base_mass = scale_object_mass(model, mass_scale, object_body)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    n_pairs = scale_object_friction(model, friction_scale, object_body)
    return {
        "base_mass_kg": base_mass,
        "mass_kg": float(model.body_mass[body_id]),
        "mass_scale": mass_scale,
        "friction_scale": friction_scale,
        "object_pair_count": n_pairs,
    }
