"""MuJoCo contact extraction utilities."""

from __future__ import annotations

import mujoco
import numpy as np


def extract_hand_contacts(model, data, hand_geom_ids: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract contact forces, positions, and velocities for hand geoms.

    Returns:
        forces: (n, 3) contact forces in world frame
        positions: (n, 3) contact positions in world frame
        velocities: (n, 3) contact point velocities (approximated from body vel)
    """
    forces_list: list[np.ndarray] = []
    pos_list: list[np.ndarray] = []
    vel_list: list[np.ndarray] = []

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if g1 not in hand_geom_ids and g2 not in hand_geom_ids:
            continue

        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        # contact.frame is 3x3 rotation matrix (contact -> world)
        frame = np.array(contact.frame, dtype=float).reshape(3, 3)
        f_world = frame @ force[:3]

        forces_list.append(f_world)
        pos_list.append(contact.pos.copy())

        # Approximate contact point velocity from parent body
        body_id = model.geom_bodyid[g1 if g1 in hand_geom_ids else g2]
        body_vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, body_vel, 0)
        vel_list.append(body_vel[:3].copy())

    if not forces_list:
        empty = np.zeros((0, 3))
        return empty, empty, empty

    return np.array(forces_list), np.array(pos_list), np.array(vel_list)


def count_hand_bottle_contacts(model, data, hand_geom_ids: set[int]) -> int:
    """Count contacts between hand geoms and bottle."""
    bottle_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_geom")
    count = 0
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if g1 == bottle_geom and g2 in hand_geom_ids:
            count += 1
        elif g2 == bottle_geom and g1 in hand_geom_ids:
            count += 1
    return count


def get_hand_geom_ids(model, hand_body_prefix: str = "rh_") -> set[int]:
    """Collect geom IDs belonging to hand bodies (Shadow Hand naming: rh_*)."""
    ids: set[int] = set()
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith(hand_body_prefix) or "hand" in body_name.lower():
            ids.add(geom_id)
    return ids
