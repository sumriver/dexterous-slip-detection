"""Physics-based grasp metrics — no kinematic cheats."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class BottleGraspMetrics:
    n_contacts: int
    force_on_bottle: np.ndarray
    support_force_z: float
    force_magnitude: float


def extract_bottle_hand_contact_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geom_ids: set[int],
) -> tuple[list[np.ndarray], int]:
    """Forces on bottle from hand contacts (world frame)."""
    bottle_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_geom")
    forces: list[np.ndarray] = []

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if g1 == bottle_geom and g2 in hand_geom_ids:
            bottle_is_first = True
        elif g2 == bottle_geom and g1 in hand_geom_ids:
            bottle_is_first = False
        else:
            continue

        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        frame = np.array(contact.frame, dtype=float).reshape(3, 3)
        # Constraint force on body of geom1 in contact frame → world
        f_on_geom1 = frame @ wrench[:3]
        f_on_bottle = f_on_geom1 if bottle_is_first else -f_on_geom1
        forces.append(f_on_bottle)

    return forces, len(forces)


def measure_bottle_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geom_ids: set[int],
) -> BottleGraspMetrics:
    forces, n = extract_bottle_hand_contact_forces(model, data, hand_geom_ids)
    if n == 0:
        zero = np.zeros(3)
        return BottleGraspMetrics(0, zero, 0.0, 0.0)

    total = np.sum(forces, axis=0)
    # Upward support: positive Z resists gravity (gravity is -Z)
    support_z = float(max(0.0, total[2]))
    return BottleGraspMetrics(
        n_contacts=n,
        force_on_bottle=total,
        support_force_z=support_z,
        force_magnitude=float(np.linalg.norm(total)),
    )


def required_support_force(model: mujoco.MjModel, bottle_body: str = "bottle") -> float:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bottle_body)
    mass = model.body_mass[bid]
    return float(mass * abs(model.opt.gravity[2]))
