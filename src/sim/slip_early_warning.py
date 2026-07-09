"""Early slip-warning signals: friction-cone utilization and contact centroid.

Tier-1 (earliest, requires known/estimated friction mu):
    friction utilization rho/mu, where rho = |f_t| / |f_n| per contact.
    Slip is imminent as utilization -> 1 (tangential force reaches the
    friction limit), which happens BEFORE the object moves macroscopically.

Tier-2 (mu-independent, real-robot friendly):
    contact centroid (COP) position. Its drift relative to the hand is an
    incipient-slip signal (the grasp starting to shift on the object).

Both use only contact forces + kinematics (real-robot compatible).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from sim.slip_vertical_support import decompose_contact_force_on_object


def object_pair_friction(model: mujoco.MjModel, object_geom_ids: set[int]) -> float:
    """Tangential friction coefficient for contacts involving the object.

    Prefers explicit contact pairs (what apply_object_physics scales); falls
    back to the element-wise max of the two geoms' friction otherwise.
    """
    for pid in range(model.npair):
        if model.pair_geom1[pid] in object_geom_ids or model.pair_geom2[pid] in object_geom_ids:
            return float(model.pair_friction[pid, 0])
    # fallback: max geom tangential friction among object geoms
    if object_geom_ids:
        return float(max(model.geom_friction[g, 0] for g in object_geom_ids))
    return 1.0


@dataclass
class EarlyWarningReading:
    n_contacts: int
    mu: float
    util_max: float        # max_c |f_t|/(mu|f_n|)   (friction utilization, 0..~1)
    util_wmean: float      # (sum|f_t|)/(mu*sum|f_n|) force-weighted
    cop_world: np.ndarray  # force(normal)-weighted contact centroid in world [3]
    total_fn: float        # sum |f_n|


def measure_early_warning(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geom_ids: set[int],
    object_geom_ids: set[int],
    mu: float,
) -> EarlyWarningReading:
    util_max = 0.0
    sum_ft = 0.0
    sum_fn = 0.0
    cop = np.zeros(3)
    n_con = 0

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        hand_hit = g1 in hand_geom_ids or g2 in hand_geom_ids
        obj_hit = g1 in object_geom_ids or g2 in object_geom_ids
        if not (hand_hit and obj_hit):
            continue

        _, f_n, f_t = decompose_contact_force_on_object(model, data, i, object_geom_ids)
        fn = float(np.linalg.norm(f_n))
        ft = float(np.linalg.norm(f_t))
        if fn < 1e-6:
            continue
        rho = ft / fn
        util_max = max(util_max, rho / mu if mu > 1e-9 else 0.0)
        sum_ft += ft
        sum_fn += fn
        cop += fn * np.asarray(contact.pos, dtype=float)
        n_con += 1

    if sum_fn > 1e-9:
        cop /= sum_fn
        util_wmean = sum_ft / (mu * sum_fn) if mu > 1e-9 else 0.0
    else:
        cop = np.full(3, np.nan)
        util_wmean = 0.0

    return EarlyWarningReading(
        n_contacts=n_con,
        mu=mu,
        util_max=util_max,
        util_wmean=util_wmean,
        cop_world=cop,
        total_fn=sum_fn,
    )
