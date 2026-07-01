"""Energy state computation from contact forces and velocities.

Based on NAIST + Honda (arXiv:2512.21043):
  P^A_i = F_i · v_i  (applied power at each contact)
  m̃_t = sum(P^A) / sum(P̃^R)  (mass estimate under massless assumption)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnergyState:
    """Snapshot of energy-related state at one timestep."""

    applied_powers: np.ndarray  # shape (n_contacts,)
    retained_power: float
    mass_estimate: float
    contact_forces: np.ndarray  # shape (n_contacts, 3)
    contact_positions: np.ndarray  # shape (n_contacts, 3)


def compute_applied_power(forces: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Compute per-contact applied power P^A_i = F_i · v_i.

    Args:
        forces: (n, 3) contact force vectors in world frame [N]
        velocities: (n, 3) contact point velocities in world frame [m/s]

    Returns:
        (n,) applied power [W]
    """
    if forces.shape != velocities.shape:
        raise ValueError(f"Shape mismatch: forces {forces.shape}, velocities {velocities.shape}")
    if len(forces.shape) != 2 or forces.shape[1] != 3:
        raise ValueError(f"Expected shape (n, 3), got forces {forces.shape}")

    return np.sum(forces * velocities, axis=1)


def compute_retained_power(
    total_force: np.ndarray,
    center_velocity: np.ndarray,
    center_angular_velocity: np.ndarray | None = None,
) -> float:
    """Simplified retained power P̃^R under massless assumption.

    Full formulation from the paper depends on grasp center kinematics.
    This minimal version uses P̃^R ≈ F_total · v_center.
    """
    return float(np.dot(total_force, center_velocity))


def compute_mass_estimate(applied_powers: np.ndarray, retained_power: float, eps: float = 1e-6) -> float:
    """m̃_t = sum(P^A) / P̃^R  (Eq. 7 simplified scalar form)."""
    total_applied = float(np.sum(applied_powers))
    if abs(retained_power) < eps:
        return float("nan")
    return total_applied / retained_power
