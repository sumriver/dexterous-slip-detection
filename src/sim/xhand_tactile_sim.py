"""Map MuJoCo contacts to XHAND1-style tactile taxel grids (Tier A)."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


FINGER_NAMES = ("thumb", "index", "mid", "ring", "pinky")
GRID_ROWS = 12
GRID_COLS = 10


@dataclass
class TactileFrame:
    """One timestep of simulated tactile output."""

    normal_force: np.ndarray  # (5, 12, 10) N
    tangential_force: np.ndarray  # (5, 12, 10, 2) N
    contact_count: int
    total_normal_force: float


def _finger_from_body(body_name: str) -> int | None:
    name = body_name.lower()
    if "thumb" in name:
        return 0
    if "index" in name:
        return 1
    if "mid" in name:
        return 2
    if "ring" in name:
        return 3
    if "pinky" in name:
        return 4
    return None


def _taxel_index(contact_pos: np.ndarray, site_pos: np.ndarray, site_xmat: np.ndarray) -> tuple[int, int] | None:
    """Project contact onto fingertip plane; map to 12x10 grid."""
    local = site_xmat.T @ (contact_pos - site_pos)
    # fingertip pad normal ~ local Z, tangential plane X-Y
    u = float(local[0])
    v = float(local[1])
    pad_half_w = 0.012
    pad_half_h = 0.018
    if abs(u) > pad_half_w or abs(v) > pad_half_h:
        return None
    col = int((u + pad_half_w) / (2 * pad_half_w) * GRID_COLS)
    row = int((v + pad_half_h) / (2 * pad_half_h) * GRID_ROWS)
    col = int(np.clip(col, 0, GRID_COLS - 1))
    row = int(np.clip(row, 0, GRID_ROWS - 1))
    return row, col


class XHandTactileSimulator:
    """Contact-based taxel mapping for energy-flow inputs."""

    def __init__(self, model: mujoco.MjModel, hand_geom_ids: set[int]):
        self.model = model
        self.hand_geom_ids = hand_geom_ids
        self.bottle_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_geom")
        self._tip_site_ids: dict[int, int] = {}
        for finger_id, finger in enumerate(FINGER_NAMES):
            site_name = f"xh_tip_{finger}"
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:
                self._tip_site_ids[finger_id] = site_id

    def sample(self, data: mujoco.MjData) -> TactileFrame:
        normal = np.zeros((5, GRID_ROWS, GRID_COLS), dtype=float)
        tangent = np.zeros((5, GRID_ROWS, GRID_COLS, 2), dtype=float)
        contact_count = 0
        total_fn = 0.0

        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if g1 == self.bottle_geom and g2 in self.hand_geom_ids:
                hand_geom = g2
            elif g2 == self.bottle_geom and g1 in self.hand_geom_ids:
                hand_geom = g1
            else:
                continue
            body_id = self.model.geom_bodyid[hand_geom]
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            finger_id = _finger_from_body(body_name)
            if finger_id is None:
                continue

            site_id = self._tip_site_ids.get(finger_id)
            if site_id is None:
                continue

            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, force)
            frame = np.array(contact.frame, dtype=float).reshape(3, 3)
            f_world = frame @ force[:3]
            site_pos = data.site_xpos[site_id]
            site_xmat = data.site_xmat[site_id].reshape(3, 3)
            f_local = site_xmat.T @ f_world
            fn_local = max(0.0, float(f_local[2]))
            ft_local = f_local[:2]

            idx = _taxel_index(contact.pos, site_pos, site_xmat)
            if idx is None:
                continue

            row, col = idx
            normal[finger_id, row, col] += fn_local
            tangent[finger_id, row, col] += ft_local
            contact_count += 1
            total_fn += fn_local

        return TactileFrame(
            normal_force=normal,
            tangential_force=tangent,
            contact_count=contact_count,
            total_normal_force=total_fn,
        )

    def aggregate_for_energy_flow(self, frame: TactileFrame) -> tuple[np.ndarray, np.ndarray]:
        """Collapse taxel grid to per-contact force/position proxies for energy-flow."""
        forces: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        for finger_id, site_id in self._tip_site_ids.items():
            patch = frame.normal_force[finger_id]
            if patch.max() <= 1e-9:
                continue
            row, col = np.unravel_index(int(patch.argmax()), patch.shape)
            fn = patch[row, col]
            ft = frame.tangential_force[finger_id, row, col]
            forces.append(np.array([ft[0], ft[1], fn]))
            # approximate taxel position on pad
            site_pos = np.zeros(3)  # filled by caller with mj_forward state if needed
            positions.append(site_pos)
        if not forces:
            empty = np.zeros((0, 3))
            return empty, empty
        return np.array(forces), np.array(positions)
