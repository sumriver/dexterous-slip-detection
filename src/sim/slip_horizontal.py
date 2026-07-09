"""Horizontal-plane slip signals: force integrals along a hand-referenced X/Y frame.

Frame definition (user proposal for horizontal anti-slip):
  * X_hat = 四指平展方向 — the direction the four fingers (index/middle/ring/pinky)
    extend, i.e. the mean knuckle→tip vector, projected onto the horizontal plane.
  * Y_hat = 与平展方向垂直、指向右侧 — perpendicular to X in the horizontal plane,
    pointing to the right: Y_hat = X_hat × up_hat.
  * Both axes lie in the world horizontal plane (z-component = 0); the grip/lift
    (vertical) axis is handled separately by slip_vertical_support.

For every hand→object contact we decompose the contact force into world-frame
normal and tangential parts, then project the full force (normal + tangential)
onto X_hat / Y_hat and sum over all contacts. This yields the net horizontal
force the hand exerts on the object along the two in-plane axes, whose time
integral (impulse) is the horizontal slip signal.

Uses only contact forces + hand kinematics (real-robot compatible).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from sim.slip_vertical_support import decompose_contact_force_on_object

# Four non-thumb fingers used to define the "flat/extended" direction.
_FINGER_KNUCKLE_BODIES = {
    "index": "right_hand_index_bend_link",
    "middle": "right_hand_mid_link1",
    "ring": "right_hand_ring_link1",
    "pinky": "right_hand_pinky_link1",
}
_FINGER_TIP_SITES = {
    "index": "right_index_tip",
    "middle": "right_middle_tip",
    "ring": "right_ring_tip",
    "pinky": "right_pinky_tip",
}


@dataclass
class HorizontalFrame:
    """In-plane hand frame used for horizontal force decomposition."""

    x_hat: np.ndarray  # finger-extension direction (horizontal, unit)
    y_hat: np.ndarray  # perpendicular-right direction (horizontal, unit)
    up_hat: np.ndarray  # world up (anti-gravity, unit)
    origin: np.ndarray  # palm reference position (world)


def _up_hat(model: mujoco.MjModel) -> np.ndarray:
    g = np.asarray(model.opt.gravity, dtype=float)
    if np.linalg.norm(g) < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return -g / np.linalg.norm(g)


def compute_hand_horizontal_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    palm_site: str = "right_palm",
) -> HorizontalFrame:
    """Build the horizontal X/Y frame from the current four-finger pose.

    X_hat = mean(knuckle→tip over 4 fingers) projected to horizontal.
    Y_hat = X_hat × up_hat (points to the right of the extended fingers).
    """
    up = _up_hat(model)

    dirs = []
    for finger in _FINGER_KNUCKLE_BODIES:
        kb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _FINGER_KNUCKLE_BODIES[finger])
        ts = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _FINGER_TIP_SITES[finger])
        if kb < 0 or ts < 0:
            continue
        dirs.append(data.site_xpos[ts] - data.xpos[kb])
    if not dirs:
        raise ValueError("Could not resolve any finger knuckle/tip for horizontal frame")

    mean_dir = np.mean(dirs, axis=0)
    # Project finger direction onto the horizontal plane (remove vertical part).
    x_hat = mean_dir - np.dot(mean_dir, up) * up
    nx = np.linalg.norm(x_hat)
    if nx < 1e-9:
        raise ValueError("Finger extension direction is degenerate in horizontal plane")
    x_hat = x_hat / nx

    y_hat = np.cross(x_hat, up)
    ny = np.linalg.norm(y_hat)
    if ny < 1e-9:
        raise ValueError("Failed to build perpendicular Y axis")
    y_hat = y_hat / ny

    ps = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, palm_site)
    origin = data.site_xpos[ps].copy() if ps >= 0 else np.zeros(3)

    return HorizontalFrame(x_hat=x_hat, y_hat=y_hat, up_hat=up, origin=origin)


@dataclass
class HorizontalForceReading:
    """Net hand→object contact force resolved in the horizontal X/Y frame."""

    n_contacts: int
    fx: float           # Σ f_total · X_hat  [N]  (signed net force)
    fy: float           # Σ f_total · Y_hat  [N]
    fx_normal: float    # Σ f_normal · X_hat
    fy_normal: float    # Σ f_normal · Y_hat
    fx_tangent: float   # Σ f_tangent · X_hat
    fy_tangent: float   # Σ f_tangent · Y_hat
    f_horiz_mag: float  # |net| = sqrt(fx^2 + fy^2)
    sum_abs_horiz: float  # Σ_c |f_c projected onto horizontal plane|  [N]

    @property
    def direction_rad(self) -> float:
        """Direction of the net horizontal force in the hand X/Y frame [rad].

        0 = along finger extension (+X), +pi/2 = perpendicular right (+Y).
        Returns nan when the net force is negligible.
        """
        if self.f_horiz_mag < 1e-9:
            return float("nan")
        return float(np.arctan2(self.fy, self.fx))

    @property
    def imbalance_ratio(self) -> float:
        """|net horizontal force| / Σ|per-contact horizontal force|.

        ~0 when opposing contact forces cancel (balanced grasp, no slip);
        grows toward 1 when forces stop cancelling (net push / slip).
        Returns nan when there is no horizontal contact force.
        """
        if self.sum_abs_horiz < 1e-9:
            return float("nan")
        return self.f_horiz_mag / self.sum_abs_horiz


def measure_horizontal_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geom_ids: set[int],
    object_geom_ids: set[int],
    frame: HorizontalFrame,
) -> HorizontalForceReading:
    """Sum hand→object contact forces projected onto X_hat / Y_hat.

    The full force (normal + tangential) is projected; the normal-only and
    tangent-only projections are also accumulated for diagnostics.
    """
    x_hat = frame.x_hat
    y_hat = frame.y_hat

    fx = fy = 0.0
    fxn = fyn = 0.0
    fxt = fyt = 0.0
    sum_abs = 0.0
    n_con = 0

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        hand_hit = g1 in hand_geom_ids or g2 in hand_geom_ids
        obj_hit = g1 in object_geom_ids or g2 in object_geom_ids
        if not (hand_hit and obj_hit):
            continue

        f_w, f_n, f_t = decompose_contact_force_on_object(model, data, i, object_geom_ids)
        cx = float(np.dot(f_w, x_hat))
        cy = float(np.dot(f_w, y_hat))
        fx += cx
        fy += cy
        fxn += float(np.dot(f_n, x_hat))
        fyn += float(np.dot(f_n, y_hat))
        fxt += float(np.dot(f_t, x_hat))
        fyt += float(np.dot(f_t, y_hat))
        sum_abs += float(np.hypot(cx, cy))
        n_con += 1

    return HorizontalForceReading(
        n_contacts=n_con,
        fx=fx,
        fy=fy,
        fx_normal=fxn,
        fy_normal=fyn,
        fx_tangent=fxt,
        fy_tangent=fyt,
        f_horiz_mag=float(np.hypot(fx, fy)),
        sum_abs_horiz=sum_abs,
    )


class HorizontalImpulseIntegrator:
    """Running time-integral (impulse) of the horizontal forces Fx(t), Fy(t).

    Accumulates ∫Fx dt and ∫Fy dt (signed) plus ∫|F_horiz| dt (magnitude).
    A sliding-window variant is available via ``window_s``.
    """

    def __init__(self, sim_dt: float, *, window_s: float | None = None):
        self.sim_dt = sim_dt
        self.window_steps = (
            None if window_s is None else max(1, int(round(window_s / sim_dt)))
        )
        self._ix = 0.0
        self._iy = 0.0
        self._imag = 0.0
        self._bx: list[float] = []
        self._by: list[float] = []
        self._bmag: list[float] = []

    def reset(self) -> None:
        self._ix = self._iy = self._imag = 0.0
        self._bx.clear()
        self._by.clear()
        self._bmag.clear()

    def update(self, reading: HorizontalForceReading) -> tuple[float, float, float]:
        dx = reading.fx * self.sim_dt
        dy = reading.fy * self.sim_dt
        dm = reading.f_horiz_mag * self.sim_dt
        if self.window_steps is None:
            self._ix += dx
            self._iy += dy
            self._imag += dm
        else:
            self._bx.append(dx)
            self._by.append(dy)
            self._bmag.append(dm)
            if len(self._bx) > self.window_steps:
                self._bx.pop(0)
                self._by.pop(0)
                self._bmag.pop(0)
            self._ix = float(sum(self._bx))
            self._iy = float(sum(self._by))
            self._imag = float(sum(self._bmag))
        return self._ix, self._iy, self._imag

    @property
    def impulse_x(self) -> float:
        return self._ix

    @property
    def impulse_y(self) -> float:
        return self._iy

    @property
    def impulse_mag(self) -> float:
        return self._imag
