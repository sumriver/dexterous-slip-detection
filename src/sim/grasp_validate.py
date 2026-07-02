"""Physics-based grasp validation — no kinematic cheats."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class GraspPhysicsReport:
    ok: bool
    n_hand_object_contacts: int
    n_floor_object_contacts: int
    support_force_z: float
    mg: float
    com: np.ndarray
    contact_points: list[np.ndarray]
    contact_hand_geoms: list[str]
    com_horizontal_offsets_mm: list[float]
    com_vertical_offsets_mm: list[float]
    has_thumb: bool
    has_finger: bool
    reasons: list[str]


def count_floor_object_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, object_geom_ids: set[int]
) -> int:
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    n = 0
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if (g1 in object_geom_ids and g2 == floor) or (g2 in object_geom_ids and g1 == floor):
            n += 1
    return n


def measure_hand_object_support_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geom_ids: set[int],
    object_geom_ids: set[int],
) -> tuple[float, list[np.ndarray], list[str]]:
    """Upward support on object from hand contacts (world Z)."""
    forces: list[np.ndarray] = []
    points: list[np.ndarray] = []
    geoms: list[str] = []
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
        f_on_hand_geom = frame @ wrench[:3]
        # Force on object is opposite of force on hand geom when object is second
        if g1 in object_geom_ids:
            f_on_obj = -f_on_hand_geom
        else:
            f_on_obj = f_on_hand_geom
        forces.append(f_on_obj)
        points.append(contact.pos.copy())
        geoms.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1 if g1 in hand_geom_ids else g2) or "")

    if not forces:
        return 0.0, [], []
    total = np.sum(forces, axis=0)
    return float(max(0.0, total[2])), points, geoms


def validate_grasp_physics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_body: str,
    hand_geom_ids: set[int],
    object_geom_ids: set[int],
    *,
    require_off_floor: bool = True,
    min_contacts: int = 2,
    min_support_ratio: float = 0.9,
    max_com_horizontal_offset_mm: float = 25.0,
    max_com_vertical_offset_mm: float = 15.0,
) -> GraspPhysicsReport:
    """Check whether grasp can statically support vertical lift."""
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    com = data.xipos[oid].copy()
    mg = float(model.body_mass[oid] * abs(model.opt.gravity[2]))

    floor_n = count_floor_object_contacts(model, data, object_geom_ids)
    support_z, points, geoms = measure_hand_object_support_z(
        model, data, hand_geom_ids, object_geom_ids
    )
    n_hand = len(points)

    h_offsets = [float(np.linalg.norm(p[:2] - com[:2]) * 1000) for p in points]
    v_offsets = [float((p[2] - com[2]) * 1000) for p in points]
    has_thumb = any("thumb" in g for g in geoms)
    has_finger = any(any(x in g for x in ("index", "middle", "ring", "pinky")) for g in geoms)

    reasons: list[str] = []
    if require_off_floor and floor_n > 0:
        reasons.append(f"object still on floor ({floor_n} floor contacts)")
    if n_hand < min_contacts:
        reasons.append(f"too few hand contacts ({n_hand} < {min_contacts})")
    if not has_thumb or not has_finger:
        reasons.append("missing thumb–finger opposition")
    if support_z < min_support_ratio * mg:
        reasons.append(f"support_z={support_z:.3f}N < {min_support_ratio*mg:.3f}N (mg)")
    if h_offsets and max(h_offsets) > max_com_horizontal_offset_mm:
        reasons.append(
            f"contact {max(h_offsets):.0f}mm horizontal from COM "
            f"(max {max_com_horizontal_offset_mm:.0f}mm) — no vertical support line"
        )
    if v_offsets and max(v_offsets) > max_com_vertical_offset_mm:
        reasons.append(
            f"contact {max(v_offsets):.0f}mm above COM "
            f"(max {max_com_vertical_offset_mm:.0f}mm) — lever, not pinch under COM"
        )

    return GraspPhysicsReport(
        ok=len(reasons) == 0,
        n_hand_object_contacts=n_hand,
        n_floor_object_contacts=floor_n,
        support_force_z=support_z,
        mg=mg,
        com=com,
        contact_points=points,
        contact_hand_geoms=geoms,
        com_horizontal_offsets_mm=h_offsets,
        com_vertical_offsets_mm=v_offsets,
        has_thumb=has_thumb,
        has_finger=has_finger,
        reasons=reasons,
    )


def format_grasp_report(report: GraspPhysicsReport) -> str:
    lines = [
        f"  hand↔object contacts: {report.n_hand_object_contacts}",
        f"  floor↔object contacts: {report.n_floor_object_contacts}",
        f"  support_z: {report.support_force_z:.3f}N  mg: {report.mg:.3f}N",
        f"  COM: ({report.com[0]:.3f}, {report.com[1]:.3f}, {report.com[2]:.3f})",
        f"  thumb+finger opposition: {report.has_thumb and report.has_finger}",
    ]
    for g, h, v in zip(report.contact_hand_geoms, report.com_horizontal_offsets_mm, report.com_vertical_offsets_mm):
        lines.append(f"    {g}: Δxy={h:.0f}mm  Δz={v:+.0f}mm from COM")
    if report.reasons:
        lines.append("  FAIL:")
        for r in report.reasons:
            lines.append(f"    - {r}")
    else:
        lines.append("  PASS: grasp can support vertical lift")
    return "\n".join(lines)
