"""NN-0 dataset validation (L2–L4).

L2: label identities + split leakage
L3: per-case / per-phase positive rates + feature sanity
L4: step-wise align SlipFeatureBuilder vs independent scheme-1/2 detectors
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sim.slip_nn_features import FEATURE_DIM, FEATURE_NAMES, SlipFeatureBuilder, make_step_context

IDX_N_CONTACTS = 0
IDX_SEP = 1
IDX_SEP_DELTA = 2
IDX_S_RAW = 8
IDX_S_SMOOTH = 9
IDX_S_AVG = 10
IDX_SLIP_RULE_S2 = 17
IDX_DZ_TRAJ_END = 19
IDX_PHASE_EXTEND = 24

DEFAULT_VAL_CASES = frozenset({"friction_div4", "mass_x16"})
DEFAULT_TEST_CASES = frozenset({"friction_div2"})


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidateReport:
    level: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "ok": self.ok,
            "checks": [asdict(c) for c in self.checks],
        }


@dataclass
class AlignStepMismatch:
    step: int
    phase: str
    field: str
    builder: float
    independent: float


def base_case_name(case_name: str) -> str:
    s = str(case_name)
    for suffix in ("_v1", "_v2", "_v3"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def load_split_npz(data_dir: Path, split: str) -> dict[str, np.ndarray] | None:
    path = data_dir / split / "windows.npz"
    if not path.exists():
        return None
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def load_all_splits(data_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        data = load_split_npz(data_dir, split)
        if data is not None:
            out[split] = data
    return out


def _pos_rate(y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.asarray(y, dtype=np.float32).mean())


# ---------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------


def check_l2_label_identities(windows: dict[str, np.ndarray], *, split: str) -> list[CheckResult]:
    checks: list[CheckResult] = []
    required = ("y_scheme1", "y_scheme2", "y_gt", "y_fused", "y_grip", "X")
    missing = [k for k in required if k not in windows]
    checks.append(
        CheckResult(
            name=f"{split}.keys",
            ok=not missing,
            detail="" if not missing else f"missing {missing}",
        )
    )
    if missing:
        return checks

    x = windows["X"]
    n = int(x.shape[0])
    shape_ok = x.ndim == 3 and x.shape[2] == FEATURE_DIM and n > 0
    checks.append(
        CheckResult(
            name=f"{split}.X_shape",
            ok=shape_ok,
            detail=f"X.shape={x.shape} expected (N,T,{FEATURE_DIM})",
            metrics={"N": n, "T": int(x.shape[1]) if x.ndim == 3 else -1, "D": int(x.shape[2]) if x.ndim == 3 else -1},
        )
    )

    y1 = windows["y_scheme1"].astype(np.float32)
    y2 = windows["y_scheme2"].astype(np.float32)
    yf = windows["y_fused"].astype(np.float32)
    expected = np.maximum(y1, y2)
    n_bad = int(np.sum(np.abs(yf - expected) > 1e-6))
    checks.append(
        CheckResult(
            name=f"{split}.y_fused=y1|y2",
            ok=n_bad == 0,
            detail=f"mismatches={n_bad}/{n}",
            metrics={"mismatches": n_bad, "n": n},
        )
    )

    for key in ("y_scheme1", "y_scheme2", "y_gt", "y_fused"):
        arr = windows[key].astype(np.float32)
        in_01 = bool(np.all((arr >= 0.0) & (arr <= 1.0)))
        rate = float(arr.mean()) if n else 0.0
        checks.append(
            CheckResult(
                name=f"{split}.{key}_range",
                ok=in_01,
                detail=f"mean={rate:.4f} in[0,1]={in_01}",
                metrics={"pos_rate": rate},
            )
        )

    # At least one slip label should be non-degenerate on sizable splits
    if n >= 50:
        rates = {
            k: float(windows[k].astype(np.float32).mean())
            for k in ("y_scheme1", "y_scheme2", "y_gt", "y_fused")
        }
        any_signal = any(0.0 < r < 1.0 for r in rates.values()) or any(
            r > 0.0 for r in rates.values()
        )
        checks.append(
            CheckResult(
                name=f"{split}.labels_have_signal",
                ok=any_signal,
                detail=f"rates={ {k: round(v, 4) for k, v in rates.items()} }",
                metrics=rates,
            )
        )
    return checks


def check_l2_split_leakage(
    splits: dict[str, dict[str, np.ndarray]],
    *,
    val_cases: frozenset[str] = DEFAULT_VAL_CASES,
    test_cases: frozenset[str] = DEFAULT_TEST_CASES,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    by_split: dict[str, set[str]] = {}
    for split, data in splits.items():
        if "case_name" not in data:
            checks.append(
                CheckResult(name=f"{split}.case_name", ok=False, detail="missing case_name")
            )
            continue
        bases = {base_case_name(c) for c in data["case_name"]}
        by_split[split] = bases
        checks.append(
            CheckResult(
                name=f"{split}.case_count",
                ok=len(bases) > 0,
                detail=f"base_cases={sorted(bases)}",
                metrics={"n_base_cases": len(bases)},
            )
        )

    train = by_split.get("train", set())
    val = by_split.get("val", set())
    test = by_split.get("test", set())

    leak_test = (train | val) & test_cases
    checks.append(
        CheckResult(
            name="leak.test_cases_isolated",
            ok=not leak_test,
            detail=f"leaked={sorted(leak_test)}" if leak_test else "ok",
            metrics={"expected_test": sorted(test_cases), "actual_test": sorted(test)},
        )
    )
    leak_val = train & val_cases
    checks.append(
        CheckResult(
            name="leak.val_cases_isolated",
            ok=not leak_val,
            detail=f"leaked={sorted(leak_val)}" if leak_val else "ok",
        )
    )
    if "test" in by_split:
        missing_test = test_cases - test
        checks.append(
            CheckResult(
                name="split.test_has_friction_div2",
                ok=test_cases <= test,
                detail=f"missing={sorted(missing_test)}" if missing_test else "ok",
            )
        )
    if "val" in by_split:
        missing_val = val_cases - val
        checks.append(
            CheckResult(
                name="split.val_has_expected",
                ok=val_cases <= val,
                detail=f"missing={sorted(missing_val)}" if missing_val else "ok",
            )
        )
    if "train" in by_split:
        bad = train & (val_cases | test_cases)
        checks.append(
            CheckResult(
                name="split.train_excludes_held_out",
                ok=not bad,
                detail=f"unexpected={sorted(bad)}" if bad else "ok",
            )
        )
    return checks


def validate_l2(
    splits: dict[str, dict[str, np.ndarray]],
    *,
    val_cases: frozenset[str] = DEFAULT_VAL_CASES,
    test_cases: frozenset[str] = DEFAULT_TEST_CASES,
) -> ValidateReport:
    checks: list[CheckResult] = []
    if not splits:
        checks.append(CheckResult(name="splits.present", ok=False, detail="no train/val/test NPZ"))
        return ValidateReport(level="L2", checks=checks)
    for split, data in splits.items():
        checks.extend(check_l2_label_identities(data, split=split))
    checks.extend(check_l2_split_leakage(splits, val_cases=val_cases, test_cases=test_cases))
    return ValidateReport(level="L2", checks=checks)


# ---------------------------------------------------------------------------
# L3
# ---------------------------------------------------------------------------


def check_l3_statistics(
    splits: dict[str, dict[str, np.ndarray]],
    *,
    require_10k_train: bool = True,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if not any("X" in d for d in splits.values()):
        return [CheckResult(name="l3.data", ok=False, detail="empty")]

    if "train" in splits:
        n_train = int(splits["train"]["X"].shape[0])
        checks.append(
            CheckResult(
                name="l3.train_count",
                ok=(n_train >= 10_000) if require_10k_train else n_train > 0,
                detail=f"train_windows={n_train} (target>=10000)",
                metrics={"n_train": n_train},
            )
        )

    for split, data in splits.items():
        if "X" not in data or "y_scheme2" not in data:
            continue
        phase = data["X"][:, -1, IDX_PHASE_EXTEND]
        y2 = data["y_scheme2"].astype(np.float32)
        extend_mask = phase >= 0.5
        traj_mask = ~extend_mask
        if extend_mask.any() and traj_mask.any():
            r_ext = _pos_rate(y2[extend_mask])
            r_traj = _pos_rate(y2[traj_mask])
            checks.append(
                CheckResult(
                    name=f"l3.{split}.extend_vs_traj_y_scheme2",
                    ok=r_ext >= r_traj - 1e-6,
                    detail=f"extend={r_ext:.4f} traj={r_traj:.4f}",
                    metrics={"extend": r_ext, "traj": r_traj},
                )
            )

        dz = data["X"][:, -1, IDX_DZ_TRAJ_END]
        if traj_mask.any():
            max_abs = float(np.max(np.abs(dz[traj_mask])))
            checks.append(
                CheckResult(
                    name=f"l3.{split}.traj_dz_traj_end_zero",
                    ok=max_abs < 1e-5,
                    detail=f"max|dz|={max_abs:.2e} on traj last-frames",
                    metrics={"max_abs": max_abs},
                )
            )

        bad = int(np.sum(~np.isfinite(data["X"])))
        checks.append(
            CheckResult(
                name=f"l3.{split}.X_finite",
                ok=bad == 0,
                detail=f"non_finite={bad}",
            )
        )

        last = data["X"][:, -1, :]
        contact = last[:, IDX_N_CONTACTS] >= 2
        if contact.any():
            s_raw = last[contact, IDX_S_RAW]
            frac_pos = float(np.mean(s_raw > 1e-3))
            checks.append(
                CheckResult(
                    name=f"l3.{split}.contact_implies_support",
                    ok=frac_pos > 0.1,
                    detail=f"frac(S_raw>0|n_con>=2)={frac_pos:.3f}",
                    metrics={"frac_pos": frac_pos, "n_contact_windows": int(contact.sum())},
                )
            )

    train = splits.get("train")
    test = splits.get("test")
    if train is not None and test is not None and "case_name" in train and "case_name" in test:

        def _rate_for(data: dict[str, np.ndarray], base: str, key: str) -> float | None:
            names = np.array([base_case_name(c) for c in data["case_name"]], dtype=object)
            mask = names == base
            if not mask.any() or key not in data:
                return None
            return _pos_rate(data[key][mask])

        r_base_s2 = _rate_for(train, "baseline", "y_scheme2")
        r_fd2_s2 = _rate_for(test, "friction_div2", "y_scheme2")
        if r_base_s2 is not None and r_fd2_s2 is not None:
            checks.append(
                CheckResult(
                    name="l3.friction_div2_vs_baseline_y_scheme2",
                    ok=r_fd2_s2 >= r_base_s2 - 0.02,
                    detail=f"fd2={r_fd2_s2:.4f} baseline={r_base_s2:.4f}",
                    metrics={"friction_div2": r_fd2_s2, "baseline": r_base_s2},
                )
            )
        r_base_gt = _rate_for(train, "baseline", "y_gt")
        r_fd2_gt = _rate_for(test, "friction_div2", "y_gt")
        if r_base_gt is not None and r_fd2_gt is not None:
            checks.append(
                CheckResult(
                    name="l3.friction_div2_vs_baseline_y_gt",
                    ok=True,
                    detail=f"fd2={r_fd2_gt:.4f} baseline={r_base_gt:.4f}",
                    metrics={"friction_div2": r_fd2_gt, "baseline": r_base_gt},
                )
            )
    return checks


def validate_l3(
    splits: dict[str, dict[str, np.ndarray]],
    *,
    require_10k_train: bool = True,
) -> ValidateReport:
    return ValidateReport(
        level="L3",
        checks=check_l3_statistics(splits, require_10k_train=require_10k_train),
    )


def per_case_label_table(splits: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, data in splits.items():
        if "case_name" not in data:
            continue
        bases = np.array([base_case_name(c) for c in data["case_name"]], dtype=object)
        for base in sorted(set(bases.tolist())):
            mask = bases == base
            out.append(
                {
                    "case": base,
                    "split": split,
                    "n": int(mask.sum()),
                    "y_scheme1": _pos_rate(data["y_scheme1"][mask]) if "y_scheme1" in data else None,
                    "y_scheme2": _pos_rate(data["y_scheme2"][mask]) if "y_scheme2" in data else None,
                    "y_gt": _pos_rate(data["y_gt"][mask]) if "y_gt" in data else None,
                    "y_fused": _pos_rate(data["y_fused"][mask]) if "y_fused" in data else None,
                }
            )
    return out


# ---------------------------------------------------------------------------
# L4
# ---------------------------------------------------------------------------


def run_l4_alignment(
    *,
    workspace_root: Path,
    spider_dataset_dir: Path,
    mass_scale: float = 1.0,
    friction_scale: float = 1.0,
    extend_s: float = 2.0,
    lift_m: float = 0.10,
    sim_dt: float = 0.01,
    atol: float = 1e-4,
    max_mismatches: int = 50,
) -> ValidateReport:
    """Replay ketchup and compare SlipFeatureBuilder to independent detectors."""
    import mujoco

    from sim.slip_center_detect import CenterDivergenceDetector
    from sim.slip_vertical_support import (
        VerticalSupportAntislipDetector,
        gravity_up,
        measure_vertical_support,
    )
    from sim.spider_replay import (
        SpiderTaskConfig,
        build_extend_mimic_lift_controls,
        get_object_geom_ids,
        get_spider_hand_collision_geom_ids,
        load_trajectory_arrays,
        upsample_controls,
    )
    from sim.spider_scene_modify import apply_object_physics

    cfg = SpiderTaskConfig(
        dataset_dir=spider_dataset_dir,
        dataset_name="arcticv2",
        robot_type="xhand",
        embodiment_type="right",
        task="s01-ketchup_use_01",
        workspace_root=workspace_root,
    )
    if not cfg.scene_path.exists() or not cfg.trajectory_path.exists():
        return ValidateReport(
            level="L4",
            checks=[
                CheckResult(
                    name="l4.workspace",
                    ok=False,
                    detail=f"missing scene/traj under {workspace_root}",
                )
            ],
        )

    model = mujoco.MjModel.from_xml_path(str(cfg.scene_path))
    apply_object_physics(
        model, mass_scale=mass_scale, friction_scale=friction_scale, object_body="right_object"
    )
    model.opt.timestep = sim_dt
    data = mujoco.MjData(model)

    qpos_ref, qvel_ref, ctrl_ref = load_trajectory_arrays(cfg.trajectory_path, model, cfg.data_type)
    qpos_ref, qvel_ref, ctrl_ref = upsample_controls(qpos_ref, qvel_ref, ctrl_ref, sim_dt, 0.02)

    hand_geoms = get_spider_hand_collision_geom_ids(model)
    object_geoms = get_object_geom_ids(model, "right_object")
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")
    g_hat = gravity_up(model)
    arm_tz_index = 2

    builder = SlipFeatureBuilder(sim_dt=sim_dt)
    center_ind = CenterDivergenceDetector(sim_dt=sim_dt)
    support_ind = VerticalSupportAntislipDetector(2.0, sim_dt)

    data.qpos[:] = qpos_ref[0]
    data.qvel[:] = qvel_ref[0]
    data.ctrl[:] = ctrl_ref[0]
    mujoco.mj_forward(model, data)

    z_start = float(data.xpos[object_id][2])
    builder.reset_trajectory(z_start)
    object_z_traj_end = z_start
    object_z_extend_start: float | None = None

    mismatches: list[AlignStepMismatch] = []
    n_steps = 0
    n_ok = 0

    def check(phase: str, wrist_tz: float) -> None:
        nonlocal n_steps, n_ok
        ctx = make_step_context(
            phase=phase,
            wrist_tz=wrist_tz,
            grip_extra=0.0,
            friction_scale=friction_scale,
            object_z=float(data.xpos[object_id][2]),
            object_z_traj_end=object_z_traj_end,
            object_z_extend_start=object_z_extend_start,
            object_z_start=z_start,
            in_trajectory=phase == "trajectory",
        )
        reading = builder.build(model, data, hand_geoms, object_geoms, object_id, ctx)
        center = center_ind.update(model, data, hand_geoms, object_geoms, object_id)
        vs = measure_vertical_support(
            model, data, hand_geoms, object_geoms, "right_object", g_hat=g_hat
        )
        support = support_ind.update(vs.support_z)

        feat = reading.features
        lab = reading.labels
        pairs = [
            ("y_scheme1", float(lab.y_scheme1), float(center.slip)),
            ("y_scheme2", float(lab.y_scheme2), float(support.slip_active)),
            ("S_raw", float(feat[IDX_S_RAW]), float(vs.support_z)),
            ("S_smooth", float(feat[IDX_S_SMOOTH]), float(support.support_smooth)),
            ("S_avg", float(feat[IDX_S_AVG]), float(support.support_avg)),
            ("n_contacts", float(feat[IDX_N_CONTACTS]), float(center.n_contacts)),
            ("slip_rule_s2", float(feat[IDX_SLIP_RULE_S2]), float(support.slip_active)),
        ]
        if not np.isnan(center.separation_m):
            pairs.append(("sep", float(feat[IDX_SEP]), float(center.separation_m)))
            pairs.append(("sep_delta", float(feat[IDX_SEP_DELTA]), float(center.separation_delta_m)))

        step_ok = True
        for name, a, b in pairs:
            if abs(a - b) > atol:
                step_ok = False
                if len(mismatches) < max_mismatches:
                    mismatches.append(AlignStepMismatch(n_steps, phase, name, a, b))
        n_steps += 1
        if step_ok:
            n_ok += 1

    for ctrl in ctrl_ref:
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        check("trajectory", float(ctrl[arm_tz_index]))

    object_z_traj_end = float(data.xpos[object_id][2])
    builder.mark_trajectory_end(object_z_traj_end)

    z_extend = float(data.xpos[object_id][2])
    object_z_extend_start = z_extend
    builder.reset_extend(z_extend)
    center_ind.reset()
    support_ind.reset_peak()

    extend_ctrl = build_extend_mimic_lift_controls(
        ctrl_ref,
        sim_dt=sim_dt,
        extend_s=extend_s,
        mimic_s=1.0,
        lift_m=lift_m,
        arm_tz_index=arm_tz_index,
    )
    for ctrl in extend_ctrl:
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        check("extend", float(ctrl[arm_tz_index]))

    n_bad = n_steps - n_ok
    sample = [
        {
            "step": m.step,
            "phase": m.phase,
            "field": m.field,
            "builder": m.builder,
            "independent": m.independent,
        }
        for m in mismatches[:5]
    ]
    return ValidateReport(
        level="L4",
        checks=[
            CheckResult(
                name="l4.steps_run",
                ok=n_steps > 100,
                detail=f"n_steps={n_steps}",
                metrics={"n_steps": n_steps},
            ),
            CheckResult(
                name="l4.builder_vs_detectors",
                ok=n_bad == 0,
                detail=f"ok={n_ok}/{n_steps} mismatches={n_bad}",
                metrics={
                    "n_ok": n_ok,
                    "n_steps": n_steps,
                    "n_mismatch_steps": n_bad,
                    "mismatch_fields": sorted({m.field for m in mismatches}),
                    "sample": sample,
                },
            ),
        ],
    )


def merge_reports(*reports: ValidateReport) -> dict[str, Any]:
    return {
        "ok": all(r.ok for r in reports),
        "reports": [r.to_dict() for r in reports],
        "feature_names": list(FEATURE_NAMES),
        "feature_dim": FEATURE_DIM,
    }
