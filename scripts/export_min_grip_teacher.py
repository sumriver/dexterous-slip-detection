#!/usr/bin/env python3
"""NN-Policy-1: export min-sufficient grip teacher ``y_policy``.

For each ketchup case:
  1) Run NN-2 (or rule) antislip with logging → grip sequence ``y_grip``.
  2) If PASS (with margin), binary-search smallest ``antislip_grip_max`` that still
     passes; set ``y_policy[t] = min(y_grip[t], G*)``.
  3) If FAIL at 0.25, keep ``y_policy = y_grip``.

Writes ``data/slip_nn_policy/{train,val,test}/windows.npz`` + manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_slip_dataset import (  # noqa: E402
    EXTEND_LIFT_TARGET_M,
    EXTEND_S,
    POLICY_TEST_CASES,
    POLICY_VAL_CASES,
    WINDOW_STEPS,
    ExportCase,
    _base_case_name,
    _workspace_ready,
    build_policy_cases,
)
from sim.slip_dataset_logger import (  # noqa: E402
    SlipDatasetLogger,
    compute_norm_stats,
    merge_npz_shards,
    split_by_case,
    write_manifest,
)
from sim.slip_nn_detector import load_detector_from_dir  # noqa: E402
from sim.slip_nn_features import FEATURE_DIM, SlipFeatureBuilder  # noqa: E402
from sim.spider_ketchup import DEFAULT_WORKSPACE  # noqa: E402
from sim.spider_replay import SpiderTaskConfig, replay_spider_task  # noqa: E402

SPIDER = ROOT / "third_party" / "spider"
OUT_DIR = ROOT / "data" / "slip_nn_policy"
EXTEND_STEPS = 200

# Margin vs PASS gate (spec § risk): require slightly stronger than 6 cm.
DZ_MARGIN_M = 0.07
DROP_MAX_M = 0.03
GRIP_FLOOR = 0.05
GRIP_MAX = 0.25


def _cfg() -> SpiderTaskConfig:
    return SpiderTaskConfig(
        dataset_dir=SPIDER / "example_datasets",
        dataset_name="arcticv2",
        robot_type="xhand",
        embodiment_type="right",
        task="s01-ketchup_use_01",
        workspace_root=DEFAULT_WORKSPACE,
    )


def _passes(result, *, dz_min_m: float) -> bool:
    drop = max(0.0, result.object_z_after_trajectory - result.object_z_end)
    return (
        result.post_extend_object_dz >= dz_min_m
        and result.post_extend_contact_steps >= EXTEND_STEPS
        and drop <= DROP_MAX_M
    )


def _replay(
    spec: ExportCase,
    *,
    grip_max: float,
    logger: SlipDatasetLogger | None,
    nn_detector,
    use_nn: bool,
    out_sub: str,
):
    builder = SlipFeatureBuilder(sim_dt=0.01) if logger is not None else None
    # Fresh detector state each trial.
    if nn_detector is not None:
        nn_detector.reset_extend()
    return replay_spider_task(
        _cfg(),
        OUT_DIR / "replay_logs" / out_sub / spec.name,
        save_video=False,
        post_lift_m=spec.lift_m,
        post_extend_s=spec.extend_s,
        post_mimic_s=1.0,
        mass_scale=spec.mass_scale,
        friction_scale=spec.friction_scale,
        log_energy=False,
        antislip=not use_nn,
        antislip_scheme=2,
        antislip_grip_max=grip_max,
        antislip_nn=use_nn,
        nn_detector=nn_detector if use_nn else None,
        dataset_logger=logger,
        feature_builder=builder,
        dataset_case_name=spec.name,
    )


def _binary_search_grip_cap(
    spec: ExportCase,
    *,
    nn_detector,
    use_nn: bool,
    lo: float,
    hi: float,
    iters: int,
) -> float:
    """Smallest grip cap in [lo, hi] that still passes with DZ margin."""
    best = hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = _replay(
            spec,
            grip_max=mid,
            logger=None,
            nn_detector=nn_detector,
            use_nn=use_nn,
            out_sub=f"search_{mid:.3f}",
        )
        if _passes(r, dz_min_m=DZ_MARGIN_M):
            best = mid
            hi = mid
        else:
            lo = mid
    return float(best)


def _run_case(
    spec: ExportCase,
    *,
    window_steps: int,
    nn_model_dir: Path,
    use_nn: bool,
    search_iters: int,
) -> tuple[Path, dict]:
    nn_detector = None
    if use_nn:
        nn_detector = load_detector_from_dir(nn_model_dir, threshold=None)

    logger = SlipDatasetLogger(window_steps=window_steps)
    result = _replay(
        spec,
        grip_max=GRIP_MAX,
        logger=logger,
        nn_detector=nn_detector,
        use_nn=use_nn,
        out_sub="log_full",
    )
    grips = np.array([lab.grip_extra for lab in logger._labels], dtype=np.float32)
    meta = {
        "case": spec.name,
        "mass_scale": spec.mass_scale,
        "friction_scale": spec.friction_scale,
        "full_pass": _passes(result, dz_min_m=0.06),
        "full_dz_cm": float(result.post_extend_object_dz * 100),
        "full_contacts": int(result.post_extend_contact_steps),
        "full_max_grip": float(grips.max()) if len(grips) else 0.0,
    }

    if _passes(result, dz_min_m=DZ_MARGIN_M) and float(grips.max()) > 1e-6:
        g_star = _binary_search_grip_cap(
            spec,
            nn_detector=nn_detector,
            use_nn=use_nn,
            lo=GRIP_FLOOR,
            hi=max(float(grips.max()), GRIP_FLOOR),
            iters=search_iters,
        )
        # Re-check at g_star; if search undershoots, bump to hi.
        check = _replay(
            spec,
            grip_max=g_star,
            logger=None,
            nn_detector=nn_detector,
            use_nn=use_nn,
            out_sub=f"verify_{g_star:.3f}",
        )
        if not _passes(check, dz_min_m=DZ_MARGIN_M):
            g_star = max(float(grips.max()), GRIP_FLOOR)
        y_policy = np.minimum(grips, g_star).astype(np.float32)
        meta["g_star"] = g_star
        meta["teacher"] = "min_cap"
    elif _passes(result, dz_min_m=0.06):
        # Passes with little/no grip — teach near-zero.
        y_policy = np.zeros_like(grips)
        meta["g_star"] = 0.0
        meta["teacher"] = "zero"
    else:
        y_policy = grips.copy()
        meta["g_star"] = float(grips.max()) if len(grips) else GRIP_MAX
        meta["teacher"] = "fail_keep_grip"

    logger.set_policy_grip(y_policy)
    shard_dir = OUT_DIR / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{spec.name}.npz"
    n_win = logger.save_npz(shard_path)
    meta["windows"] = n_win
    meta["y_policy_max"] = float(y_policy.max()) if len(y_policy) else 0.0
    meta["y_policy_mean"] = float(y_policy.mean()) if len(y_policy) else 0.0
    return shard_path, meta


def _save_split(name: str, data: dict[str, np.ndarray]) -> int:
    if not data:
        return 0
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "windows.npz", **data)
    return int(data["X"].shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NN-Policy-1 y_policy teachers")
    parser.add_argument("--window", type=int, default=WINDOW_STEPS)
    parser.add_argument("--case", default="", help="Single case name")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke subset: baseline + ÷2 + s060 + mass×2×÷2",
    )
    parser.add_argument("--no-variants", action="store_true")
    parser.add_argument(
        "--teacher",
        choices=("nn2", "rule"),
        default="nn2",
        help="Closed-loop teacher for logged grip (default: models/slip_nn_v2)",
    )
    parser.add_argument("--nn-model-dir", type=Path, default=ROOT / "models" / "slip_nn_v2")
    parser.add_argument("--search-iters", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or (ROOT / "data" / "slip_nn_policy")
    # Helpers read module-level OUT_DIR.
    import sys as _sys

    _sys.modules[__name__].OUT_DIR = out_dir  # type: ignore[attr-defined]

    if not _workspace_ready():
        print(
            "Missing ketchup workspace. Run setup_spider + build_spider_ketchup_right.",
            file=sys.stderr,
        )
        sys.exit(1)
    use_nn = args.teacher == "nn2"
    if use_nn and not any(args.nn_model_dir.glob("*.pt")):
        print(f"Missing NN checkpoint in {args.nn_model_dir}", file=sys.stderr)
        sys.exit(2)

    cases = build_policy_cases(include_variants=not args.no_variants)
    if args.quick:
        keep = {
            "baseline",
            "friction_div2",
            "friction_s060",
            "mass_x2",
            "mass_x2_friction_div2",
        }
        cases = [c for c in cases if _base_case_name(c.name) in keep]
    if args.case:
        cases = [c for c in cases if c.name == args.case]
        if not cases:
            print(f"Unknown case: {args.case}", file=sys.stderr)
            sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    case_meta: list[dict] = []

    for spec in cases:
        print(
            f"Policy teacher {spec.name} [{args.teacher}] "
            f"(mass×{spec.mass_scale}, μ×{spec.friction_scale})..."
        )
        shard, meta = _run_case(
            spec,
            window_steps=args.window,
            nn_model_dir=args.nn_model_dir,
            use_nn=use_nn,
            search_iters=args.search_iters,
        )
        shards.append(shard)
        case_meta.append(meta)
        print(
            f"  teacher={meta['teacher']} g*={meta.get('g_star', 0):.3f} "
            f"y_policy_max={meta['y_policy_max']:.3f} windows={meta['windows']} "
            f"dz={meta['full_dz_cm']:.1f}cm"
        )

    merged = merge_npz_shards(shards)
    if not merged or "y_policy" not in merged:
        print("No windows / missing y_policy.", file=sys.stderr)
        sys.exit(1)

    base_cases = np.array([_base_case_name(str(n)) for n in merged["case_name"]], dtype=object)
    merged_for_split = dict(merged)
    merged_for_split["case_name"] = base_cases
    train, val, test = split_by_case(
        merged_for_split,
        val_cases=POLICY_VAL_CASES,
        test_cases=POLICY_TEST_CASES,
    )
    n_train = _save_split("train", train)
    n_val = _save_split("val", val)
    n_test = _save_split("test", test)

    norm = compute_norm_stats(train["X"]) if n_train else {}
    write_manifest(
        out_dir / "manifest.json",
        window_steps=args.window,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        norm_stats=norm,
        extra={
            "feature_dim": FEATURE_DIM,
            "dataset": "slip_nn_policy",
            "policy_teacher": args.teacher,
            "nn_model_dir": str(args.nn_model_dir) if use_nn else "",
            "dz_margin_m": DZ_MARGIN_M,
            "grip_floor": GRIP_FLOOR,
            "y_policy_agg": "window_end",
            "val_cases": sorted(POLICY_VAL_CASES),
            "test_cases": sorted(POLICY_TEST_CASES),
            "policy_dose_grid": True,
            "per_case": case_meta,
        },
    )

    y_pol = train["y_policy"] if n_train else np.zeros(1)
    summary = {
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "total": n_train + n_val + n_test,
        "y_policy_train_max": float(np.max(y_pol)),
        "y_policy_train_mean": float(np.mean(y_pol)),
        "y_grip_train_max": float(np.max(train["y_grip"])) if n_train else 0.0,
        "teacher": args.teacher,
        "n_min_cap": sum(1 for m in case_meta if m["teacher"] == "min_cap"),
        "n_zero": sum(1 for m in case_meta if m["teacher"] == "zero"),
        "n_fail_keep": sum(1 for m in case_meta if m["teacher"] == "fail_keep_grip"),
        "manifest": str(out_dir / "manifest.json"),
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "case_meta.json").write_text(json.dumps(case_meta, indent=2))
    print()
    print(json.dumps(summary, indent=2))
    if summary["y_policy_train_max"] >= 0.249 and summary["n_min_cap"] == 0:
        print("WARNING: y_policy still looks capped at 0.25 — check search", file=sys.stderr)


if __name__ == "__main__":
    main()
