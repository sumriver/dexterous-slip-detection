#!/usr/bin/env python3
"""Open-loop robustness sweep: ketchup mass ×2/×4/×8/×16/×32 and friction ÷2/÷4/÷8.

No anti-slip control — baseline摸底 for later closed-loop comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.spider_ketchup import DEFAULT_WORKSPACE
from sim.spider_replay import SpiderTaskConfig, replay_spider_task

SPIDER = ROOT / "third_party" / "spider"
OUT_DIR = ROOT / "data" / "ketchup_robustness"

# Baseline extend lift target ~9 cm on nominal physics.
EXTEND_LIFT_TARGET_M = 0.10
EXTEND_STEPS = 200  # 2s @ 0.01s


@dataclass
class CaseSpec:
    name: str
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    sweep: str = "baseline"


@dataclass
class CaseResult:
    name: str
    sweep: str
    mass_scale: float
    friction_scale: float
    mass_kg: float
    status: str
    fail_reason: str
    steps: int
    contact_steps: int
    object_dz_cm: float
    traj_dz_cm: float
    extend_dz_cm: float
    extend_contact_steps: int
    extend_contact_ratio: float
    object_drop_cm: float
    sim_ok: bool
    video_path: str = ""
    center_slip_events: int = 0
    antislip_max_grip: float = 0.0


def _evaluate(result, *, extend_steps: int = EXTEND_STEPS) -> tuple[str, str]:
    """Return (status, fail_reason). status: pass | partial | fail."""
    extend_dz = result.post_extend_object_dz
    extend_contacts = result.post_extend_contact_steps
    ratio = extend_contacts / extend_steps if extend_steps > 0 else 0.0
    drop = max(0.0, result.object_z_after_trajectory - result.object_z_end)

    if result.steps < 400:
        return "fail", "sim_too_short"
    if drop > 0.03:
        return "fail", f"object_dropped_{drop * 100:.1f}cm_during_extend"
    if extend_contacts < 30:
        return "fail", f"lost_grasp_extend_contacts={extend_contacts}"
    if extend_dz < 0.02:
        return "fail", f"no_lift_extend_dz={extend_dz * 100:.1f}cm"
    if extend_dz >= 0.06 and ratio >= 0.5:
        return "pass", ""
    if extend_dz >= 0.03 or ratio >= 0.25:
        return "partial", f"weak_lift_dz={extend_dz * 100:.1f}cm_ratio={ratio:.2f}"
    return "fail", f"weak_lift_dz={extend_dz * 100:.1f}cm_ratio={ratio:.2f}"


def _run_case(spec: CaseSpec, *, save_video: bool = False, antislip: bool = False) -> CaseResult:
    cfg = SpiderTaskConfig(
        dataset_dir=SPIDER / "example_datasets",
        dataset_name="arcticv2",
        robot_type="xhand",
        embodiment_type="right",
        task="s01-ketchup_use_01",
        workspace_root=DEFAULT_WORKSPACE,
    )
    case_dir = OUT_DIR / spec.sweep / spec.name
    result = replay_spider_task(
        cfg,
        case_dir,
        save_video=save_video,
        post_lift_m=EXTEND_LIFT_TARGET_M,
        post_extend_s=2.0,
        post_mimic_s=1.0,
        mass_scale=spec.mass_scale,
        friction_scale=spec.friction_scale,
        log_energy=False,
        antislip=antislip,
    )
    status, reason = _evaluate(result)
    meta = result.physics_meta or {}
    traj_dz = result.object_z_after_trajectory - result.object_z_start
    ratio = result.post_extend_contact_steps / EXTEND_STEPS
    drop = max(0.0, result.object_z_after_trajectory - result.object_z_end)
    video_path = ""
    if result.video_path and result.video_path.exists():
        named = case_dir / f"{spec.name}.mp4"
        if result.video_path != named:
            result.video_path.replace(named)
        video_path = str(named)
    return CaseResult(
        name=spec.name,
        sweep=spec.sweep,
        mass_scale=spec.mass_scale,
        friction_scale=spec.friction_scale,
        mass_kg=float(meta.get("mass_kg", 0.0)),
        status=status,
        fail_reason=reason,
        steps=result.steps,
        contact_steps=result.contact_steps,
        object_dz_cm=result.object_dz * 100,
        traj_dz_cm=traj_dz * 100,
        extend_dz_cm=result.post_extend_object_dz * 100,
        extend_contact_steps=result.post_extend_contact_steps,
        extend_contact_ratio=ratio,
        object_drop_cm=drop * 100,
        sim_ok=result.steps >= 400,
        video_path=video_path,
        center_slip_events=result.center_slip_events,
        antislip_max_grip=result.antislip_max_grip,
    )


def build_cases() -> list[CaseSpec]:
    cases = [CaseSpec("baseline", sweep="baseline")]
    for scale in (2, 4, 8, 16, 32):
        cases.append(CaseSpec(f"mass_x{scale}", mass_scale=float(scale), sweep="mass"))
    for div in (2, 4, 8):
        cases.append(CaseSpec(f"friction_div{div}", friction_scale=1.0 / div, sweep="friction"))
    return cases


def _print_table(results: list[CaseResult]) -> None:
    header = (
        f"{'case':<18} {'mass':>6} {'μ×':>5} {'status':<8} "
        f"{'Δz_tot':>7} {'Δz_ext':>7} {'ext_ct':>6} {'drop':>6} reason"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<18} {r.mass_kg:6.3f} {r.friction_scale:5.3f} {r.status:<8} "
            f"{r.object_dz_cm:6.1f}cm {r.extend_dz_cm:6.1f}cm "
            f"{r.extend_contact_steps:6d} {r.object_drop_cm:5.1f}cm {r.fail_reason}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ketchup open-loop mass/friction sweep")
    parser.add_argument("--video", action="store_true", help="Record MP4 per case (slow)")
    parser.add_argument("--fail-video", action="store_true", help="Record MP4 for friction fail cases only")
    parser.add_argument("--antislip", action="store_true", help="Enable center-divergence anti-slip on extend")
    parser.add_argument("--case", default="", help="Run single case name only")
    args = parser.parse_args()

    if not DEFAULT_WORKSPACE.joinpath("scene.xml").exists():
        print("Missing workspace. Run: python3 scripts/build_spider_ketchup_right.py", file=sys.stderr)
        sys.exit(1)

    cases = build_cases()
    if args.case:
        cases = [c for c in cases if c.name == args.case]
        if not cases:
            print(f"Unknown case: {args.case}", file=sys.stderr)
            sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[CaseResult] = []
    for spec in cases:
        record = args.video or (args.fail_video and spec.sweep == "friction")
        print(f"Running {spec.name} (mass×{spec.mass_scale}, friction×{spec.friction_scale})...")
        results.append(_run_case(spec, save_video=record, antislip=args.antislip))

    summary = {
        "extend_lift_target_m": EXTEND_LIFT_TARGET_M,
        "antislip": args.antislip,
        "pass_criteria": {
            "extend_dz_m_min": 0.06,
            "extend_contact_ratio_min": 0.5,
            "max_drop_m": 0.03,
        },
        "cases": [asdict(r) for r in results],
    }
    summary_path = OUT_DIR / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print()
    _print_table(results)
    print()
    for status in ("pass", "partial", "fail"):
        names = [r.name for r in results if r.status == status]
        if names:
            print(f"{status.upper()}: {', '.join(names)}")
    videos = [r for r in results if r.video_path]
    if videos:
        print("\nVideos:")
        for r in videos:
            print(f"  {r.name}: {r.video_path}")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
