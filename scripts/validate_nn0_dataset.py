#!/usr/bin/env python3
"""Validate NN-0 slip dataset (L2–L4).

L2  label identities + train/val/test leakage
L3  per-phase / per-case stats + feature sanity
L4  step-wise SlipFeatureBuilder vs independent scheme-1/2 detectors (MuJoCo)

Examples::

    # NPZ already exported:
    python3 scripts/validate_nn0_dataset.py

    # Export if missing, then validate + L4 align:
    python3 scripts/validate_nn0_dataset.py --export-if-missing --align

    # Quick smoke (small export, relax 10k train gate):
    python3 scripts/validate_nn0_dataset.py --export-if-missing --quick --align
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.slip_nn_validate import (
    DEFAULT_TEST_CASES,
    DEFAULT_VAL_CASES,
    load_all_splits,
    merge_reports,
    per_case_label_table,
    run_l4_alignment,
    validate_l2,
    validate_l3,
)
from sim.spider_ketchup import DEFAULT_WORKSPACE

SPIDER = ROOT / "third_party" / "spider"
DEFAULT_DATA = ROOT / "data" / "slip_nn"


def _print_checks(report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"\n=== {report.level} [{status}] ===")
    for c in report.checks:
        mark = "OK" if c.ok else "FAIL"
        print(f"  [{mark}] {c.name}: {c.detail}")


def _print_case_table(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n=== Per-case label rates ===")
    hdr = f"{'case':<16} {'split':<6} {'n':>6} {'y1':>7} {'y2':>7} {'y_gt':>7} {'y_f':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['case']:<16} {r['split']:<6} {r['n']:6d} "
            f"{(r['y_scheme1'] or 0):7.3f} {(r['y_scheme2'] or 0):7.3f} "
            f"{(r['y_gt'] or 0):7.3f} {(r['y_fused'] or 0):7.3f}"
        )


def _workspace_ready() -> bool:
    traj = DEFAULT_WORKSPACE / "trajectory_mjwp_fast.npz"
    return DEFAULT_WORKSPACE.joinpath("scene.xml").exists() and traj.exists() and traj.stat().st_size > 1000


def _maybe_export(*, quick: bool) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / "export_slip_dataset.py")]
    if quick:
        cmd.append("--quick")
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate NN-0 dataset (L2–L4)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="data/slip_nn root")
    parser.add_argument(
        "--export-if-missing",
        action="store_true",
        help="Run export_slip_dataset.py when train NPZ is absent",
    )
    parser.add_argument("--quick", action="store_true", help="Use --quick export / soft 10k gate")
    parser.add_argument("--align", action="store_true", help="Run L4 MuJoCo step alignment")
    parser.add_argument(
        "--align-case",
        default="baseline",
        help="Physics case for L4 (baseline|friction_div2|mass_x2|...)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write JSON report path")
    args = parser.parse_args()

    data_dir: Path = args.data
    train_npz = data_dir / "train" / "windows.npz"
    if not train_npz.exists():
        if not args.export_if_missing:
            print(
                f"Missing {train_npz}\n"
                "Re-run export, or pass --export-if-missing",
                file=sys.stderr,
            )
            sys.exit(2)
        if not _workspace_ready():
            print(
                "Ketchup workspace missing. Run:\n"
                "  bash scripts/setup_spider.sh\n"
                "  python3 scripts/build_spider_ketchup_right.py",
                file=sys.stderr,
            )
            sys.exit(2)
        rc = _maybe_export(quick=args.quick)
        if rc != 0:
            sys.exit(rc)

    splits = load_all_splits(data_dir)
    if not splits:
        print(f"No splits loaded from {data_dir}", file=sys.stderr)
        sys.exit(2)

    l2 = validate_l2(splits, val_cases=DEFAULT_VAL_CASES, test_cases=DEFAULT_TEST_CASES)
    l3 = validate_l3(splits, require_10k_train=not args.quick)
    _print_checks(l2)
    _print_checks(l3)
    table = per_case_label_table(splits)
    _print_case_table(table)

    reports = [l2, l3]
    if args.align:
        if not _workspace_ready():
            print("L4 skipped: ketchup workspace not ready", file=sys.stderr)
            sys.exit(2)
        mass_scale, friction_scale = 1.0, 1.0
        name = args.align_case
        if name.startswith("mass_x"):
            mass_scale = float(name.replace("mass_x", ""))
        elif name.startswith("friction_div"):
            friction_scale = 1.0 / float(name.replace("friction_div", ""))
        elif name != "baseline":
            print(f"Unknown --align-case {name}", file=sys.stderr)
            sys.exit(2)
        print(f"\nRunning L4 alignment on case={name} ...")
        l4 = run_l4_alignment(
            workspace_root=DEFAULT_WORKSPACE,
            spider_dataset_dir=SPIDER / "example_datasets",
            mass_scale=mass_scale,
            friction_scale=friction_scale,
        )
        _print_checks(l4)
        reports.append(l4)

    summary = merge_reports(*reports)
    summary["per_case"] = table
    summary["data_dir"] = str(data_dir)

    out_path = args.out or (data_dir / "validate_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nReport: {out_path}")
    print("OVERALL:", "PASS" if summary["ok"] else "FAIL")
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
