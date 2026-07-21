"""Optional closed-loop smoke gates (skip without MuJoCo ketchup scene)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SCENE = ROOT / "data" / "spider_ketchup_right" / "scene.xml"
CKPT_DIR = ROOT / "models" / "slip_nn"
TRAJ = (
    ROOT
    / "third_party"
    / "spider"
    / "example_datasets"
    / "processed"
    / "arcticv2"
    / "xhand"
    / "bimanual"
    / "s01-ketchup_use_01"
    / "0"
    / "trajectory_mjwp_fast.npz"
)


@pytest.mark.skipif(not SCENE.exists(), reason="ketchup scene not built")
@pytest.mark.skipif(not TRAJ.exists(), reason="spider ketchup trajectory missing")
@pytest.mark.skipif(not any(CKPT_DIR.glob("*.pt")), reason="no slip_nn checkpoint")
def test_nn1_closedloop_gates(tmp_path):
    from eval_slip_nn_closedloop import CASES, _print_table
    from run_ketchup_robustness_sweep import _run_case

    meta = json.loads((CKPT_DIR / "train_meta.json").read_text())
    thr = float(meta.get("default_threshold", 0.7))
    results = [
        _run_case(
            spec,
            save_video=False,
            antislip_nn=True,
            nn_model_dir=CKPT_DIR,
            nn_threshold=thr,
        )
        for spec in CASES
    ]
    _print_table(results)
    by_name = {r.name: r for r in results}
    base = by_name["baseline"]
    fd2 = by_name["friction_div2"]
    assert base.status == "pass"
    assert base.nn_slip_events < 100
    assert fd2.extend_dz_cm >= 6.0
    assert fd2.extend_contact_steps >= 200
