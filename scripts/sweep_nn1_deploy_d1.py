#!/usr/bin/env python3
"""NN-2 step D1: sweep τ × confirm_steps on v1 checkpoint (no retrain).

Writes data/slip_nn/deploy_sweep_d1.json.
Gates: baseline nn_slip < 50 (NN-2) / < 100 (NN-1); friction÷2 Δz≥6 & contacts≥200.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ketchup_robustness_sweep import CaseSpec, _run_case  # noqa: E402

OUT = ROOT / "data" / "slip_nn" / "deploy_sweep_d1.json"
CASES = [
    CaseSpec("baseline", sweep="baseline"),
    CaseSpec("friction_div2", friction_scale=0.5, sweep="friction"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 deploy sweep τ × confirm")
    parser.add_argument("--nn-model-dir", type=Path, default=ROOT / "models" / "slip_nn")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    taus = [0.7, 0.8, 0.85]
    confirms = [15, 20, 25]
    rows: list[dict] = []

    for tau in taus:
        for confirm in confirms:
            print(f"=== τ={tau} confirm={confirm} ===")
            for spec in CASES:
                r = _run_case(
                    spec,
                    save_video=False,
                    antislip_nn=True,
                    nn_model_dir=args.nn_model_dir,
                    nn_threshold=tau,
                    nn_confirm_steps=confirm,
                )
                rows.append(
                    {
                        "tau": tau,
                        "confirm_steps": confirm,
                        "case": r.name,
                        "status": r.status,
                        "extend_dz_cm": r.extend_dz_cm,
                        "extend_contact_steps": r.extend_contact_steps,
                        "nn_slip_events": r.nn_slip_events,
                        "antislip_max_grip": r.antislip_max_grip,
                        "nn2_ft_ok": r.name != "baseline" or r.nn_slip_events < 50,
                        "nn1_ft_ok": r.name != "baseline" or r.nn_slip_events < 100,
                        "fd2_ok": r.name != "friction_div2"
                        or (r.extend_dz_cm >= 6.0 and r.extend_contact_steps >= 200),
                    }
                )
                print(
                    f"  {r.name}: status={r.status} dz={r.extend_dz_cm:.2f} "
                    f"nn_slip={r.nn_slip_events} grip={r.antislip_max_grip:.2f}"
                )

    pairs: dict[tuple[float, int], dict] = {}
    for row in rows:
        key = (row["tau"], row["confirm_steps"])
        pairs.setdefault(key, {})[row["case"]] = row

    candidates = []
    for key, by_case in pairs.items():
        b = by_case.get("baseline")
        f = by_case.get("friction_div2")
        if not b or not f:
            continue
        ok_nn2 = (
            b["nn_slip_events"] < 50
            and f["extend_dz_cm"] >= 6.0
            and f["extend_contact_steps"] >= 200
        )
        ok_nn1 = (
            b["nn_slip_events"] < 100
            and f["extend_dz_cm"] >= 6.0
            and f["extend_contact_steps"] >= 200
        )
        candidates.append(
            {
                "tau": key[0],
                "confirm_steps": key[1],
                "baseline_nn_slip": b["nn_slip_events"],
                "fd2_dz_cm": f["extend_dz_cm"],
                "fd2_ok": f["extend_dz_cm"] >= 6.0 and f["extend_contact_steps"] >= 200,
                "nn1_gates_ok": ok_nn1,
                "nn2_gates_ok": ok_nn2,
            }
        )
    candidates.sort(
        key=lambda c: (
            not c["fd2_ok"],
            not c["nn2_gates_ok"],
            not c["nn1_gates_ok"],
            c["baseline_nn_slip"],
            -c["fd2_dz_cm"],
        )
    )

    summary = {
        "goal": {"baseline_nn_slip_max": 50, "fd2_dz_min_cm": 6.0},
        "rows": rows,
        "ranked": candidates,
        "best": candidates[0] if candidates else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")
    if candidates:
        print("Best:", json.dumps(candidates[0], indent=2))


if __name__ == "__main__":
    main()
