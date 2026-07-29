#!/usr/bin/env python3
"""Policy-2 stage-0 teacher search: prove (g*, Δwrist) can rescue hard μ.

Searches open-loop P2-A actions during extend (no NN). Writes hits under
``data/slip_nn_policy2/search/``. Success criterion (spec): μ×0.40 ≥1 PASS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.antislip_control import Policy2Action, Policy2OpenLoopController  # noqa: E402
from sim.spider_ketchup import DEFAULT_WORKSPACE  # noqa: E402
from sim.spider_replay import SpiderTaskConfig, replay_spider_task  # noqa: E402

SPIDER = ROOT / "third_party" / "spider"
OUT_DIR = ROOT / "data" / "slip_nn_policy2" / "search"

EXTEND_S = 2.0
LIFT_M = 0.10
EXTEND_STEPS = 200
DZ_MIN_M = 0.06
DZ_MARGIN_M = 0.07
DROP_MAX_M = 0.03


@dataclass(frozen=True)
class SearchCase:
    name: str
    mass_scale: float = 1.0
    friction_scale: float = 1.0


DEFAULT_CASES = (
    SearchCase("friction_div2", friction_scale=0.50),
    SearchCase("friction_s045", friction_scale=0.45),
    SearchCase("friction_s040", friction_scale=0.40),
)


def _friction_tag(friction_scale: float) -> str:
    if abs(friction_scale - 1.0) < 1e-9:
        return ""
    if abs(friction_scale - 0.5) < 1e-9:
        return "friction_div2"
    if abs(friction_scale - 0.45) < 1e-9:
        return "friction_s045"
    if abs(friction_scale - 0.40) < 1e-9:
        return "friction_s040"
    return f"friction_x{friction_scale:g}"


def _mass_tag(mass_scale: float) -> str:
    if abs(mass_scale - 1.0) < 1e-9:
        return ""
    if abs(mass_scale - int(mass_scale)) < 1e-9:
        return f"mass_x{int(mass_scale)}"
    return f"mass_x{mass_scale:g}"


def build_search_cases(
    *,
    mass_scales: list[float] | None = None,
    friction_scales: list[float] | None = None,
    names: list[str] | None = None,
) -> list[SearchCase]:
    """Build SearchCase grid. Empty mass/friction → defaults to friction-only DEFAULT_CASES."""
    if names:
        by_name = {c.name: c for c in DEFAULT_CASES}
        # Also allow reconstructing mass/friction from known tags later.
        out: list[SearchCase] = []
        for n in names:
            if n in by_name:
                out.append(by_name[n])
                continue
            # mass_x2 / mass_x4 / mass_x2_friction_div2
            m, f = 1.0, 1.0
            if "mass_x" in n:
                rest = n.split("mass_x", 1)[1]
                num = rest.split("_", 1)[0]
                m = float(num)
            if "friction_div2" in n:
                f = 0.5
            elif "friction_s045" in n:
                f = 0.45
            elif "friction_s040" in n:
                f = 0.40
            out.append(SearchCase(n, mass_scale=m, friction_scale=f))
        return out

    if not mass_scales and not friction_scales:
        return list(DEFAULT_CASES)

    masses = mass_scales if mass_scales else [1.0]
    frictions = friction_scales if friction_scales else [1.0]
    cases: list[SearchCase] = []
    for m in masses:
        for f in frictions:
            mt, ft = _mass_tag(m), _friction_tag(f)
            parts = [p for p in (mt, ft) if p]
            name = "_".join(parts) if parts else "baseline"
            cases.append(SearchCase(name, mass_scale=float(m), friction_scale=float(f)))
    return cases


def parse_float_list(raw: str) -> list[float]:
    if not raw or not raw.strip():
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


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


def _score(result, *, dz_min_m: float) -> float:
    """Higher is better; used by CEM even when not PASS."""
    drop = max(0.0, result.object_z_after_trajectory - result.object_z_end)
    dz = float(result.post_extend_object_dz)
    ct = float(result.post_extend_contact_steps) / float(EXTEND_STEPS)
    # Soft score: reward lift + contacts, penalize drop.
    score = 10.0 * dz + 2.0 * ct - 5.0 * drop
    if _passes(result, dz_min_m=dz_min_m):
        score += 50.0
    return float(score)


def _eval_action(
    case: SearchCase,
    action: Policy2Action,
    *,
    g_max: float,
    d_max: float,
    rate_g: float,
    rate_w: float,
    out_sub: str,
    dz_min_m: float,
) -> dict:
    ctrl = Policy2OpenLoopController(
        action,
        g_max=g_max,
        d_max=d_max,
        rate_g=rate_g,
        rate_w=rate_w,
    )
    result = replay_spider_task(
        _cfg(),
        OUT_DIR / "replay_logs" / out_sub / case.name,
        save_video=False,
        post_lift_m=LIFT_M,
        post_extend_s=EXTEND_S,
        post_mimic_s=1.0,
        mass_scale=case.mass_scale,
        friction_scale=case.friction_scale,
        log_energy=False,
        antislip=False,
        antislip_nn=False,
        policy2_controller=ctrl,
        dataset_case_name=case.name,
    )
    ok = _passes(result, dz_min_m=dz_min_m)
    return {
        "action": {
            "grip": action.grip,
            "wrist_delta": list(action.wrist_delta),
        },
        "pass": bool(ok),
        "score": _score(result, dz_min_m=dz_min_m),
        "extend_dz_cm": float(result.post_extend_object_dz * 100),
        "extend_contact_steps": int(result.post_extend_contact_steps),
        "max_grip": float(result.antislip_max_grip),
        "drop_cm": float(
            max(0.0, result.object_z_after_trajectory - result.object_z_end) * 100
        ),
    }


def _sample_uniform(
    rng: np.random.Generator,
    *,
    n: int,
    g_max: float,
    d_max: float,
    g_min: float = 0.05,
) -> list[Policy2Action]:
    acts = []
    for _ in range(n):
        g = float(rng.uniform(g_min, g_max))
        w = rng.uniform(-d_max, d_max, size=3)
        acts.append(Policy2Action(grip=g, wrist_delta=tuple(float(x) for x in w)))
    return acts


def _sample_cem(
    rng: np.random.Generator,
    *,
    n: int,
    mean: np.ndarray,
    std: np.ndarray,
    g_max: float,
    d_max: float,
) -> list[Policy2Action]:
    acts = []
    lo = np.array([0.0, -d_max, -d_max, -d_max])
    hi = np.array([g_max, d_max, d_max, d_max])
    for _ in range(n):
        v = rng.normal(mean, std)
        v = np.clip(v, lo, hi)
        acts.append(Policy2Action.from_vector(v))
    return acts


def _seed_actions(g_max: float, d_max: float) -> list[Policy2Action]:
    """Hand priors: high grip ± single-axis wrist tilts."""
    seeds = [
        Policy2Action(grip=g_max, wrist_delta=(0.0, 0.0, 0.0)),
        Policy2Action(grip=min(0.25, g_max), wrist_delta=(0.0, 0.0, 0.0)),
    ]
    for ax in range(3):
        for s in (-1.0, 1.0):
            w = [0.0, 0.0, 0.0]
            w[ax] = s * d_max
            seeds.append(Policy2Action(grip=g_max, wrist_delta=tuple(w)))
            seeds.append(Policy2Action(grip=g_max, wrist_delta=tuple(0.5 * x for x in w)))
    # mild two-axis combos
    for a, b in ((0, 1), (1, 2), (0, 2)):
        for sa in (-1.0, 1.0):
            for sb in (-1.0, 1.0):
                w = [0.0, 0.0, 0.0]
                w[a] = sa * 0.7 * d_max
                w[b] = sb * 0.7 * d_max
                seeds.append(Policy2Action(grip=g_max, wrist_delta=tuple(w)))
    return seeds


def search_case(
    case: SearchCase,
    *,
    method: str,
    n_samples: int,
    cem_iters: int,
    cem_elite: int,
    g_max: float,
    d_max: float,
    rate_g: float,
    rate_w: float,
    dz_min_m: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    trials: list[dict] = []
    t0 = time.time()

    def run_batch(actions: list[Policy2Action], tag: str) -> None:
        for i, act in enumerate(actions):
            row = _eval_action(
                case,
                act,
                g_max=g_max,
                d_max=d_max,
                rate_g=rate_g,
                rate_w=rate_w,
                out_sub=f"{tag}_{i:04d}",
                dz_min_m=dz_min_m,
            )
            row["tag"] = tag
            trials.append(row)
            mark = "PASS" if row["pass"] else "fail"
            print(
                f"  [{case.name}] {tag}#{i:03d} {mark} "
                f"g={act.grip:.3f} w=({act.wrist_delta[0]:+.2f},{act.wrist_delta[1]:+.2f},{act.wrist_delta[2]:+.2f}) "
                f"dz={row['extend_dz_cm']:+.1f}cm ct={row['extend_contact_steps']} "
                f"score={row['score']:.2f}",
                flush=True,
            )

    # Always evaluate seeds first (cheap priors).
    seeds = _seed_actions(g_max, d_max)
    run_batch(seeds, "seed")
    hits = [t for t in trials if t["pass"]]
    if hits and method == "seed":
        return _pack_case(case, trials, t0, method, g_max, d_max)

    if method in ("random", "both"):
        run_batch(
            _sample_uniform(rng, n=n_samples, g_max=g_max, d_max=d_max),
            "rand",
        )
        hits = [t for t in trials if t["pass"]]
        if hits and method == "random":
            return _pack_case(case, trials, t0, method, g_max, d_max)

    if method in ("cem", "both"):
        mean = np.array([0.8 * g_max, 0.0, 0.0, 0.0], dtype=np.float64)
        std = np.array([0.15 * g_max, 0.5 * d_max, 0.5 * d_max, 0.5 * d_max], dtype=np.float64)
        for it in range(cem_iters):
            batch = _sample_cem(
                rng, n=n_samples, mean=mean, std=std, g_max=g_max, d_max=d_max
            )
            run_batch(batch, f"cem{it}")
            # Elite update on this iter's trials
            recent = [t for t in trials if t["tag"] == f"cem{it}"]
            recent_sorted = sorted(recent, key=lambda r: r["score"], reverse=True)
            elite = recent_sorted[: max(1, cem_elite)]
            vecs = np.array(
                [
                    [e["action"]["grip"], *e["action"]["wrist_delta"]]
                    for e in elite
                ],
                dtype=np.float64,
            )
            mean = vecs.mean(axis=0)
            std = np.maximum(vecs.std(axis=0), 1e-3)
            if any(e["pass"] for e in elite):
                break

    return _pack_case(case, trials, t0, method, g_max, d_max)


def _pack_case(
    case: SearchCase,
    trials: list[dict],
    t0: float,
    method: str,
    g_max: float,
    d_max: float,
) -> dict:
    hits = [t for t in trials if t["pass"]]
    best = max(trials, key=lambda r: r["score"]) if trials else None
    return {
        "case": asdict(case),
        "method": method,
        "g_max": g_max,
        "d_max": d_max,
        "n_trials": len(trials),
        "n_hits": len(hits),
        "solvable": len(hits) > 0,
        "elapsed_s": time.time() - t0,
        "best": best,
        "hits": hits,
        "trials": trials,
    }


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Policy-2 open-loop teacher search")
    parser.add_argument(
        "--method",
        choices=("seed", "random", "cem", "both"),
        default="both",
        help="seed priors + random and/or CEM",
    )
    parser.add_argument("--n-samples", type=int, default=24, help="samples per random/CEM round")
    parser.add_argument("--cem-iters", type=int, default=4)
    parser.add_argument("--cem-elite", type=int, default=4)
    parser.add_argument("--g-max", type=float, default=0.25)
    parser.add_argument("--d-max", type=float, default=0.25, help="wrist residual max |rad|")
    parser.add_argument("--rate-g", type=float, default=0.02)
    parser.add_argument("--rate-w", type=float, default=0.02)
    parser.add_argument("--dz-min", type=float, default=DZ_MIN_M)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--case",
        default="",
        help="Single case name, or empty to use --mass-scales/--friction-scales/defaults",
    )
    parser.add_argument(
        "--mass-scales",
        default="",
        help="Comma list e.g. 2,4 (with --friction-scales builds a grid)",
    )
    parser.add_argument(
        "--friction-scales",
        default="",
        help="Comma list e.g. 1.0,0.5 (default 1.0 when mass-scales set)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for per-case JSON + summary",
    )
    parser.add_argument("--out", type=Path, default=None, help="Summary JSON path")
    parser.add_argument(
        "--expand",
        action="store_true",
        help="If default bounds fail on s040, retry g_max=0.35 d_max=0.5",
    )
    args = parser.parse_args()

    if not DEFAULT_WORKSPACE.joinpath("scene.xml").exists():
        print("Missing ketchup workspace. Run setup_spider + build.", file=sys.stderr)
        sys.exit(1)

    # Allow writing under alternate out-dir (heavy+gripcap datasets).
    OUT_DIR = Path(args.out_dir) if args.out_dir is not None else OUT_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        args.out = OUT_DIR / "search_summary.json"

    if args.case:
        cases = build_search_cases(names=[args.case])
    else:
        cases = build_search_cases(
            mass_scales=parse_float_list(args.mass_scales) or None,
            friction_scales=parse_float_list(args.friction_scales) or None,
        )
    if not cases:
        print("No search cases resolved", file=sys.stderr)
        sys.exit(2)

    results = []
    for case in cases:
        print(
            f"\n=== Search {case.name} mass×{case.mass_scale} μ×{case.friction_scale} "
            f"g_max={args.g_max} ===",
            flush=True,
        )
        pack = search_case(
            case,
            method=args.method,
            n_samples=args.n_samples,
            cem_iters=args.cem_iters,
            cem_elite=args.cem_elite,
            g_max=args.g_max,
            d_max=args.d_max,
            rate_g=args.rate_g,
            rate_w=args.rate_w,
            dz_min_m=args.dz_min,
            seed=args.seed + hash(case.name) % 1000,
        )
        if (
            args.expand
            and case.friction_scale <= 0.40 + 1e-9
            and not pack["solvable"]
        ):
            print(f"  expand bounds for {case.name}: g_max=0.35 d_max=0.50", flush=True)
            pack2 = search_case(
                case,
                method=args.method,
                n_samples=args.n_samples,
                cem_iters=args.cem_iters,
                cem_elite=args.cem_elite,
                g_max=0.35,
                d_max=0.50,
                rate_g=args.rate_g,
                rate_w=args.rate_w,
                dz_min_m=args.dz_min,
                seed=args.seed + 17 + hash(case.name) % 1000,
            )
            pack2["expanded_bounds"] = True
            pack = pack2
        results.append(pack)
        # Persist per-case (hits only + best) for dataset building
        slim = {k: v for k, v in pack.items() if k != "trials"}
        (OUT_DIR / f"{case.name}.json").write_text(json.dumps(slim, indent=2))

    s040 = next((r for r in results if r["case"]["name"] == "friction_s040"), None)
    summary = {
        "spec": "NN-Policy-2 stage-0 teacher search",
        "method": args.method,
        "g_max": args.g_max,
        "d_max": args.d_max,
        "dz_min_m": args.dz_min,
        "seed": args.seed,
        "s040_solvable": bool(s040["solvable"]) if s040 else None,
        "cases": [
            {
                "name": r["case"]["name"],
                "friction_scale": r["case"]["friction_scale"],
                "solvable": r["solvable"],
                "n_hits": r["n_hits"],
                "n_trials": r["n_trials"],
                "best_dz_cm": (r["best"] or {}).get("extend_dz_cm"),
                "best_action": (r["best"] or {}).get("action"),
                "expanded_bounds": r.get("expanded_bounds", False),
            }
            for r in results
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Full dump without every fail trial to keep size sane
    dump = {
        "summary": summary,
        "results": [{k: v for k, v in r.items() if k != "trials"} for r in results],
    }
    args.out.write_text(json.dumps(dump, indent=2))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")
    if s040 is not None and not s040["solvable"]:
        print(
            "NOTE: μ×0.40 not solved under tried bounds — "
            "per spec, do not train a policy yet; expand physics/action or accept OOD.",
            file=sys.stderr,
        )
        sys.exit(3)


if __name__ == "__main__":
    main()
