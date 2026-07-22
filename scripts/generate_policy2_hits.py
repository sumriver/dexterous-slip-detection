#!/usr/bin/env python3
"""Bulk-generate Policy-2 open-loop PASS hits (target ~2000).

Strategy:
  1) Seed from existing search hits + axis priors
  2) Perturb elites + uniform/CEM samples on solvable μ (÷2, s045)
  3) Optional motion variants (extend_s / lift_m) for diversity
  4) Dedup quantized actions; stop at ``--target`` PASSes

Writes ``data/slip_nn_policy2/search/hits_pool.json`` (+ per-case mirrors).
Does **not** attempt s040 (known unsolvable under P2-A bounds).
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
sys.path.insert(0, str(ROOT / "scripts"))

from search_policy2_teacher import (  # noqa: E402
    DROP_MAX_M,
    DZ_MIN_M,
    SearchCase,
    _cfg,
    _sample_cem,
    _sample_uniform,
    _seed_actions,
)
from sim.antislip_control import Policy2Action, Policy2OpenLoopController  # noqa: E402
from sim.spider_replay import replay_spider_task  # noqa: E402

OUT_DIR = ROOT / "data" / "slip_nn_policy2" / "search"

SOLVABLE_CASES = (
    SearchCase("friction_div2", friction_scale=0.50),
    SearchCase("friction_s045", friction_scale=0.45),
)

# Motion variants (same as NN-0 extend variants) to multiply coverage.
MOTION_VARIANTS: tuple[tuple[str, float, float], ...] = (
    ("", 2.0, 0.10),
    ("_v1", 1.5, 0.08),
    ("_v2", 2.5, 0.12),
    ("_v3", 2.0, 0.14),
)


@dataclass(frozen=True)
class MotionSpec:
    suffix: str
    extend_s: float
    lift_m: float


def _quantize_key(action: Policy2Action, *, g_q: float = 0.01, w_q: float = 0.02) -> tuple:
    g = round(action.grip / g_q) * g_q
    w = tuple(round(x / w_q) * w_q for x in action.wrist_delta)
    return (round(g, 4),) + tuple(round(v, 4) for v in w)


def _load_seed_actions(search_dir: Path) -> list[Policy2Action]:
    acts: list[Policy2Action] = []
    for path in sorted(search_dir.glob("friction_*.json")):
        if "s040" in path.name:
            continue
        pack = json.loads(path.read_text())
        for h in pack.get("hits") or []:
            a = h.get("action") or {}
            acts.append(
                Policy2Action(
                    grip=float(a["grip"]),
                    wrist_delta=tuple(float(x) for x in a["wrist_delta"]),
                )
            )
    # Also seeds from teacher search priors
    acts.extend(_seed_actions(g_max=0.25, d_max=0.25))
    return acts


def _perturb(
    rng: np.random.Generator,
    base: Policy2Action,
    *,
    n: int,
    g_max: float,
    d_max: float,
    g_sigma: float = 0.03,
    w_sigma: float = 0.06,
) -> list[Policy2Action]:
    out = []
    for _ in range(n):
        g = float(np.clip(base.grip + rng.normal(0, g_sigma), 0.05, g_max))
        w = np.clip(
            np.asarray(base.wrist_delta, dtype=np.float64) + rng.normal(0, w_sigma, size=3),
            -d_max,
            d_max,
        )
        out.append(Policy2Action(grip=g, wrist_delta=tuple(float(x) for x in w)))
    return out


def _eval_with_motion(
    case: SearchCase,
    motion: MotionSpec,
    action: Policy2Action,
    *,
    g_max: float,
    d_max: float,
    rate_g: float,
    rate_w: float,
    dz_min_m: float,
    out_sub: str,
) -> dict:
    """Like _eval_action but with configurable extend/lift."""
    ctrl = Policy2OpenLoopController(
        action, g_max=g_max, d_max=d_max, rate_g=rate_g, rate_w=rate_w
    )
    result = replay_spider_task(
        _cfg(),
        OUT_DIR / "replay_logs" / out_sub / f"{case.name}{motion.suffix}",
        save_video=False,
        post_lift_m=motion.lift_m,
        post_extend_s=motion.extend_s,
        post_mimic_s=1.0,
        mass_scale=case.mass_scale,
        friction_scale=case.friction_scale,
        log_energy=False,
        antislip=False,
        antislip_nn=False,
        policy2_controller=ctrl,
        dataset_case_name=f"{case.name}{motion.suffix}",
    )
    # Contact budget scales with extend duration (10ms step).
    extend_steps = max(1, int(round(motion.extend_s / 0.01)))
    drop = max(0.0, result.object_z_after_trajectory - result.object_z_end)
    ok = (
        result.post_extend_object_dz >= dz_min_m
        and result.post_extend_contact_steps >= extend_steps
        and drop <= DROP_MAX_M
    )
    dz = float(result.post_extend_object_dz)
    ct = float(result.post_extend_contact_steps) / float(extend_steps)
    score = 10.0 * dz + 2.0 * ct - 5.0 * drop + (50.0 if ok else 0.0)
    return {
        "action": {"grip": action.grip, "wrist_delta": list(action.wrist_delta)},
        "pass": bool(ok),
        "score": float(score),
        "extend_dz_cm": float(result.post_extend_object_dz * 100),
        "extend_contact_steps": int(result.post_extend_contact_steps),
        "extend_steps_required": extend_steps,
        "max_grip": float(result.antislip_max_grip),
        "drop_cm": float(drop * 100),
        "motion": {"suffix": motion.suffix, "extend_s": motion.extend_s, "lift_m": motion.lift_m},
        "case": asdict(case),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-generate Policy-2 PASS hits")
    parser.add_argument("--target", type=int, default=2000, help="Target number of unique PASS hits")
    parser.add_argument("--g-max", type=float, default=0.25)
    parser.add_argument("--d-max", type=float, default=0.25)
    parser.add_argument("--rate-g", type=float, default=0.02)
    parser.add_argument("--rate-w", type=float, default=0.02)
    parser.add_argument("--dz-min", type=float, default=DZ_MIN_M)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=64, help="Candidates per round")
    parser.add_argument("--max-trials", type=int, default=8000)
    parser.add_argument("--no-motion-variants", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "hits_pool.json")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true", help="Continue from existing hits_pool.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    motions = (
        [MotionSpec(*m) for m in MOTION_VARIANTS]
        if not args.no_motion_variants
        else [MotionSpec("", 2.0, 0.10)]
    )

    elites = _load_seed_actions(OUT_DIR)
    seen: set[tuple] = set()
    hits: list[dict] = []
    n_trials = 0
    t0 = time.time()

    if args.resume and args.out.exists():
        prev = json.loads(args.out.read_text())
        hits = list(prev.get("hits") or [])
        for h in hits:
            act = Policy2Action(
                grip=float(h["action"]["grip"]),
                wrist_delta=tuple(float(x) for x in h["action"]["wrist_delta"]),
            )
            key = _quantize_key(act) + (
                h["case"]["name"],
                h["motion"]["suffix"],
                round(h["motion"]["extend_s"], 2),
                round(h["motion"]["lift_m"], 3),
            )
            seen.add(key)
            elites.append(act)
        print(f"Resumed {len(hits)} hits from {args.out}", flush=True)

    def accept(row: dict, action: Policy2Action) -> bool:
        nonlocal n_trials
        n_trials += 1
        key = _quantize_key(action) + (
            row["case"]["name"],
            row["motion"]["suffix"],
            round(row["motion"]["extend_s"], 2),
            round(row["motion"]["lift_m"], 3),
        )
        if key in seen:
            return False
        seen.add(key)
        if not row["pass"]:
            return False
        # Require non-trivial wrist for Policy-2 (at least one |Δ|≥0.02)
        if max(abs(x) for x in action.wrist_delta) < 0.02:
            return False
        hits.append(row)
        elites.append(action)
        return True

    print(
        f"Target={args.target} cases={[c.name for c in SOLVABLE_CASES]} "
        f"motions={len(motions)} g_max={args.g_max} d_max={args.d_max}",
        flush=True,
    )

    # Round 0: evaluate seed elites × motions (cap)
    seed_acts = elites[:80]
    for case in SOLVABLE_CASES:
        for motion in motions:
            for i, act in enumerate(seed_acts):
                if len(hits) >= args.target or n_trials >= args.max_trials:
                    break
                row = _eval_with_motion(
                    case,
                    motion,
                    act,
                    g_max=args.g_max,
                    d_max=args.d_max,
                    rate_g=args.rate_g,
                    rate_w=args.rate_w,
                    dz_min_m=args.dz_min,
                    out_sub=f"bulk_seed_{len(hits):04d}",
                )
                accept(row, act)
            if len(hits) >= args.target:
                break
        if len(hits) >= args.target:
            break
    print(f"After seeds: hits={len(hits)} trials={n_trials}", flush=True)

    # Main loop: perturb + random + light CEM
    mean = np.array([0.2, 0.0, -0.05, 0.18], dtype=np.float64)
    std = np.array([0.05, 0.12, 0.12, 0.08], dtype=np.float64)
    round_id = 0
    while len(hits) < args.target and n_trials < args.max_trials:
        round_id += 1
        case = SOLVABLE_CASES[round_id % len(SOLVABLE_CASES)]
        motion = motions[round_id % len(motions)]

        cands: list[Policy2Action] = []
        # 50% perturb elites
        if elites:
            bases = [elites[int(rng.integers(0, len(elites)))] for _ in range(args.batch // 2)]
            for b in bases:
                cands.extend(
                    _perturb(rng, b, n=1, g_max=args.g_max, d_max=args.d_max)
                )
        # 25% uniform
        cands.extend(
            _sample_uniform(
                rng, n=max(1, args.batch // 4), g_max=args.g_max, d_max=args.d_max
            )
        )
        # 25% CEM-like
        cands.extend(
            _sample_cem(
                rng,
                n=max(1, args.batch // 4),
                mean=mean,
                std=std,
                g_max=args.g_max,
                d_max=args.d_max,
            )
        )

        newly = 0
        scored = []
        for act in cands:
            if len(hits) >= args.target or n_trials >= args.max_trials:
                break
            row = _eval_with_motion(
                case,
                motion,
                act,
                g_max=args.g_max,
                d_max=args.d_max,
                rate_g=args.rate_g,
                rate_w=args.rate_w,
                dz_min_m=args.dz_min,
                out_sub=f"bulk_{n_trials:05d}",
            )
            scored.append((row["score"], act, row))
            if accept(row, act):
                newly += 1

        # Update CEM stats from top scored this round
        scored.sort(key=lambda t: t[0], reverse=True)
        elite_vecs = []
        for _, act, row in scored[:8]:
            elite_vecs.append([act.grip, *act.wrist_delta])
        if elite_vecs:
            mat = np.asarray(elite_vecs, dtype=np.float64)
            mean = 0.7 * mean + 0.3 * mat.mean(axis=0)
            std = np.maximum(0.7 * std + 0.3 * mat.std(axis=0), 0.02)

        if round_id % 1 == 0:
            print(
                f"round={round_id:04d} case={case.name}{motion.suffix} "
                f"hits={len(hits)}/{args.target} trials={n_trials} "
                f"+{newly} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
        # Checkpoint every round after seed phase
        if round_id % 5 == 0 or len(hits) >= args.target:
            _write_pool(args.out, hits, n_trials, t0, args)

    _write_pool(args.out, hits, n_trials, t0, args)
    print(
        f"Done hits={len(hits)} trials={n_trials} elapsed={time.time()-t0:.1f}s -> {args.out}",
        flush=True,
    )
    if len(hits) < args.target:
        print(
            f"WARNING: only {len(hits)}/{args.target} unique wrist≠0 PASSes "
            f"(trials cap {args.max_trials})",
            file=sys.stderr,
        )
        sys.exit(3)


def _write_pool(path: Path, hits: list[dict], n_trials: int, t0: float, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # wrist stats
    wr = np.array([h["action"]["wrist_delta"] for h in hits], dtype=np.float64) if hits else np.zeros((0, 3))
    summary = {
        "n_hits": len(hits),
        "n_trials": n_trials,
        "elapsed_s": time.time() - t0,
        "target": args.target,
        "g_max": args.g_max,
        "d_max": args.d_max,
        "cases": sorted({h["case"]["name"] for h in hits}),
        "motions": sorted({h["motion"]["suffix"] or "base" for h in hits}),
        "wrist_abs_mean": np.abs(wr).mean(axis=0).tolist() if len(wr) else [0, 0, 0],
        "grip_mean": float(np.mean([h["action"]["grip"] for h in hits])) if hits else 0.0,
        "all_wrist_nonzero": bool(
            all(max(abs(x) for x in h["action"]["wrist_delta"]) >= 0.02 for h in hits)
        )
        if hits
        else False,
    }
    path.write_text(json.dumps({"summary": summary, "hits": hits}, indent=2))
    # slim per-case files for export compatibility
    by_case: dict[str, list] = {}
    for h in hits:
        by_case.setdefault(h["case"]["name"], []).append(h)
    for name, hs in by_case.items():
        case = hs[0]["case"]
        pack = {
            "case": case,
            "method": "bulk_generate",
            "n_hits": len(hs),
            "solvable": True,
            "hits": hs,
            "best": max(hs, key=lambda r: r["score"]),
        }
        (path.parent / f"{name}.json").write_text(json.dumps(pack, indent=2))
    (path.parent / "hits_pool_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  checkpoint hits={len(hits)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
