#!/usr/bin/env python3
"""Export Policy-2 open-loop hit actions as windowed training shards.

Reads ``data/slip_nn_policy2/search/*.json`` hits, replays each with a
SlipDatasetLogger, writes ``data/slip_nn_policy2/{train,val}/`` + labels
``y_grip, y_wr, y_wp, y_wy`` (constant teacher over extend windows).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.antislip_control import Policy2Action, Policy2OpenLoopController  # noqa: E402
from sim.slip_dataset_logger import (  # noqa: E402
    SlipDatasetLogger,
    compute_norm_stats,
    merge_npz_shards,
    write_manifest,
)
from sim.slip_nn_features import FEATURE_DIM, SlipFeatureBuilder  # noqa: E402
from sim.spider_ketchup import DEFAULT_WORKSPACE  # noqa: E402
from sim.spider_replay import SpiderTaskConfig, replay_spider_task  # noqa: E402

SPIDER = ROOT / "third_party" / "spider"
SEARCH_DIR = ROOT / "data" / "slip_nn_policy2" / "search"
OUT_DIR = ROOT / "data" / "slip_nn_policy2"

EXTEND_S = 2.0
LIFT_M = 0.10
WINDOW_STEPS = 40


def _cfg() -> SpiderTaskConfig:
    return SpiderTaskConfig(
        dataset_dir=SPIDER / "example_datasets",
        dataset_name="arcticv2",
        robot_type="xhand",
        embodiment_type="right",
        task="s01-ketchup_use_01",
        workspace_root=DEFAULT_WORKSPACE,
    )


def _load_hits(search_dir: Path, max_total: int) -> list[dict]:
    """Load hits from hits_pool.json (preferred) or per-case search JSONs."""
    pool = search_dir / "hits_pool.json"
    rows: list[dict] = []
    if pool.exists():
        pack = json.loads(pool.read_text())
        for h in pack.get("hits") or []:
            rows.append({"case": h["case"], "hit": h})
            if len(rows) >= max_total:
                break
        return rows

    for path in sorted(search_dir.glob("friction_*.json")):
        if "s040" in path.name:
            continue
        pack = json.loads(path.read_text())
        case = pack["case"]
        hits = sorted(pack.get("hits") or [], key=lambda h: h.get("score", 0), reverse=True)
        for h in hits:
            rows.append({"case": case, "hit": h})
            if len(rows) >= max_total:
                return rows
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Policy-2 hit trajectories")
    parser.add_argument("--search-dir", type=Path, default=SEARCH_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--max-hits", type=int, default=2000, help="Max PASS hits to export")
    parser.add_argument("--max-per-case", type=int, default=0, help="Deprecated; use --max-hits")
    parser.add_argument(
        "--one-window-per-hit",
        action="store_true",
        default=True,
        help="Keep only the last window per hit (recommended for large N)",
    )
    parser.add_argument(
        "--all-windows",
        action="store_true",
        help="Keep every sliding window (heavy for N~2000)",
    )
    parser.add_argument("--g-max", type=float, default=0.25)
    parser.add_argument("--d-max", type=float, default=0.25)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.all_windows:
        args.one_window_per_hit = False

    hits = _load_hits(args.search_dir, args.max_hits)
    if not hits:
        print(f"No hits in {args.search_dir}", file=sys.stderr)
        sys.exit(2)

    shard_dir = args.out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    meta_rows = []

    for i, row in enumerate(hits):
        case = row["case"]
        hit = row["hit"]
        act = Policy2Action(
            grip=float(hit["action"]["grip"]),
            wrist_delta=tuple(hit["action"]["wrist_delta"]),
        )
        # Skip pure-grip (should already be filtered in pool)
        if max(abs(x) for x in act.wrist_delta) < 0.02:
            continue
        name = f"{case['name']}_hit{i:04d}"
        motion = hit.get("motion") or {"extend_s": EXTEND_S, "lift_m": LIFT_M}
        extend_s = float(motion.get("extend_s", EXTEND_S))
        lift_m = float(motion.get("lift_m", LIFT_M))
        print(
            f"Export {name} g={act.grip:.3f} w={act.wrist_delta} "
            f"ext={extend_s:.1f}s lift={lift_m:.2f} "
            f"(dz={hit.get('extend_dz_cm', 0):+.1f}cm)",
            flush=True,
        )
        logger = SlipDatasetLogger(window_steps=WINDOW_STEPS)
        builder = SlipFeatureBuilder(sim_dt=0.01)
        ctrl = Policy2OpenLoopController(
            act, g_max=max(args.g_max, act.grip), d_max=args.d_max
        )
        replay_spider_task(
            _cfg(),
            args.out / "replay_logs" / name,
            save_video=False,
            post_lift_m=lift_m,
            post_extend_s=extend_s,
            post_mimic_s=1.0,
            mass_scale=float(case.get("mass_scale", 1.0)),
            friction_scale=float(case["friction_scale"]),
            log_energy=False,
            antislip=False,
            antislip_nn=False,
            policy2_controller=ctrl,
            dataset_logger=logger,
            feature_builder=builder,
            dataset_case_name=name,
        )
        # Constant teacher labels for this episode
        n = len(logger._labels)
        y_g = np.full(n, act.grip, dtype=np.float32)
        y_wr = np.full(n, act.wrist_delta[0], dtype=np.float32)
        y_wp = np.full(n, act.wrist_delta[1], dtype=np.float32)
        y_wy = np.full(n, act.wrist_delta[2], dtype=np.float32)
        # Attach via save: SlipDatasetLogger may not know these keys — inject after windows
        shard = shard_dir / f"{name}.npz"
        # Use set_policy_grip for g; store wrist in side arrays after load
        if hasattr(logger, "set_policy_grip"):
            logger.set_policy_grip(y_g)
        n_win = logger.save_npz(shard)
        raw = dict(np.load(shard, allow_pickle=True))
        if args.one_window_per_hit and n_win > 0:
            raw = {k: (v[-1:] if hasattr(v, "shape") and len(v.shape) >= 1 else v) for k, v in raw.items()}
            n_win = 1
        raw["y_grip_p2"] = np.full(n_win, act.grip, dtype=np.float32)
        raw["y_wr"] = np.full(n_win, act.wrist_delta[0], dtype=np.float32)
        raw["y_wp"] = np.full(n_win, act.wrist_delta[1], dtype=np.float32)
        raw["y_wy"] = np.full(n_win, act.wrist_delta[2], dtype=np.float32)
        if "y_policy" not in raw:
            raw["y_policy"] = raw["y_grip_p2"]
        np.savez_compressed(shard, **raw)
        shards.append(shard)
        meta_rows.append(
            {
                "name": name,
                "case": case["name"],
                "friction_scale": case["friction_scale"],
                "action": hit["action"],
                "motion": motion,
                "extend_dz_cm": hit.get("extend_dz_cm"),
                "windows": n_win,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  ... exported {i+1}/{len(hits)}", flush=True)

    merged = merge_npz_shards(shards)
    if not merged:
        print("No windows", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    n = merged["X"].shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(round(args.val_frac * n)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def take(idxs):
        return {k: v[idxs] for k, v in merged.items()}

    train, val = take(train_idx), take(val_idx)
    for split, data in (("train", train), ("val", val)):
        d = args.out / split
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "windows.npz", **data)

    norm = compute_norm_stats(train["X"])
    write_manifest(
        args.out / "manifest.json",
        window_steps=WINDOW_STEPS,
        n_train=int(train["X"].shape[0]),
        n_val=int(val["X"].shape[0]),
        n_test=0,
        norm_stats=norm,
        extra={
            "feature_dim": FEATURE_DIM,
            "dataset": "slip_nn_policy2",
            "label_keys": ["y_grip_p2", "y_wr", "y_wp", "y_wy", "y_policy"],
            "source": "search_hits_openloop",
            "per_hit": meta_rows,
        },
    )
    summary = {
        "train": int(train["X"].shape[0]),
        "val": int(val["X"].shape[0]),
        "n_hits_exported": len(meta_rows),
        "one_window_per_hit": bool(args.one_window_per_hit),
        "cases": sorted({m["case"] for m in meta_rows}),
        "wrist_nonzero_frac": float(
            np.mean(
                [
                    max(abs(x) for x in m["action"]["wrist_delta"]) >= 0.02
                    for m in meta_rows
                ]
            )
        )
        if meta_rows
        else 0.0,
        "manifest": str(args.out / "manifest.json"),
    }
    (args.out / "export_summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "hits_catalog.json").write_text(json.dumps(meta_rows, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
