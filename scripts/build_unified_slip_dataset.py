#!/usr/bin/env python3
"""Build a unified train/test dataset from slip_nn / policy / policy2 / heavy.

Fairness: one shared feature schema + one shared label schema, then retrain
NN-2 / P1 / P2-grip-only / P2-grip+wrist on the **same** windows.

``GCD`` here means schema intersection (not row intersection):
  - features: 26-D windows (identical across sources)
  - labels: y_event, y_grip/y_policy, y_wr/y_wp/y_wy
    (missing wrist → 0; missing slip → 0)

Sources are unioned (coverage), then split train/test by stratified case.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_HIT_SUFFIX = re.compile(r"_hit\d+$", re.IGNORECASE)


def _base_case(name: str) -> str:
    """Strip per-hit suffixes so policy2 hits share a stratum with the case."""
    return _HIT_SUFFIX.sub("", str(name))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.slip_nn_features import FEATURE_DIM  # noqa: E402
from sim.slip_dataset_logger import compute_norm_stats, write_manifest  # noqa: E402

OUT_DEFAULT = ROOT / "data" / "slip_nn_unified"
WINDOW_STEPS = 40

UNIFIED_LABELS = (
    "y_event",
    "y_grip",
    "y_policy",
    "y_grip_p2",
    "y_wr",
    "y_wp",
    "y_wy",
)


def _as_f32(a, n: int, default: float = 0.0) -> np.ndarray:
    if a is None:
        return np.full(n, default, dtype=np.float32)
    out = np.asarray(a)
    if out.shape[0] != n:
        raise ValueError(f"length mismatch: got {out.shape[0]} want {n}")
    return out.astype(np.float32).reshape(n)


def _as_obj(a, n: int, default: str = "") -> np.ndarray:
    if a is None:
        return np.array([default] * n, dtype=object)
    out = np.asarray(a, dtype=object).reshape(-1)
    if out.shape[0] != n:
        raise ValueError(f"length mismatch: got {out.shape[0]} want {n}")
    return out


def _load_source(path: Path, source_name: str) -> dict[str, np.ndarray]:
    """Load train+val(+test) windows from a source dataset root."""
    chunks: list[dict[str, np.ndarray]] = []
    for split in ("train", "val", "test"):
        p = path / split / "windows.npz"
        if not p.exists():
            continue
        raw = dict(np.load(p, allow_pickle=True))
        if "X" not in raw:
            raise KeyError(f"{p} missing X")
        n = int(raw["X"].shape[0])
        if raw["X"].shape[1:] != (WINDOW_STEPS, FEATURE_DIM):
            raise ValueError(f"{p} bad X shape {raw['X'].shape}")

        # Slip event label (detect)
        if "y_event" in raw:
            y_event = _as_f32(raw["y_event"], n)
        elif "y_fused" in raw:
            y_event = _as_f32(raw["y_fused"], n)
        else:
            y_event = np.zeros(n, dtype=np.float32)

        # Grip teachers — prefer explicit policy / p2 grip, else y_grip
        if "y_grip_p2" in raw:
            y_grip = _as_f32(raw["y_grip_p2"], n)
        elif "y_policy" in raw:
            y_grip = _as_f32(raw["y_policy"], n)
        elif "y_grip" in raw:
            y_grip = _as_f32(raw["y_grip"], n)
        else:
            y_grip = np.zeros(n, dtype=np.float32)

        y_wr = _as_f32(raw["y_wr"], n) if "y_wr" in raw else np.zeros(n, dtype=np.float32)
        y_wp = _as_f32(raw["y_wp"], n) if "y_wp" in raw else np.zeros(n, dtype=np.float32)
        y_wy = _as_f32(raw["y_wy"], n) if "y_wy" in raw else np.zeros(n, dtype=np.float32)

        mass = _as_f32(raw["mass_scale"], n, 1.0) if "mass_scale" in raw else np.ones(n, dtype=np.float32)
        fric = (
            _as_f32(raw["friction_scale"], n, 1.0)
            if "friction_scale" in raw
            else np.ones(n, dtype=np.float32)
        )
        case = _as_obj(raw["case_name"], n, source_name) if "case_name" in raw else _as_obj(None, n, source_name)

        chunks.append(
            {
                "X": raw["X"].astype(np.float32),
                "y_event": y_event,
                "y_grip": y_grip,
                "y_policy": y_grip.copy(),
                "y_grip_p2": y_grip.copy(),
                "y_wr": y_wr,
                "y_wp": y_wp,
                "y_wy": y_wy,
                "mass_scale": mass,
                "friction_scale": fric,
                "case_name": case,
                "source_dataset": np.array([source_name] * n, dtype=object),
                "has_wrist_teacher": (
                    np.abs(np.stack([y_wr, y_wp, y_wy], axis=-1)).max(axis=-1) >= 0.02
                ).astype(np.float32),
                "has_slip_teacher": (y_event > 0).astype(np.float32),
            }
        )
        print(f"  loaded {source_name}/{split}: n={n}", flush=True)

    if not chunks:
        raise FileNotFoundError(f"No windows.npz under {path}/{{train,val,test}}")

    keys = chunks[0].keys()
    return {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}


def _stratified_split(
    n: int,
    keys: np.ndarray,
    *,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for k in sorted(set(str(x) for x in keys.tolist())):
        idxs = np.where(keys.astype(str) == k)[0]
        rng.shuffle(idxs)
        n_test = max(1, int(round(test_frac * len(idxs)))) if len(idxs) >= 5 else max(0, int(round(test_frac * len(idxs))))
        if len(idxs) == 1:
            n_test = 0
        test_idx.extend(idxs[:n_test].tolist())
        train_idx.extend(idxs[n_test:].tolist())
    if not test_idx:
        # ensure non-empty test
        move = train_idx[: max(1, int(round(test_frac * n)))]
        test_idx = list(move)
        train_idx = train_idx[len(move) :]
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sources into unified slip dataset")
    parser.add_argument("--slip-nn", type=Path, default=ROOT / "data" / "slip_nn")
    parser.add_argument("--policy", type=Path, default=ROOT / "data" / "slip_nn_policy")
    parser.add_argument("--policy2", type=Path, default=ROOT / "data" / "slip_nn_policy2")
    parser.add_argument("--heavy", type=Path, default=ROOT / "data" / "slip_nn_policy2_heavy")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sources",
        default="slip_nn,policy,policy2,heavy",
        help="Comma list of sources to include",
    )
    args = parser.parse_args()

    mapping = {
        "slip_nn": ("slip_nn", args.slip_nn),
        "policy": ("slip_nn_policy", args.policy),
        "policy2": ("slip_nn_policy2", args.policy2),
        "heavy": ("slip_nn_policy2_heavy", args.heavy),
    }
    want = [s.strip() for s in args.sources.split(",") if s.strip()]
    parts = []
    for key in want:
        if key not in mapping:
            print(f"Unknown source {key}", file=sys.stderr)
            sys.exit(2)
        name, path = mapping[key]
        print(f"Loading {name} from {path} ...", flush=True)
        parts.append(_load_source(path, name))

    data = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    n = int(data["X"].shape[0])
    # Stratify by source|base_case (strip _hitNNNN so policy2 hits are not singleton strata)
    strat = np.array(
        [
            f"{s}|{_base_case(c)}"
            for s, c in zip(data["source_dataset"].tolist(), data["case_name"].tolist())
        ],
        dtype=object,
    )
    train_idx, test_idx = _stratified_split(n, strat, test_frac=args.test_frac, seed=args.seed)

    def take(idxs: np.ndarray) -> dict:
        return {k: v[idxs] for k, v in data.items()}

    train, test = take(train_idx), take(test_idx)
    # Also make a val = half of test for early-stopping convenience
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(len(test_idx))
    n_val = max(1, len(perm) // 2)
    val = take(test_idx[perm[:n_val]])
    test_only = take(test_idx[perm[n_val:]])

    args.out.mkdir(parents=True, exist_ok=True)
    for split, pack in (("train", train), ("val", val), ("test", test_only)):
        d = args.out / split
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "windows.npz", **pack)
        print(f"wrote {split}: {pack['X'].shape[0]}", flush=True)

    norm = compute_norm_stats(train["X"])
    # Source counts
    src_counts = {
        str(s): int(np.sum(data["source_dataset"].astype(str) == s))
        for s in sorted(set(data["source_dataset"].astype(str)))
    }
    wrist_frac = float(data["has_wrist_teacher"].mean())
    slip_frac = float((data["y_event"] > 0).mean())

    write_manifest(
        args.out / "manifest.json",
        window_steps=WINDOW_STEPS,
        n_train=int(train["X"].shape[0]),
        n_val=int(val["X"].shape[0]),
        n_test=int(test_only["X"].shape[0]),
        norm_stats=norm,
        extra={
            "feature_dim": FEATURE_DIM,
            "dataset": "slip_nn_unified",
            "label_keys": list(UNIFIED_LABELS),
            "gcd_note": (
                "Schema GCD across slip_nn/policy/policy2/heavy; "
                "missing wrist→0, missing slip→0; sources unioned"
            ),
            "sources": src_counts,
            "wrist_teacher_frac": wrist_frac,
            "slip_positive_frac": slip_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "fairness": "same_windows_for_nn2_p1_p2_grip_p2_wrist",
        },
    )
    summary = {
        "train": int(train["X"].shape[0]),
        "val": int(val["X"].shape[0]),
        "test": int(test_only["X"].shape[0]),
        "total": n,
        "sources": src_counts,
        "wrist_teacher_frac": wrist_frac,
        "slip_positive_frac": slip_frac,
        "y_grip_train_mean": float(train["y_grip"].mean()),
        "y_grip_train_max": float(train["y_grip"].max()),
        "manifest": str(args.out / "manifest.json"),
    }
    (args.out / "export_summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "README.md").write_text(
        "# Unified slip dataset (schema GCD)\n\n"
        "Built from `slip_nn` + `slip_nn_policy` + `slip_nn_policy2` + "
        "`slip_nn_policy2_heavy` with a **shared label schema**.\n\n"
        "- Features: 26-D × 40 steps\n"
        "- Labels: `y_event`, `y_grip`/`y_policy`/`y_grip_p2`, `y_wr/y_wp/y_wy`\n"
        "- Missing wrist teachers → 0; missing slip → 0\n"
        "- Split: stratified by `source|case`\n\n"
        f"Summary: `{summary}`\n\n"
        "Retrain all heads on this data before comparing.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
