#!/usr/bin/env python3
"""Offline eval of NN-1 checkpoint on val/test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.slip_nn_data import SlipWindowDataset, evaluate_loader, load_split
from sim.slip_nn_features import FEATURE_DIM
from sim.slip_nn_model import build_slip_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline F1 for slip NN")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "slip_nn")
    parser.add_argument("--ckpt", type=Path, default=ROOT / "models" / "slip_nn" / "slip_tcn_v1.pt")
    parser.add_argument("--meta", type=Path, default=None, help="train_meta.json (optional)")
    parser.add_argument("--label", default=None, help="Override label key")
    parser.add_argument("--split", default="val", choices=("val", "test", "train"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.ckpt.exists():
        print(
            f"Missing checkpoint: {args.ckpt}\n"
            "Train first: python3 scripts/train_slip_tcn.py --label y_fused",
            file=sys.stderr,
        )
        sys.exit(2)

    meta_path = args.meta or args.ckpt.parent / "train_meta.json"
    manifest = json.loads((args.data / "manifest.json").read_text())
    mean = np.asarray(manifest["norm"]["mean"], dtype=np.float32)
    std = np.asarray(manifest["norm"]["std"], dtype=np.float32)
    arch = "tcn"
    label = args.label or "y_fused"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        arch = meta.get("arch", arch)
        label = args.label or meta.get("label", label)
        if "norm" in meta:
            mean = np.asarray(meta["norm"]["mean"], dtype=np.float32)
            std = np.asarray(meta["norm"]["std"], dtype=np.float32)

    x, y = load_split(args.data, args.split, label)
    ds = SlipWindowDataset(x, y, mean=mean, std=std, augment=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        arch = ckpt.get("arch", arch)
        feature_dim = int(ckpt.get("feature_dim", FEATURE_DIM))
    else:
        state = ckpt
        feature_dim = FEATURE_DIM

    model = build_slip_model(arch, feature_dim=feature_dim).to(device)
    model.load_state_dict(state)
    metrics = evaluate_loader(model, loader, device, thr=args.threshold)
    metrics.update({"split": args.split, "label": label, "arch": arch, "ckpt": str(args.ckpt)})

    print(json.dumps(metrics, indent=2))
    out = args.out or (args.ckpt.parent / f"eval_{args.split}.json")
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")
    if args.split == "val" and metrics["f1"] < 0.90:
        print(f"WARNING: val F1={metrics['f1']:.4f} < 0.90 gate", file=sys.stderr)


if __name__ == "__main__":
    main()
