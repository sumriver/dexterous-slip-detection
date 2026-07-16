#!/usr/bin/env python3
"""Train NN-1 slip TCN/GRU on NN-0 windows (default label: y_fused)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.slip_nn_features import FEATURE_DIM
from sim.slip_nn_model import build_slip_model, count_params
from sim.slip_nn_data import SlipWindowDataset, evaluate_loader, load_split

LABEL_CHOICES = ("y_fused", "y_scheme2", "y_scheme1")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NN-1 slip detector")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "slip_nn")
    parser.add_argument("--label", default="y_fused", choices=LABEL_CHOICES)
    parser.add_argument("--arch", default="tcn", choices=("tcn", "gru"))
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "slip_nn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build model/loaders and exit without training",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    manifest_path = args.data / "manifest.json"
    if not manifest_path.exists():
        print(f"Missing manifest: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())
    mean = np.asarray(manifest["norm"]["mean"], dtype=np.float32)
    std = np.asarray(manifest["norm"]["std"], dtype=np.float32)

    x_train, y_train = load_split(args.data, "train", args.label)
    x_val, y_val = load_split(args.data, "val", args.label)

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)

    train_ds = SlipWindowDataset(
        x_train, y_train, mean=mean, std=std, augment=not args.no_augment
    )
    val_ds = SlipWindowDataset(x_val, y_val, mean=mean, std=std, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = build_slip_model(args.arch, feature_dim=FEATURE_DIM).to(device)
    n_params = count_params(model)
    print(f"arch={args.arch} params={n_params} label={args.label} train={len(train_ds)} val={len(val_ds)}")
    print(f"pos_weight={float(pos_weight):.4f} (n_pos={n_pos:.0f} n_neg={n_neg:.0f})")

    if args.dry_run:
        xb, yb = next(iter(train_loader))
        logits = model(xb.to(device))
        print(f"dry-run batch X={tuple(xb.shape)} logits={tuple(logits.shape)}")
        sys.exit(0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    args.out.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_epoch = -1
    bad = 0
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_m = evaluate_loader(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} "
            f"val_f1={val_m['f1']:.4f} P={val_m['precision']:.3f} R={val_m['recall']:.3f}"
        )
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_epoch = epoch
            bad = 0
            ckpt = {
                "model_state": model.state_dict(),
                "arch": args.arch,
                "feature_dim": FEATURE_DIM,
                "label": args.label,
                "seed": args.seed,
                "best_val_f1": best_f1,
                "epoch": epoch,
            }
            torch.save(ckpt, args.out / "slip_tcn_v1.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch} (best={best_epoch} f1={best_f1:.4f})")
                break

    elapsed = time.time() - t0
    meta = {
        "arch": args.arch,
        "label": args.label,
        "seed": args.seed,
        "params": n_params,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "elapsed_s": elapsed,
        "pos_weight": float(pos_weight),
        "norm": {"mean": mean.tolist(), "std": std.tolist()},
        "manifest": str(manifest_path),
        "history": history,
        "data_counts": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
        },
    }
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out / "metrics.json").write_text(
        json.dumps({"best_val_f1": best_f1, "best_epoch": best_epoch, "label": args.label}, indent=2)
    )
    print(f"Wrote {args.out / 'slip_tcn_v1.pt'} best_val_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
