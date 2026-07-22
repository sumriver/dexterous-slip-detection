#!/usr/bin/env python3
"""Train NN-Policy-1 grip policy head (detect backbone frozen).

Data: ``data/slip_nn_policy`` from ``export_min_grip_teacher.py``.
Init backbone from ``models/slip_nn_v2``. Writes ``models/slip_nn_policy1/``.
"""

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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.slip_nn_detector import LEAK_FEATURE_INDICES_MULTITASK  # noqa: E402
from sim.slip_nn_features import FEATURE_DIM  # noqa: E402
from sim.slip_nn_policy import (  # noqa: E402
    DEFAULT_POLICY_WIDTH,
    SlipDetectAndPolicy,
    load_backbone_from_multitask_ckpt,
    policy_param_count,
)

IDX_GRIP_EXTRA = 23


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PolicyWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_policy: np.ndarray,
        *,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.x = x.astype(np.float32)
        self.y = y_policy.astype(np.float32)
        # Raw grip channel before leak-zero (for policy conditioning).
        self.grip_raw = self.x[:, -1, IDX_GRIP_EXTRA].astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        x = (self.x[idx] - self.mean) / self.std
        return (
            torch.from_numpy(x),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.grip_raw[idx], dtype=torch.float32),
        )


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / split / "windows.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python3 scripts/export_min_grip_teacher.py"
        )
    raw = np.load(path, allow_pickle=True)
    if "y_policy" not in raw.files:
        raise KeyError(f"y_policy missing in {path}")
    return raw["X"].astype(np.float32), raw["y_policy"].astype(np.float32)


@torch.no_grad()
def eval_policy(model: SlipDetectAndPolicy, loader: DataLoader, device: torch.device) -> dict:
    model.backbone.eval()
    model.policy.eval()
    preds, targets = [], []
    for xb, yb, gb in loader:
        xb, yb, gb = xb.to(device), yb.to(device), gb.to(device)
        _, _, g = model.forward_policy(xb, gb)
        preds.append(g.cpu().numpy())
        targets.append(yb.cpu().numpy())
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    return {
        "mae": mae,
        "rmse": rmse,
        "pred_mean": float(p.mean()),
        "target_mean": float(t.mean()),
        "pred_max": float(p.max()),
        "n": float(len(t)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NN-Policy-1 policy head")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "slip_nn_policy")
    parser.add_argument("--backbone", type=Path, default=ROOT / "models" / "slip_nn_v2" / "slip_tcn_v1.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "slip_nn_policy1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda-sparse", type=float, default=0.05)
    parser.add_argument("--max-grip", type=float, default=0.25)
    parser.add_argument(
        "--policy-width",
        type=int,
        default=DEFAULT_POLICY_WIDTH,
        help="Policy MLP hidden width (tier A default 64; tiny ablation: 32)",
    )
    parser.add_argument("--policy-dropout", type=float, default=0.0)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--unfreeze-detect", action="store_true")
    parser.add_argument("--drop-leak-features", action="store_true", default=True)
    parser.add_argument("--no-drop-leak-features", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.no_drop_leak_features:
        args.drop_leak_features = False

    set_seed(args.seed)
    manifest = json.loads((args.data / "manifest.json").read_text())
    mean = np.asarray(manifest["norm"]["mean"], dtype=np.float32)
    std = np.asarray(manifest["norm"]["std"], dtype=np.float32)

    x_tr, y_tr = load_split(args.data, "train")
    x_va, y_va = load_split(args.data, "val")

    if args.drop_leak_features:
        x_tr = x_tr.copy()
        x_va = x_va.copy()
        for idx in LEAK_FEATURE_INDICES_MULTITASK:
            x_tr[:, :, idx] = 0.0
            x_va[:, :, idx] = 0.0
        flat = x_tr.reshape(-1, x_tr.shape[-1])
        mean = flat.mean(axis=0).astype(np.float32)
        std = flat.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std)

    train_loader = DataLoader(
        PolicyWindowDataset(x_tr, y_tr, mean=mean, std=std),
        batch_size=args.batch,
        shuffle=True,
    )
    val_loader = DataLoader(
        PolicyWindowDataset(x_va, y_va, mean=mean, std=std),
        batch_size=args.batch,
        shuffle=False,
    )

    device = torch.device(args.device)
    model = SlipDetectAndPolicy(
        max_grip=args.max_grip,
        residual=args.residual,
        policy_width=args.policy_width,
        policy_dropout=args.policy_dropout,
    ).to(device)
    if args.backbone.exists():
        load_backbone_from_multitask_ckpt(model, args.backbone, map_location=str(device))
        print(f"Loaded backbone from {args.backbone}")
    else:
        print(f"WARNING: backbone missing ({args.backbone}); training from scratch", file=sys.stderr)

    if not args.unfreeze_detect:
        model.freeze_detect()
    counts = policy_param_count(model)
    print(
        f"params={counts} width={args.policy_width} dropout={args.policy_dropout} "
        f"residual={args.residual} lambda_sparse={args.lambda_sparse}"
    )
    print(f"train={len(y_tr)} val={len(y_va)} y_policy_mean={float(y_tr.mean()):.4f}")

    if args.dry_run:
        xb, yb, gb = next(iter(train_loader))
        p, gref, gpol = model.forward_policy(xb.to(device), gb.to(device))
        print(f"dry-run X={tuple(xb.shape)} p={tuple(p.shape)} g={tuple(gpol.shape)}")
        sys.exit(0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    mse = nn.MSELoss()

    args.out.mkdir(parents=True, exist_ok=True)
    best_mae = 1e9
    best_epoch = -1
    bad = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.policy.train()
        if args.unfreeze_detect:
            model.backbone.train()
        else:
            model.backbone.eval()
        losses = []
        for xb, yb, gb in train_loader:
            xb, yb, gb = xb.to(device), yb.to(device), gb.to(device)
            opt.zero_grad(set_to_none=True)
            _, _, g = model.forward_policy(xb, gb)
            loss = mse(g, yb) + args.lambda_sparse * g.mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_m = eval_policy(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} "
            f"val_mae={val_m['mae']:.4f} pred_mean={val_m['pred_mean']:.4f}"
        )
        if val_m["mae"] < best_mae:
            best_mae = val_m["mae"]
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": "detect_and_policy",
                    "feature_dim": FEATURE_DIM,
                    "residual": args.residual,
                    "max_grip": args.max_grip,
                    "policy_width": args.policy_width,
                    "policy_dropout": args.policy_dropout,
                    "backbone_ckpt": str(args.backbone),
                    "drop_leak_features": args.drop_leak_features,
                    "best_val_mae": best_mae,
                    "epoch": epoch,
                },
                args.out / "slip_policy_v1.pt",
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    meta = {
        "arch": "detect_and_policy",
        "tier": "A",
        "residual": args.residual,
        "seed": args.seed,
        "params": counts,
        "policy_width": args.policy_width,
        "policy_dropout": args.policy_dropout,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "elapsed_s": time.time() - t0,
        "lambda_sparse": args.lambda_sparse,
        "max_grip": args.max_grip,
        "backbone_ckpt": str(args.backbone),
        "data": str(args.data),
        "norm": {"mean": mean.tolist(), "std": std.tolist()},
        "drop_leak_features": args.drop_leak_features,
        "freeze_detect": not args.unfreeze_detect,
        "history": history,
        "note": (
            "NN-Policy-1 tier A: frozen detect + 34→W→W→1 LayerNorm policy "
            "on y_policy teacher (W=policy_width)"
        ),
    }
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_mae": best_mae,
                "best_epoch": best_epoch,
                "policy_width": args.policy_width,
                "policy_trainable": counts["policy_trainable"],
            },
            indent=2,
        )
    )
    (args.out / "README.md").write_text(
        "# Slip NN-Policy-1 (tier A)\n\n"
        f"- backbone: `{args.backbone}`\n"
        f"- data: `{args.data}`\n"
        f"- policy MLP: `34 → {args.policy_width} → {args.policy_width} → 1` (+ LayerNorm)\n"
        f"- policy trainable params: {counts['policy_trainable']}\n"
        f"- best val MAE: {best_mae:.4f} (epoch {best_epoch})\n"
        f"- residual={args.residual}, λ_sparse={args.lambda_sparse}, "
        f"dropout={args.policy_dropout}\n"
        "- Spec: [`docs/NN-Policy-1-实现规格.md`](../../docs/NN-Policy-1-实现规格.md)\n"
        "- Tiny ablation: `--policy-width 32` (single hidden if rebuilding head).\n"
    )
    print(f"Wrote {args.out / 'slip_policy_v1.pt'} best_val_mae={best_mae:.4f}")


if __name__ == "__main__":
    main()
