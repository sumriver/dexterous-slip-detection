#!/usr/bin/env python3
"""Train NN-Policy-2 grip+wrist head (detect backbone frozen).

Data: ``data/slip_nn_policy2`` from ``export_policy2_teacher.py``.
Init backbone from ``models/slip_nn_v2``. Writes ``models/slip_nn_policy2/``.
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
from sim.slip_nn_policy import DEFAULT_POLICY_WIDTH  # noqa: E402
from sim.slip_nn_policy2 import (  # noqa: E402
    DEFAULT_WRIST_MAX,
    SlipDetectAndPolicy2,
    load_policy2_backbone,
    policy2_param_count,
)

IDX_GRIP_EXTRA = 23


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Policy2WindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_grip: np.ndarray,
        y_wrist: np.ndarray,
        *,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.x = x.astype(np.float32)
        self.y_g = y_grip.astype(np.float32)
        self.y_w = y_wrist.astype(np.float32)
        self.grip_raw = self.x[:, -1, IDX_GRIP_EXTRA].astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        x = (self.x[idx] - self.mean) / self.std
        return (
            torch.from_numpy(x),
            torch.tensor(self.y_g[idx], dtype=torch.float32),
            torch.from_numpy(self.y_w[idx]),
            torch.tensor(self.grip_raw[idx], dtype=torch.float32),
        )


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = data_dir / split / "windows.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python3 scripts/export_policy2_teacher.py --max-hits 2000"
        )
    raw = np.load(path, allow_pickle=True)
    for k in ("y_grip_p2", "y_wr", "y_wp", "y_wy"):
        if k not in raw.files:
            raise KeyError(f"{k} missing in {path}")
    y_g = raw["y_grip_p2"].astype(np.float32)
    y_w = np.stack(
        [raw["y_wr"], raw["y_wp"], raw["y_wy"]], axis=-1
    ).astype(np.float32)
    return raw["X"].astype(np.float32), y_g, y_w


@torch.no_grad()
def eval_policy(model: SlipDetectAndPolicy2, loader: DataLoader, device: torch.device) -> dict:
    model.backbone.eval()
    model.policy.eval()
    g_err, w_err = [], []
    g_pred, g_tgt = [], []
    w_pred, w_tgt = [], []
    for xb, yg, yw, gb in loader:
        xb, yg, yw, gb = xb.to(device), yg.to(device), yw.to(device), gb.to(device)
        _, _, g, w = model.forward_policy(xb, gb)
        g_err.append(torch.abs(g - yg).cpu().numpy())
        w_err.append(torch.abs(w - yw).cpu().numpy())
        g_pred.append(g.cpu().numpy())
        g_tgt.append(yg.cpu().numpy())
        w_pred.append(w.cpu().numpy())
        w_tgt.append(yw.cpu().numpy())
    ge = np.concatenate(g_err)
    we = np.concatenate(w_err, axis=0)
    gp = np.concatenate(g_pred)
    gt = np.concatenate(g_tgt)
    wp = np.concatenate(w_pred, axis=0)
    wt = np.concatenate(w_tgt, axis=0)
    return {
        "mae_grip": float(ge.mean()),
        "mae_wrist": float(we.mean()),
        "mae_wr": float(we[:, 0].mean()),
        "mae_wp": float(we[:, 1].mean()),
        "mae_wy": float(we[:, 2].mean()),
        "mae_joint": float(ge.mean() + we.mean()),
        "pred_grip_mean": float(gp.mean()),
        "tgt_grip_mean": float(gt.mean()),
        "pred_wrist_abs_mean": float(np.abs(wp).mean()),
        "tgt_wrist_abs_mean": float(np.abs(wt).mean()),
        "n": float(len(ge)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NN-Policy-2 grip+wrist head")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "slip_nn_policy2")
    parser.add_argument("--backbone", type=Path, default=ROOT / "models" / "slip_nn_v2" / "slip_tcn_v1.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "slip_nn_policy2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda-sparse-g", type=float, default=0.02)
    parser.add_argument("--lambda-sparse-w", type=float, default=0.01)
    parser.add_argument("--wrist-weight", type=float, default=1.0, help="MSE weight on wrist vs grip")
    parser.add_argument("--max-grip", type=float, default=0.25)
    parser.add_argument("--max-wrist", type=float, default=DEFAULT_WRIST_MAX)
    parser.add_argument("--policy-width", type=int, default=DEFAULT_POLICY_WIDTH)
    parser.add_argument("--policy-dropout", type=float, default=0.0)
    parser.add_argument("--drop-leak-features", action="store_true", default=True)
    parser.add_argument("--no-drop-leak-features", action="store_true")
    parser.add_argument(
        "--recompute-norm",
        action="store_true",
        help="Recompute mean/std on policy2 train windows (breaks detect; debug only)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.no_drop_leak_features:
        args.drop_leak_features = False

    set_seed(args.seed)
    backbone_meta_path = args.backbone.parent / "train_meta.json"
    manifest = json.loads((args.data / "manifest.json").read_text())
    # Keep detect backbone on its training norm. Recomputing norm on open-loop
    # PASS hits shifts force stats so far that closed-loop p_slip never fires.
    if backbone_meta_path.exists() and not args.recompute_norm:
        bmeta = json.loads(backbone_meta_path.read_text())
        mean = np.asarray(bmeta["norm"]["mean"], dtype=np.float32)
        std = np.asarray(bmeta["norm"]["std"], dtype=np.float32)
        norm_source = str(backbone_meta_path)
    else:
        mean = np.asarray(manifest["norm"]["mean"], dtype=np.float32)
        std = np.asarray(manifest["norm"]["std"], dtype=np.float32)
        norm_source = "data_manifest_or_recompute"

    x_tr, yg_tr, yw_tr = load_split(args.data, "train")
    x_va, yg_va, yw_va = load_split(args.data, "val")

    if args.drop_leak_features:
        x_tr = x_tr.copy()
        x_va = x_va.copy()
        for idx in LEAK_FEATURE_INDICES_MULTITASK:
            x_tr[:, :, idx] = 0.0
            x_va[:, :, idx] = 0.0
        if args.recompute_norm:
            flat = x_tr.reshape(-1, x_tr.shape[-1])
            mean = flat.mean(axis=0).astype(np.float32)
            std = flat.std(axis=0).astype(np.float32)
            std = np.where(std < 1e-8, 1.0, std)
            norm_source = "recompute_on_policy2_train"
        else:
            # Zero leak channels in the frozen backbone norm as well.
            mean = mean.copy()
            std = std.copy()
            for idx in LEAK_FEATURE_INDICES_MULTITASK:
                mean[idx] = 0.0
                std[idx] = 1.0
            std = np.where(std < 1e-8, 1.0, std)

    train_loader = DataLoader(
        Policy2WindowDataset(x_tr, yg_tr, yw_tr, mean=mean, std=std),
        batch_size=args.batch,
        shuffle=True,
    )
    val_loader = DataLoader(
        Policy2WindowDataset(x_va, yg_va, yw_va, mean=mean, std=std),
        batch_size=args.batch,
        shuffle=False,
    )

    device = torch.device(args.device)
    model = SlipDetectAndPolicy2(
        max_grip=args.max_grip,
        max_wrist=args.max_wrist,
        policy_width=args.policy_width,
        policy_dropout=args.policy_dropout,
    ).to(device)
    if args.backbone.exists():
        load_policy2_backbone(model, args.backbone, map_location=str(device))
        print(f"Loaded backbone from {args.backbone}")
    else:
        print(f"WARNING: backbone missing ({args.backbone})", file=sys.stderr)

    model.freeze_detect()
    counts = policy2_param_count(model)
    print(
        f"params={counts} width={args.policy_width} "
        f"λg={args.lambda_sparse_g} λw={args.lambda_sparse_w}"
    )
    print(
        f"train={len(yg_tr)} val={len(yg_va)} "
        f"y_g_mean={float(yg_tr.mean()):.4f} |y_w|_mean={float(np.abs(yw_tr).mean()):.4f}"
    )

    if args.dry_run:
        xb, yg, yw, gb = next(iter(train_loader))
        p, gr, g, w = model.forward_policy(xb.to(device), gb.to(device))
        print(f"dry-run X={tuple(xb.shape)} g={tuple(g.shape)} w={tuple(w.shape)}")
        sys.exit(0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    mse = nn.MSELoss()

    args.out.mkdir(parents=True, exist_ok=True)
    best = 1e9
    best_epoch = -1
    bad = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.policy.train()
        model.backbone.eval()
        losses = []
        for xb, yg, yw, gb in train_loader:
            xb, yg, yw, gb = xb.to(device), yg.to(device), yw.to(device), gb.to(device)
            opt.zero_grad(set_to_none=True)
            _, _, g, w = model.forward_policy(xb, gb)
            loss = (
                mse(g, yg)
                + args.wrist_weight * mse(w, yw)
                + args.lambda_sparse_g * g.mean()
                + args.lambda_sparse_w * w.abs().mean()
            )
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_m = eval_policy(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} "
            f"val_mae_g={val_m['mae_grip']:.4f} val_mae_w={val_m['mae_wrist']:.4f} "
            f"joint={val_m['mae_joint']:.4f}"
        )
        if val_m["mae_joint"] < best:
            best = val_m["mae_joint"]
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": "detect_and_policy2",
                    "feature_dim": FEATURE_DIM,
                    "max_grip": args.max_grip,
                    "max_wrist": args.max_wrist,
                    "wrist_scale": 0.5,
                    "policy_width": args.policy_width,
                    "policy_dropout": args.policy_dropout,
                    "backbone_ckpt": str(args.backbone),
                    "drop_leak_features": args.drop_leak_features,
                    "best_val_mae_joint": best,
                    "epoch": epoch,
                },
                args.out / "slip_policy2_v1.pt",
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    meta = {
        "arch": "detect_and_policy2",
        "tier": "A",
        "seed": args.seed,
        "params": counts,
        "policy_width": args.policy_width,
        "max_grip": args.max_grip,
        "max_wrist": args.max_wrist,
        "best_epoch": best_epoch,
        "best_val_mae_joint": best,
        "elapsed_s": time.time() - t0,
        "lambda_sparse_g": args.lambda_sparse_g,
        "lambda_sparse_w": args.lambda_sparse_w,
        "wrist_weight": args.wrist_weight,
        "backbone_ckpt": str(args.backbone),
        "data": str(args.data),
        "norm": {"mean": mean.tolist(), "std": std.tolist()},
        "norm_source": norm_source,
        "recompute_norm": bool(args.recompute_norm),
        "drop_leak_features": args.drop_leak_features,
        "freeze_detect": True,
        "history": history,
        "note": "NN-Policy-2 P2-A: frozen detect + grip+wrist head on open-loop search teachers",
        "deploy_latch": True,
        "confirm_steps": 30,
        "default_threshold": 0.99,
        "soft_threshold": 0.7,
        "soft_grip_scale": 1.0,
        "policy_mode": "p2a",
        "wrist_scale": 0.5,
        "note_wrist_scale": "Deploy scales NN wrist by 0.5; full 1.0 hurts s045 (BC open-loop→closed-loop shift)",
    }
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_mae_joint": best,
                "best_epoch": best_epoch,
                "policy_trainable": counts["policy_trainable"],
            },
            indent=2,
        )
    )
    (args.out / "README.md").write_text(
        "# Slip NN-Policy-2 (P2-A grip + wrist)\n\n"
        f"- backbone: `{args.backbone}`\n"
        f"- data: `{args.data}`\n"
        f"- policy MLP → grip + wrist(3), width={args.policy_width}\n"
        f"- trainable params: {counts['policy_trainable']}\n"
        f"- best val MAE joint (grip+wrist): {best:.4f} @ epoch {best_epoch}\n"
        f"- max_grip={args.max_grip}, max_wrist={args.max_wrist}\n"
        f"- norm_source: `{norm_source}`\n"
        "- Spec: [`docs/NN-Policy-2-动作空间规格.md`](../../docs/NN-Policy-2-动作空间规格.md)\n"
    )
    print(f"Wrote {args.out / 'slip_policy2_v1.pt'} best_val_mae_joint={best:.4f}")


if __name__ == "__main__":
    main()
