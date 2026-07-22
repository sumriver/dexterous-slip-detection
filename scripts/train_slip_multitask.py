#!/usr/bin/env python3
"""Train NN-2 multitask TCN (slip + Δgrip). Writes models/slip_nn_v2/.

Grip target: if NPZ y_grip is all-zero (open-loop export), synthesize
y_grip_syn = max_grip * y_event (pipeline smoke). Prefer re-export with
--antislip for real grip teachers (see docs/NN-2-实现规格.md).
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

from sim.slip_nn_data import classification_metrics, load_split  # noqa: E402
from sim.slip_nn_detector import LEAK_FEATURE_INDICES_MULTITASK  # noqa: E402
from sim.slip_nn_features import FEATURE_DIM  # noqa: E402
from sim.slip_nn_model import SlipTCNMulti, count_params  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MultiTaskWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_slip: np.ndarray,
        y_grip: np.ndarray,
        *,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.x = x.astype(np.float32)
        self.y_slip = y_slip.astype(np.float32)
        self.y_grip = y_grip.astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        x = (self.x[idx] - self.mean) / self.std
        return (
            torch.from_numpy(x),
            torch.tensor(self.y_slip[idx], dtype=torch.float32),
            torch.tensor(self.y_grip[idx], dtype=torch.float32),
        )


def load_grip(data_dir: Path, split: str, y_event: np.ndarray, max_grip: float) -> tuple[np.ndarray, str]:
    path = data_dir / split / "windows.npz"
    raw = np.load(path, allow_pickle=True)
    if "y_grip" in raw.files:
        g = raw["y_grip"].astype(np.float32)
        if float(np.abs(g).max()) > 1e-6:
            return g, "y_grip"
    syn = (y_event.astype(np.float32) * max_grip).astype(np.float32)
    return syn, "y_grip_syn=max_grip*y_event"


@torch.no_grad()
def eval_multi(model, loader, device, thr: float) -> dict:
    model.eval()
    probs, labels, grips_hat, grips_true = [], [], [], []
    for xb, ys, yg in loader:
        xb = xb.to(device)
        slip_logit, grip = model.forward_multi(xb)
        probs.append(torch.sigmoid(slip_logit).cpu().numpy())
        labels.append(ys.numpy())
        grips_hat.append(grip.cpu().numpy())
        grips_true.append(yg.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    g_hat = np.concatenate(grips_hat)
    g_true = np.concatenate(grips_true)
    out = classification_metrics(y_true, y_prob, thr)
    out["grip_mae"] = float(np.mean(np.abs(g_hat - g_true)))
    out["n"] = float(len(y_true))
    out["pos_rate"] = float(y_true.mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NN-2 multitask slip+grip")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "slip_nn")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "slip_nn_v2")
    parser.add_argument("--label", default="y_event")
    parser.add_argument("--lambda-grip", type=float, default=0.2)
    parser.add_argument("--max-grip", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pos-weight", type=float, default=None, help="Override BCE pos_weight (default: Nneg/Npos)")
    parser.add_argument("--max-mass-scale", type=float, default=None, help="Keep windows with mass_scale <= this")
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="If >0, use focal modulation on BCE")
    parser.add_argument("--drop-leak-features", action="store_true", default=True)
    parser.add_argument("--no-drop-leak-features", action="store_true")
    parser.add_argument("--deploy-latch", action="store_true", default=True)
    parser.add_argument("--confirm-steps", type=int, default=15)
    parser.add_argument("--default-threshold", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.no_drop_leak_features:
        args.drop_leak_features = False

    set_seed(args.seed)
    manifest = json.loads((args.data / "manifest.json").read_text())
    mean = np.asarray(manifest["norm"]["mean"], dtype=np.float32)
    std = np.asarray(manifest["norm"]["std"], dtype=np.float32)

    x_tr, y_tr = load_split(args.data, "train", args.label)
    x_va, y_va = load_split(args.data, "val", args.label)
    g_tr, grip_src = load_grip(args.data, "train", y_tr, args.max_grip)
    g_va, _ = load_grip(args.data, "val", y_va, args.max_grip)

    if args.max_mass_scale is not None:
        raw_tr = np.load(args.data / "train" / "windows.npz", allow_pickle=True)
        raw_va = np.load(args.data / "val" / "windows.npz", allow_pickle=True)
        m_tr = raw_tr["mass_scale"] <= args.max_mass_scale
        m_va = raw_va["mass_scale"] <= args.max_mass_scale
        x_tr, y_tr, g_tr = x_tr[m_tr], y_tr[m_tr], g_tr[m_tr]
        x_va, y_va, g_va = x_va[m_va], y_va[m_va], g_va[m_va]
        print(f"filtered max_mass_scale={args.max_mass_scale}: train={len(y_tr)} val={len(y_va)}")

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

    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr) - n_pos)
    pw = args.pos_weight if args.pos_weight is not None else (n_neg / max(n_pos, 1.0))
    pos_weight = torch.tensor([pw], dtype=torch.float32)

    train_loader = DataLoader(
        MultiTaskWindowDataset(x_tr, y_tr, g_tr, mean=mean, std=std),
        batch_size=args.batch,
        shuffle=True,
    )
    val_loader = DataLoader(
        MultiTaskWindowDataset(x_va, y_va, g_va, mean=mean, std=std),
        batch_size=args.batch,
        shuffle=False,
    )

    device = torch.device(args.device)
    model = SlipTCNMulti(feature_dim=FEATURE_DIM, max_grip=args.max_grip).to(device)
    n_params = count_params(model)
    print(
        f"arch=tcn_multi params={n_params} label={args.label} grip_src={grip_src} "
        f"lambda={args.lambda_grip} train={len(y_tr)} val={len(y_va)}"
    )

    if args.dry_run:
        xb, ys, yg = next(iter(train_loader))
        slip, grip = model.forward_multi(xb.to(device))
        print(f"dry-run X={tuple(xb.shape)} slip={tuple(slip.shape)} grip={tuple(grip.shape)}")
        sys.exit(0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce_none = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")
    mse = nn.MSELoss()

    args.out.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_epoch = -1
    bad = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, ys, yg in train_loader:
            xb, ys, yg = xb.to(device), ys.to(device), yg.to(device)
            opt.zero_grad(set_to_none=True)
            slip_logit, grip = model.forward_multi(xb)
            bce_el = bce_none(slip_logit, ys)
            if args.focal_gamma > 0:
                p = torch.sigmoid(slip_logit.detach())
                pt = torch.where(ys > 0.5, p, 1 - p)
                bce_el = ((1 - pt) ** args.focal_gamma) * bce_el
            loss = bce_el.mean() + args.lambda_grip * mse(grip, yg)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_m = eval_multi(model, val_loader, device, thr=args.default_threshold)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} "
            f"val_f1={val_m['f1']:.4f} grip_mae={val_m['grip_mae']:.4f}"
        )
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": "tcn_multi",
                    "feature_dim": FEATURE_DIM,
                    "label": args.label,
                    "lambda_grip": args.lambda_grip,
                    "grip_source": grip_src,
                    "drop_leak_features": args.drop_leak_features,
                    "deploy_latch": args.deploy_latch,
                    "confirm_steps": args.confirm_steps,
                    "use_grip_head": True,
                    "best_val_f1": best_f1,
                    "epoch": epoch,
                },
                args.out / "slip_tcn_v1.pt",
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    meta = {
        "arch": "tcn_multi",
        "label": args.label,
        "grip_source": grip_src,
        "lambda_grip": args.lambda_grip,
        "seed": args.seed,
        "params": n_params,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "elapsed_s": time.time() - t0,
        "norm": {"mean": mean.tolist(), "std": std.tolist()},
        "drop_leak_features": args.drop_leak_features,
        "deploy_latch": args.deploy_latch,
        "confirm_steps": args.confirm_steps,
        "use_grip_head": True,
        "default_threshold": args.default_threshold,
        "pos_weight": float(pos_weight),
        "max_mass_scale": args.max_mass_scale,
        "focal_gamma": args.focal_gamma,
        "history": history,
        "note": "NN-2 multitask with real/synth y_grip; deploy applies set_grip(delta_grip) when slip_active",
    }
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out / "metrics.json").write_text(
        json.dumps({"best_val_f1": best_f1, "best_epoch": best_epoch, "arch": "tcn_multi"}, indent=2)
    )
    (args.out / "README.md").write_text(
        "# Slip NN-2 (multitask)\n\n"
        f"- arch: tcn_multi ({n_params} params)\n"
        f"- slip label: {args.label}\n"
        f"- grip: {grip_src}, λ={args.lambda_grip}\n"
        f"- deploy: confirm={args.confirm_steps}, τ={args.default_threshold}, use_grip_head=True\n"
        "- Spec: [`docs/NN-2-实现规格.md`](../../docs/NN-2-实现规格.md)\n"
    )
    print(f"Wrote {args.out / 'slip_tcn_v1.pt'} best_val_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
