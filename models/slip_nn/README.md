# Slip NN-1 models

**Status**: scaffolding ready — **weights not trained yet**.

| Item | Value |
|------|--------|
| Spec | [`docs/NN-1-实现规格.md`](../../docs/NN-1-实现规格.md) |
| Default teacher | `y_fused` |
| Arch | TCN (`slip_tcn_v1.pt` after train) |
| Seed | 42 |
| Threshold τ | 0.5 (tune on val F1) |

## Train (next step)

```bash
# Need NN-0 NPZ:
python3 scripts/export_slip_dataset.py

python3 scripts/train_slip_tcn.py --label y_fused --arch tcn --out models/slip_nn
python3 scripts/eval_slip_nn_offline.py --split val
python3 scripts/eval_slip_nn_closedloop.py   # baseline + friction_div2
```

Ablation:

```bash
python3 scripts/train_slip_tcn.py --label y_scheme2 --out models/slip_nn/ablate_s2
```

## Dry-run (no training)

```bash
python3 scripts/train_slip_tcn.py --dry-run
```

## Closed-loop flags

```bash
python3 scripts/run_ketchup_robustness_sweep.py --case friction_div2 --antislip-nn
python3 scripts/run_ketchup_robustness_sweep.py --case baseline --antislip-nn
```
