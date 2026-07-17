# Slip NN-1 models

| Item | Value |
|------|--------|
| Spec | [`docs/NN-1-实现规格.md`](../../docs/NN-1-实现规格.md) |
| Checkpoint | `slip_tcn_v1.pt` (TCN, 19521 params) |
| **Teacher (default)** | **`y_event`** — future 1 cm drop within 0.5 s |
| Deploy | `drop_leak_features` (zero s2/phase/μ), `deploy_latch`, `confirm_steps=15`, **τ=0.7** |
| Seed | 42 |

## Why not `y_fused` / `y_scheme2`?

Rule scheme-2 itself fires **~191/200** steps on baseline extend (latched). Distilling `y_fused`/`y_scheme2` yields `p_slip≈1` even at τ=0.99 — τ cannot fix teacher over-trigger. Ablations in `ablate_s2/` and `data/slip_nn/tau_sweep_*.json`.

## Closed-loop gates (current default)

| Gate | Result |
|------|--------|
| friction÷2 Δz / contacts | **+8.7 cm, 200/200 PASS** |
| baseline false-trigger (`nn_slip_events` raw fires) | **93/200 PASS** (&lt;100) |
| baseline lift | **+9.4 cm PASS** |

Offline val F1 @ τ=0.7 ≈ 0.76 (event label; closed-loop gates prioritized).

## Reproduce

```bash
python3 scripts/export_slip_dataset.py   # writes y_event
python3 scripts/train_slip_tcn.py --label y_event --drop-leak-features --deploy-latch --confirm-steps 15
python3 scripts/eval_slip_nn_closedloop.py   # reads default_threshold from train_meta
```

Legacy:

```bash
python3 scripts/train_slip_tcn.py --label y_fused --out models/slip_nn/ablate_fused
python3 scripts/train_slip_tcn.py --label y_scheme2 --out models/slip_nn/ablate_s2
```
