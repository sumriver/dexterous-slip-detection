# Slip NN-1 models

| Item | Value |
|------|--------|
| Spec | [`docs/NN-1-实现规格.md`](../../docs/NN-1-实现规格.md) |
| Checkpoint | `slip_tcn_v1.pt` (80KB, TCN, 19521 params) |
| Teacher | **`y_fused`** |
| Seed | 42 |
| Threshold τ | 0.5 |
| Best epoch | 4 (early stop @ 12) |

## Offline metrics (seed 42)

| Split | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| val | 0.964 | 0.992 | **0.978** (≥0.90 gate **PASS**) |
| test | 0.964 | 0.985 | **0.974** |

## Closed-loop smoke (`eval_slip_nn_closedloop.py`)

| Case | status | extend Δz | contacts | nn_slip_events |
|------|--------|-----------|----------|----------------|
| baseline | pass | +7.1 cm | 200/200 | 200 |
| friction_div2 | pass | **+8.7 cm** | **200/200** | 200 |

- friction÷2 lift gate (Δz≥6 cm, contacts≥200): **PASS**
- baseline false-trigger gate (nn_slip_events &lt; 100): **FAIL** — `y_fused` inherits scheme-1 sensitivity; next: raise τ / ablate `y_scheme2`

## Reproduce

```bash
python3 scripts/export_slip_dataset.py   # if NPZ missing
python3 scripts/train_slip_tcn.py --label y_fused --arch tcn --out models/slip_nn
python3 scripts/eval_slip_nn_offline.py --split val
python3 scripts/eval_slip_nn_closedloop.py
```

Ablation:

```bash
python3 scripts/train_slip_tcn.py --label y_scheme2 --out models/slip_nn/ablate_s2
```
