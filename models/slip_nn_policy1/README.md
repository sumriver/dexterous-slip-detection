# Slip NN-Policy-1 (tier A)

- backbone: `/workspace/models/slip_nn_v2/slip_tcn_v1.pt`
- data: `/workspace/data/slip_nn_policy`
- policy MLP: `34 → 64 → 64 → 1` (+ LayerNorm)
- policy trainable params: 6721
- best val MAE: 0.0065 (epoch 8)
- residual=False, λ_sparse=0.05, dropout=0.0
- Spec: [`docs/NN-Policy-1-实现规格.md`](../../docs/NN-Policy-1-实现规格.md)
- Tiny ablation: `--policy-width 32` (single hidden if rebuilding head).

## Closed-loop vs NN-2 (`replace`)

| suite | baseline events | baseline max_grip | ÷2 Δz | ÷2 gate |
|---|---|---|---|---|
| NN-2 | 43 | 0.241 | +8.7 cm | PASS |
| Policy-1 | 46 | 0.071 | -19.5 cm (drop) | FAIL |

See `closedloop_policy1.json`. Policy under-predicts high doses (peak grip≈0.07 < ÷2 need≈0.13).

