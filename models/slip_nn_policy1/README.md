# Slip NN-Policy-1 (tier A)

- backbone: `/workspace/models/slip_nn_v2/slip_tcn_v1.pt`
- data: `/workspace/data/slip_nn_policy`
- policy MLP: `34 → 64 → 64 → 1` (+ LayerNorm)
- policy trainable params: 6721
- best val MAE: 0.0347 (epoch 12)
- residual=False, λ_sparse=0.05, dropout=0.0
- Spec: [`docs/NN-Policy-1-实现规格.md`](../../docs/NN-Policy-1-实现规格.md)
- Tiny ablation: `--policy-width 32` (single hidden if rebuilding head).

## Closed-loop vs NN-2 (`replace`, dose-grid retrain)

| suite | baseline events | baseline max_grip | ÷2 Δz | ÷2 gate |
|---|---|---|---|---|
| NN-2 | 43 | 0.241 | +8.7 cm | PASS |
| Policy-1 | 20 | 0.128 | +8.7 cm | PASS |

- Data: μ-neighborhood + mass×μ crosses (`build_policy_cases`); train `y≥0.13` ≈10% (was ~5%).
- Compare: events Δ=-23, grip Δ=-0.113, ÷2 dz Δ=-0.0 cm.
- Artifacts: `closedloop_policy1.json`

