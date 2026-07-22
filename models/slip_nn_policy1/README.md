# Slip NN-Policy-1 (tier A)

- backbone: `/workspace/models/slip_nn_v2/slip_tcn_v1.pt`
- data: `/workspace/data/slip_nn_policy`
- policy MLP: `34 → 64 → 64 → 1` (+ LayerNorm)
- policy trainable params: 6721
- best val MAE: 0.0261 (epoch 3)
- residual=False, λ_sparse=0.05, dropout=0.0
- Spec: [`docs/NN-Policy-1-实现规格.md`](../../docs/NN-Policy-1-实现规格.md)
- Tiny ablation: `--policy-width 32` (single hidden if rebuilding head).

## s045 fix
- Moved `friction_s045` from policy val → **train**; val now `friction_s055` + `mass_x4_friction_div2`.
- Retrain: s045 / ÷2 / baseline all PASS (`closedloop_s045_fix.json`); s040 still fails (both models).

