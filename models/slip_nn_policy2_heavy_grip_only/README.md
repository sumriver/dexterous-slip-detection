# Slip NN-Policy-1 (tier A)

- backbone: `/workspace/models/slip_nn_v2/slip_tcn_v1.pt`
- data: `data/slip_nn_policy2_heavy`
- policy MLP: `34 → 64 → 64 → 1` (+ LayerNorm)
- policy trainable params: 6721
- best val MAE: 0.0189 (epoch 38)
- residual=False, λ_sparse=0.05, dropout=0.0
- Spec: [`docs/NN-Policy-1-实现规格.md`](../../docs/NN-Policy-1-实现规格.md)
- Tiny ablation: `--policy-width 32` (single hidden if rebuilding head).
