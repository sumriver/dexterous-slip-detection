# Slip NN-Policy-2 (P2-A grip + wrist)

- backbone: `/workspace/models/slip_nn_v2/slip_tcn_v1.pt`
- data: `data/slip_nn_policy2_heavy`
- policy MLP → grip + wrist(3), width=64
- trainable params: 6916
- best val MAE joint (grip+wrist): 0.0943 @ epoch 78
- max_grip=0.15, max_wrist=0.25
- norm_source: `/workspace/models/slip_nn_v2/train_meta.json`
- Spec: [`docs/NN-Policy-2-动作空间规格.md`](../../docs/NN-Policy-2-动作空间规格.md)
