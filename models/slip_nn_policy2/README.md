# Slip NN-Policy-2 (P2-A grip + wrist)

- backbone: `/workspace/models/slip_nn_v2/slip_tcn_v1.pt`
- data: `/workspace/data/slip_nn_policy2`
- policy MLP → grip + wrist(3), width=64
- trainable params: 6916
- best val MAE joint (grip+wrist): 0.0897 @ epoch 74
- max_grip=0.25, max_wrist=0.25
- norm_source: `/workspace/models/slip_nn_v2/train_meta.json`
- Spec: [`docs/NN-Policy-2-动作空间规格.md`](../../docs/NN-Policy-2-动作空间规格.md)
- deploy wrist_scale=0.5 (full wrist hurts s045 closed-loop)
