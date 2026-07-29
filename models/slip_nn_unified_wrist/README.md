# Slip NN-Policy-2 (P2-A grip + wrist)

- backbone: `/workspace/models/slip_nn_unified_nn2/slip_tcn_v1.pt`
- data: `/workspace/data/slip_nn_unified`
- policy MLP → grip + wrist(3), width=64
- trainable params: 6916
- best val MAE joint (grip+wrist): 0.0970 @ epoch 78
- max_grip=0.25, max_wrist=0.25
- norm_source: `/workspace/models/slip_nn_unified_nn2/train_meta.json`
- Spec: [`docs/NN-Policy-2-动作空间规格.md`](../../docs/NN-Policy-2-动作空间规格.md)
