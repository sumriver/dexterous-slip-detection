# Slip NN-2 (multitask)

| Item | Value |
|------|--------|
| Spec | [`docs/NN-2-实现规格.md`](../../docs/NN-2-实现规格.md) |
| Arch | `tcn_multi` (19554 params) |
| Slip label | `y_event` |
| Grip teacher | `y_grip` (antislip export) |
| λ_grip | 0.2 |
| Deploy | soft=0.7, τ=0.99, confirm=30, latch, drop leak(+grip) |

## Closed-loop gates

| Gate | Result |
|------|--------|
| baseline nn_slip | **43/200** (&lt;50) |
| friction÷2 | **+8.7 cm**, 200/200 |

```bash
python3 scripts/export_slip_dataset.py --antislip
python3 scripts/train_slip_multitask.py --lambda-grip 0.2
python3 scripts/eval_slip_nn_closedloop.py --nn-model-dir models/slip_nn_v2
```
