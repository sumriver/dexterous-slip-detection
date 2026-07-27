# Discriminative closed-loop test suite (v2 — same-train only)

## Fairness rule (hard)

**只在同一训练集训出来的模型之间排名。**  
跨数据域（例如旧 Policy-1 vs P2-heavy）只能当 transfer 诊断，**禁止混排**。

| League | Train data | Fair models |
|--------|------------|-------------|
| `friction_p2a` | `data/slip_nn_policy2` | `p2` (grip+wrist) vs `p2_grip_only` |
| `heavy_gripcap` | `data/slip_nn_policy2_heavy` | `p2h` vs `p2h_grip_only` |

NN-2 / 旧 Policy-1 训练集不同 → `--include-reference` 才跑，且不进 ranking。

## Train same-data controls

```bash
# Friction domain: grip-only ablation on P2 windows
python3 scripts/train_slip_policy.py \
  --data data/slip_nn_policy2 --max-grip 0.25 \
  --out models/slip_nn_policy2_grip_only

# Heavy domain: grip-only ablation on heavy windows
python3 scripts/train_slip_policy.py \
  --data data/slip_nn_policy2_heavy --max-grip 0.15 \
  --out models/slip_nn_policy2_heavy_grip_only
```

## Eval

```bash
python3 scripts/eval_slip_discriminative_suite.py --domain all_fair
python3 scripts/eval_slip_discriminative_suite.py --domain friction_p2a
python3 scripts/eval_slip_discriminative_suite.py --domain heavy_gripcap
```

Artifacts: `discriminative_suite_latest.json`, `discriminative_rankings.json`.

## Unified same-data suite (`unified_suite_latest.json`)

All four models trained on `data/slip_nn_unified` (schema-GCD union of slip_nn / policy / policy2 / heavy).

Latest ranking (frontier PASS @ ≥6 cm lift):

| Rank | Model | Frontier | Economy mean grip |
|------|-------|----------|-------------------|
| 1 | P2 grip-only | 5/7 | 0.243 |
| 2 | P2 grip+wrist | 5/7 | 0.248 |
| 3 | P1 grip-only | 5/7 | 0.227 |
| 4 | NN-2 | 1/7 | 0.189 |

Rebuild: `python3 scripts/build_unified_slip_dataset.py && python3 scripts/retrain_unified_models.py`
