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

## Unified suite v2 (clean ranking)

Ranked: **NN-2 / grip-only / grip+wrist** only (same `data/slip_nn_unified`).
- Second grip seed → variance, not ranked
- `friction_s040` → envelope, not ranked
- Wrist ablation table in `unified_suite_latest.json`

Hard cases: `hard_cases_status.json` (s040 still unsolvable open-loop).

```bash
python3 scripts/build_unified_slip_dataset.py --boost-cases friction_div2:3
python3 scripts/retrain_unified_models.py
```
