# Policy-2 extend PPO (smoke)

Small RL experiment after BC plateau on hard ketchup cells.

## What it learns

- **Segment:** ketchup `extend` only (200 steps, mimic+lift)
- **Action:** Policy-2 `(grip, Δroll, Δpitch, Δyaw)` per step
- **Obs:** 26-D slip features + grip/wrist proprio + time/friction/mass
- **Not trained:** SPIDER trajectory, NN-2 detect backbone

## Train

```bash
pip install 'gymnasium>=0.29' 'stable-baselines3>=2.3' tensorboard
python3 scripts/train_policy2_extend_rl.py --timesteps 80000 --n-envs 2
```

## Eval smoke (80k)

| Case | PPO env pass | BC grip CL | BC wrist CL |
|------|--------------|------------|-------------|
| friction_div2 | 8/8 | PASS | FAIL |
| baseline | 8/8 | PASS | PASS |
| friction_s045 | 0/8 | FAIL* | FAIL |
| friction_s040 | 0/8 | FAIL | FAIL |

\*BC closed-loop on s045 can be flaky across runs; see unified suite for multi-cell ranking.

## Fairness caveat

PPO numbers are from the **extend-only Gym MDP** (action available every extend step from t=0).  
BC numbers are **full closed-loop** with frozen detect + policy head.

So PPO succeeding on `div2` mainly shows: *open-loop-solvable cells are learnable by RL*.  
It does **not** yet prove RL beats BC under the deployed NN detect trigger.

**s040** remains unsolved (same as open-loop teacher search).

## Next (if continuing RL)

1. Deploy PPO inside `replay_spider_task` extend loop for true closed-loop compare
2. Longer train + denser `s045`/`s040` curriculum
3. Optional: BC warm-start of the PPO actor from Policy-2 teachers
