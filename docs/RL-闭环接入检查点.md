# Closed-loop PPO wiring — gated steps

Goal: put PPO into `replay_spider_task` for a fair closed-loop compare with BC,
**but each step is reviewable and default-off**.

## Modes

| mode | when hook is asked | apply action? |
|------|--------------------|---------------|
| `off` | never | — |
| `always` | every extend step | if hook returns action |
| `on_detect` | soft/hard slip only | if hook returns action |
| `inspect` | every extend step | only if hook returns action (RecordingHook returns none) |

## Steps (stop after each for review)

1. **Hook API only** ✅ — types + `replay_spider_task` params; `RecordingExtendHook` smoke.
2. **PPO adapter** ✅ — `PPOExtendHook` loads zip → `ExtendStepDecision`; `apply_actions` flag.
3. **Eval script**: same cases, BC vs PPO(`on_detect`) vs PPO(`always`).
4. **Run smoke** and compare numbers.

## How to check Step 1

```bash
python3 scripts/smoke_extend_hook_step1.py --friction 0.5
# expect physics_unchanged=true, overrides=0, queries=200
```

## How to check Step 2

```bash
python3 scripts/smoke_extend_hook_step2.py --friction 0.5
# expect:
#   A always+propose-only: overrides=0, queries=200
#   B always+apply:        overrides=200 (div2 lift should recover)
#   C on_detect without NN: queries=0 (needs detect — Step3)
```

## How to check Step 3

```bash
python3 scripts/eval_ppo_closedloop_vs_bc.py
# matrix: bc_grip | bc_wrist | ppo_always | ppo_on_detect
# artifact: data/slip_eval/ppo_closedloop_vs_bc.json
```
