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

1. **Hook API only** (this PR slice): types + `replay_spider_task` params; default unchanged; `RecordingExtendHook` smoke.
2. **PPO adapter**: load zip → map obs → `ExtendStepDecision`; still gated by mode flag.
3. **Eval script**: same cases, BC vs PPO(`on_detect`) vs PPO(`always`).
4. **Run smoke** and compare numbers.

## How to check Step 1

```bash
python3 scripts/smoke_extend_hook_step1.py --friction 0.5
# expect physics_unchanged=true, overrides=0, queries=200
```
