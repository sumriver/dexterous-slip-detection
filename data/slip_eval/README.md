# Discriminative closed-loop test suite

## Problem

Previous gates were **all-PASS** (baseline / ÷2 / mass×4@g≤0.15) or **all-FAIL** (s040).
Binary pass/fail could not rank NN-2 vs Policy-1 vs Policy-2.

## Design

Case grid: `data/slip_eval/discriminative_case_grid.json`

| Tier | Role |
|------|------|
| **A_frontier** | Cells where models **split** (primary ranking) |
| **B_economy** | Easy PASSes — rank by **grip peak** + lift margin |
| **C_envelope** | Physics floor — expect **all FAIL** (s040 / s042) |

### Ranking

1. Maximize **frontier pass rate** (Tier A)
2. Higher mean **lift margin** on Tier-A passes
3. Lower mean **grip peak** on Tier-B passes
4. Fewer baseline slip events

### Run

```bash
python3 scripts/eval_slip_discriminative_suite.py
python3 scripts/eval_slip_discriminative_suite.py --models nn2,p1,p2
```

Artifacts:
- `data/slip_eval/discriminative_suite_latest.json`
- `data/slip_eval/discriminative_rankings.json`

## Frontier cells (why they discriminate)

| cell | grip_cap | Expected split |
|------|----------|----------------|
| mass×2 × μ0.45 | 0.25 | NN-2 PASS; policies often FAIL |
| mass×4 × μ0.45 | 0.25 | Near gate / partial |
| mass×2 × μ0.42 | 0.25 | P2H OOD FAIL |
| μ0.45 | 0.25 | P2H FAIL (heavy-only train) |
| mass×4 ÷2 | 0.12 | Tight grip; P2 FAIL |
| mass×2 ÷2 | 0.10 | P2H FAIL |
| mass×8 × μ0.45 | 0.15 | P1/P2 PASS; NN-2 FAIL |
