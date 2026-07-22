# Slip NN-Policy-2 (stage-0 search + hit export)

Open-loop **grip + wrist residual** teacher search (spec: `docs/NN-Policy-2-动作空间规格.md`).

## Bulk hits (current)

| item | value |
|------|-------|
| unique PASS hits | **2000** (`search/hits_pool.json`) |
| wrist≠0 | **100%** (~1400 unique wrist bins @ 0.02 rad) |
| cases | `friction_div2`, `friction_s045` |
| motions | base + `_v1/_v2/_v3` |
| exported windows | **2000** (1 last-window / hit) → train 1600 / val 400 |

μ×0.40 still **unsolvable** under P2-A bounds — not included.

## Reproduce

```bash
pytest -q tests/test_policy2_control.py
python3 scripts/generate_policy2_hits.py --target 2000
python3 scripts/export_policy2_teacher.py --max-hits 2000 --one-window-per-hit
```

## Artifacts

- `search/hits_pool.json` / `hits_pool_summary.json`
- `hits_catalog.json`, `train|val/windows.npz`
- Labels: `y_grip_p2`, `y_wr`, `y_wp`, `y_wy`
