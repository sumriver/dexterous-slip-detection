# Slip NN-Policy-2 (stage-0 search + hit export)

Open-loop **grip + wrist residual** teacher search (spec: `docs/NN-Policy-2-动作空间规格.md`).

## Search result (this tree)

| case | μ× | solvable | hits (trials) | notes |
|------|-----|----------|---------------|-------|
| friction_div2 | 0.50 | **yes** | 71 / 90 | many open-loop PASS |
| friction_s045 | 0.45 | **yes** | 64 / 90 | wrist helps lift |
| friction_s040 | 0.40 | **no** | 0 / 218 | also failed at g≤0.35, \|Δw\|≤0.5 |

→ μ×0.40 still **outside** P2-A open-loop envelope under tried bounds; do **not** train a net to “fix” it yet.

## Reproduce

```bash
pytest -q tests/test_policy2_control.py
python3 scripts/search_policy2_teacher.py --method both --expand
python3 scripts/export_policy2_teacher.py --max-per-case 8
```

## Artifacts

- `search/search_summary.json` — solvability summary  
- `search/friction_*.json` — per-case hits/best  
- `hits_catalog.json` / `train|val/windows.npz` — BC-ready windows from ÷2 & s045 hits  
- Labels: `y_grip_p2`, `y_wr`, `y_wp`, `y_wy` (+ `y_policy`=grip)
