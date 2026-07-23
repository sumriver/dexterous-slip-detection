# Policy-2 heavy + grip-cap teacher data

Open-loop P2-A teachers under **heavier objects** and a **hard grip ceiling**.

## Envelope

| knob | value |
|------|-------|
| mass_scale | ×2, ×4 |
| friction_scale | 1.0, 0.5 (÷2) |
| **g_max** | **0.15** (was 0.25) |
| d_max | 0.25 rad |
| hits | 1500 unique PASS, wrist≠0 |

## Cases

- `mass_x2`
- `mass_x2_friction_div2`
- `mass_x4`
- `mass_x4_friction_div2`

## Reproduce

```bash
# Stage-0 solvability
python3 scripts/search_policy2_teacher.py \
  --mass-scales 2,4 --friction-scales 1.0,0.5 \
  --g-max 0.15 --out-dir data/slip_nn_policy2_heavy/search

# Bulk hits
python3 scripts/generate_policy2_hits.py \
  --mass-scales 2,4 --friction-scales 1.0,0.5 \
  --g-max 0.15 --target 1500 \
  --out-dir data/slip_nn_policy2_heavy/search

# Export windows
python3 scripts/export_policy2_teacher.py \
  --search-dir data/slip_nn_policy2_heavy/search \
  --out data/slip_nn_policy2_heavy \
  --max-hits 1500 --g-max 0.15
```

## Notes

- Grip labels are strictly ≤ `g_max` (controller + export clip).
- Does **not** overwrite `data/slip_nn_policy2/` (friction-only pool).
- Train next: `python3 scripts/train_slip_policy2.py --data data/slip_nn_policy2_heavy --max-grip 0.15 --out models/slip_nn_policy2_heavy`
