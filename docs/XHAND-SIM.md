# XHAND1 MuJoCo Simulation

## Current Scenario (Phase 1b)

1. **Bottle horizontal** on floor (cylinder axis || world X)
2. **Mid-grasp** — hand descends from above, fingers close on cylinder mid-section
3. **Lift 20 cm** — gated on contact count
4. **Rotate 90°** — hand + bottle swing to upright (FLIP phase)
5. **Hold upright** — DONE phase checks sustained contacts + low tilt

```bash
bash scripts/setup_models.sh
python scripts/run_xhand_grasp_sim.py
python scripts/run_xhand_grasp_sim.py --video   # MP4 in data/xhand_grasp/
```

## Asset Pipeline

URDF from [MaureenZOU/worldstring](https://github.com/MaureenZOU/worldstring) → `urdf2mjcf` → `scripts/build_xhand_mjcf.py` → `xhand_right_sim.xml`

See `models/xhand/README.md` for file list.

## Pass Criteria (physics-only)

| Step | Metric |
|------|--------|
| Mid grasp | HOLD phase sustained contacts |
| Lift 20 cm | `max_bottle_z > initial + 0.15 m` with lift contacts |
| Stand upright | `final_tilt < 20°` |
| Hold without slip | DONE phase contacts + upright tilt |

## Grasp Geometry (critical)

Horizontal bottle axis is **world +X**. Valid grasp:

- **Four fingers** parallel to bottle axis
- **Thumb** on +Y side, **fingers** on −Y side (tripod / opposition)
- **No top-down insertion** — approach from −Y with open hand

Validate before every sim run:

```bash
python scripts/preview_xhand_grasp_pose.py   # must print "Pose OK", saves pose_preview.png
```

## Status

- Lateral tripod pre-grasp: **validated** (~64 mm clearance, thumb vs fingers opposed)
- Mid-grasp (HOLD contacts): **PASS** in latest run
- Lift / stand / hold upright: still **FAIL** (contacts but insufficient support force)
