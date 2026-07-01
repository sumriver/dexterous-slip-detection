# XHAND1 MuJoCo Simulation

## Asset Pipeline

```
worldstring URDF  →  urdf2mjcf  →  build_xhand_mjcf.py  →  xhand_right_sim.xml
     (setup)            (auto)          (actuators/keyframes)         (scene include)
```

**URDF source**: [MaureenZOU/worldstring](https://github.com/MaureenZOU/worldstring) `assets/xhand_right/`  
(same model as [RoboVerseOrg/roboverse_data](https://huggingface.co/datasets/RoboVerseOrg/roboverse_data) `robots/xhand_right/`)

```bash
bash scripts/setup_models.sh          # fetch URDF + meshes, build MJCF
python scripts/run_xhand_grasp_sim.py   # run Phase-1 bottle scene
```

## Components

| File | Role |
|------|------|
| `models/xhand/xhand_right_sim.xml` | 12-DoF hand + `hand_free` floating base |
| `models/scenes/xhand_grasp_scene.xml` | Desk + upright bottle |
| `src/sim/xhand_grasp_controller.py` | Planned arm pose + finger position actuators |
| `src/sim/xhand_tactile_sim.py` | Tier-A contact → 12×10 taxel grid per finger |
| `scripts/run_xhand_grasp_sim.py` | Physics metrics + PNG keyframes |

## Tactile Simulation (Tier A)

`XHandTactileSimulator` maps `mj_contact` on **bottle ∩ fingertip** to a 5×12×10 normal-force grid (plus tangential components in pad frame). This is sufficient for energy-flow inputs (Fᵢ, μᵢ) without full 600-taxel fidelity.

## Current Status (honest)

First run with XHAND shows **visible hand–bottle interaction** but **grasp FAIL**:

- Bottle tips horizontal under closing fingers (contacts peak ~24, then lost)
- Lift / flip not achieved
- Needs collision tuning (fingertip spheres done; palm mesh still stiff) and pose refinement

No kinematic bottle coupling — same physics honesty policy as Shadow Hand Phase 1.

## Next Tuning

1. Replace palm mesh collision with box/capsule primitives
2. Slower finger closing during `GRASP` phase
3. Align ByteDance-style virtual taxel spring model (Tier B) for smoother force readout
