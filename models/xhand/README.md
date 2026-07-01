# XHAND1 Simulation Assets

URDF/MJCF for Robotera XHAND1 right hand, used in MuJoCo grasp simulation.

## Source

| Asset | Origin | License |
|-------|--------|---------|
| `xhand_right.urdf` + STL meshes | [MaureenZOU/worldstring](https://github.com/MaureenZOU/worldstring) `assets/xhand_right/` | See upstream repo |
| Same model also appears in | [RoboVerseOrg/roboverse_data](https://huggingface.co/datasets/RoboVerseOrg/roboverse_data) `robots/xhand_right/` | Research dataset |

Fetched automatically by `bash scripts/setup_models.sh` (sparse clone of worldstring).

## Files

- `xhand_right.urdf` — ROS-compatible URDF (package:// paths rewritten to `meshes/`)
- `xhand_right.xml` — auto-generated from URDF via `urdf2mjcf`
- `xhand_right_sim.xml` — sim-ready MJCF (position actuators, keyframes, tactile sites); built by `scripts/build_xhand_mjcf.py`
- `meshes/` — symlink to `third_party/worldstring/assets/xhand_right/meshes`

## DOF

12 active revolute joints: thumb (3), index (3 incl. lateral bend), middle/ring/pinky (2 each).

## Run

```bash
bash scripts/setup_models.sh
python scripts/run_xhand_grasp_sim.py
```
