#!/usr/bin/env python3
"""Side-by-side ÷4 grasp comparison video: A (left, Fail) vs C (right, Success).

A = normal-speed open-loop grasp -> bottle squirts out immediately (Fail).
C = slow grasp + release grip when horizontal |F_n| > |F_t| -> bottle is
    grasped and held through the grasp motion (Success at the grasp phase).

Scope: the GRASP (trajectory) phase, where the two strategies decisively
differ. (At friction ÷4 the object still cannot survive the later static
hold/lift for any strategy — see docs/水平防滑力积分分析.md.)

Frames are rendered in real time (C's slow grasp takes ~2x longer; A is
padded with its final frame). Chinese captions use WenQuanYi Micro Hei.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.spider_ketchup import DEFAULT_WORKSPACE
from sim.spider_replay import (
    SpiderTaskConfig,
    get_object_geom_ids,
    get_spider_hand_collision_geom_ids,
    load_trajectory_arrays,
    upsample_controls,
)
from sim.spider_scene_modify import apply_object_physics
from sim.antislip_control import NormalTangentGripController
from sim.slip_horizontal import compute_hand_horizontal_frame, measure_horizontal_forces
from sim.video_recorder import make_spider_ketchup_camera

OUT = ROOT / "data" / "horizontal_grip_release" / "grip_release_div4_A_vs_C.mp4"
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
W, H = 640, 480
FPS = 30
SIM_DT = 0.01
FRICTION = 0.25
SLOW_FACTOR = 2.0
DROP_Z = 0.10  # below this = object dropped


def stretch_ctrl(ctrl: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return ctrl
    T = ctrl.shape[0]
    new_T = int(round(T * factor))
    src = np.linspace(0, T - 1, new_T)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    f = (src - lo)[:, None]
    return ctrl[lo] * (1 - f) + ctrl[hi] * f


def run_capture(slow: bool, release: bool):
    cfg = SpiderTaskConfig(
        dataset_dir=ROOT / "third_party/spider/example_datasets",
        dataset_name="arcticv2", robot_type="xhand", embodiment_type="right",
        task="s01-ketchup_use_01", workspace_root=DEFAULT_WORKSPACE,
    )
    model = mujoco.MjModel.from_xml_path(str(cfg.scene_path))
    apply_object_physics(model, friction_scale=FRICTION)
    model.opt.timestep = SIM_DT
    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H
    data = mujoco.MjData(model)
    q, v, c = load_trajectory_arrays(cfg.trajectory_path, model, cfg.data_type)
    q, v, c = upsample_controls(q, v, c, SIM_DT, 0.02)
    tc = stretch_ctrl(c, SLOW_FACTOR) if slow else c

    hg = get_spider_hand_collision_geom_ids(model)
    og = get_object_geom_ids(model, "right_object")
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = make_spider_ketchup_camera(model, "right_object")
    ctl = NormalTangentGripController(
        release_step=0.005, restore_step=0.004, max_release=0.05,
        trigger_ratio=1.25, min_normal=5.0,
    )
    fn_buf: list[float] = []
    ft_buf: list[float] = []

    data.qpos[:] = q[0]; data.qvel[:] = v[0]; data.ctrl[:] = c[0]
    mujoco.mj_forward(model, data)

    interval = max(1, int(round(1.0 / (FPS * SIM_DT))))
    frames: list[np.ndarray] = []
    zs: list[float] = []
    step = 0
    for cc in tc:
        applied = cc
        if release:
            fr = compute_hand_horizontal_frame(model, data)
            r = measure_horizontal_forces(model, data, hg, og, fr)
            fn_buf.append(float(np.hypot(r.fx_normal, r.fy_normal)))
            ft_buf.append(float(np.hypot(r.fx_tangent, r.fy_tangent)))
            fn_buf, ft_buf = fn_buf[-10:], ft_buf[-10:]
            ctl.update(float(np.mean(fn_buf)), float(np.mean(ft_buf)))
            applied = ctl.apply(cc, model)
        data.ctrl[:] = applied
        mujoco.mj_step(model, data)
        if step % interval == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
            zs.append(float(data.xpos[oid][2]))
        step += 1
    renderer.close()
    return frames, zs


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def annotate(frame: np.ndarray, title: str, strategy: str, result: str | None,
             result_color: tuple[int, int, int]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    f_title = _font(24)
    f_strat = _font(18)
    f_badge = _font(40)
    # top banner
    draw.rectangle([0, 0, W, 64], fill=(0, 0, 0, 150))
    draw.text((10, 6), title, font=f_title, fill=(255, 255, 255))
    draw.text((10, 38), strategy, font=f_strat, fill=(220, 220, 220))
    if result:
        bw = draw.textlength(result, font=f_badge)
        draw.rectangle([W - bw - 24, H - 60, W - 8, H - 8], fill=(0, 0, 0, 140))
        draw.text((W - bw - 16, H - 56), result, font=f_badge, fill=result_color)
    return np.array(img)


def main() -> None:
    if not DEFAULT_WORKSPACE.joinpath("scene.xml").exists():
        print("Run: python3 scripts/build_spider_ketchup_right.py", file=sys.stderr)
        sys.exit(1)

    print("Rendering A (normal-speed, open-loop) ...")
    a_frames, a_z = run_capture(slow=False, release=False)
    print("Rendering C (slow grasp + grip release) ...")
    c_frames, c_z = run_capture(slow=True, release=True)

    n = max(len(a_frames), len(c_frames))

    def pad(frames, zs):
        if not frames:
            return frames, zs
        frames = frames + [frames[-1]] * (n - len(frames))
        zs = zs + [zs[-1]] * (n - len(zs))
        return frames, zs

    a_frames, a_z = pad(a_frames, a_z)
    c_frames, c_z = pad(c_frames, c_z)

    out_frames = []
    for i in range(n):
        a_res = "FAIL" if a_z[i] < DROP_Z else None
        c_res = "SUCCESS" if c_z[i] >= DROP_Z else "FAIL"
        left = annotate(a_frames[i], "A  正常速抓取", "策略：开环抓取（不减速）",
                        a_res, (255, 70, 70))
        right = annotate(c_frames[i], "C  慢速抓取 + 松握", "策略：慢速 + |F法向|>|F切向| 时减小握力",
                         c_res, (80, 230, 120))
        out_frames.append(np.concatenate([left, right], axis=1))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUT, np.stack(out_frames), fps=FPS, codec="libx264", pixelformat="yuv420p")
    print(f"Saved: {OUT}  ({len(out_frames)} frames, {len(out_frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
