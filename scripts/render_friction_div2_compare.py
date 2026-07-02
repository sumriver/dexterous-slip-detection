#!/usr/bin/env python3
"""Side-by-side video: friction÷2 open-loop (FAIL) vs anti-slip (PASS)."""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.spider_ketchup import DEFAULT_WORKSPACE
from sim.spider_replay import SpiderTaskConfig, replay_spider_task

SPIDER = ROOT / "third_party" / "spider"
OUT_DIR = ROOT / "data" / "ketchup_robustness" / "compare"
ARTIFACTS = Path("/opt/cursor/artifacts/ketchup-fail-videos")
FINAL_NAME = "friction_div2_compare.mp4"

PANEL_W = 1280
PANEL_H = 720
FPS = 30


def _run_case(antislip: bool, out_sub: str) -> Path:
    cfg = SpiderTaskConfig(
        dataset_dir=SPIDER / "example_datasets",
        dataset_name="arcticv2",
        robot_type="xhand",
        embodiment_type="right",
        task="s01-ketchup_use_01",
        workspace_root=DEFAULT_WORKSPACE,
    )
    case_dir = OUT_DIR / out_sub
    result = replay_spider_task(
        cfg,
        case_dir,
        save_video=True,
        video_fps=FPS,
        post_lift_m=0.10,
        post_extend_s=2.0,
        post_mimic_s=1.0,
        friction_scale=0.5,
        log_energy=False,
        antislip=antislip,
    )
    assert result.video_path is not None
    return result.video_path


def _load_font(size: int = 28) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        p = Path(name)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _annotate_panel(
    frame: np.ndarray,
    *,
    status: str,
    status_color: tuple[int, int, int],
    line1: str,
    line2: str,
) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font_lg = _load_font(30)
    font_sm = _load_font(22)

    banner_h = 88
    draw.rectangle([0, 0, img.width, banner_h], fill=(16, 16, 20))
    draw.text((16, 10), status, fill=status_color, font=font_lg)
    draw.text((16, 46), line1, fill=(230, 230, 230), font=font_sm)
    draw.text((min(16, img.width - 400), 66), line2, fill=(180, 180, 180), font=font_sm)

    # bottom strip: outcome
    strip_h = 36
    y0 = img.height - strip_h
    draw.rectangle([0, y0, img.width, img.height], fill=(16, 16, 20))
    draw.text((16, y0 + 6), line2, fill=status_color, font=font_sm)

    return np.asarray(img)


def _resize_panel(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(frame).resize((PANEL_W, PANEL_H), Image.Resampling.LANCZOS)
    return np.asarray(img)


def compose_side_by_side(left_path: Path, right_path: Path, out_path: Path) -> None:
    left = iio.imread(left_path)
    right = iio.imread(right_path)
    if left.ndim == 3:
        left = left[None, ...]
    if right.ndim == 3:
        right = right[None, ...]

    n = min(len(left), len(right))
    font_note = "friction ÷2  |  ketchup right-hand extend +10cm"

    out_frames: list[np.ndarray] = []
    for i in range(n):
        lf = _annotate_panel(
            _resize_panel(left[i]),
            status="FAIL",
            status_color=(255, 90, 90),
            line1=font_note,
            line2="开环回放 · 无防滑算法 · 物体滑落",
        )
        rf = _annotate_panel(
            _resize_panel(right[i]),
            status="PASS",
            status_color=(90, 220, 120),
            line1=font_note,
            line2="方案1防滑 · 中心背离检测 + 握力增大",
        )
        out_frames.append(np.concatenate([lf, rf], axis=1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        out_path,
        np.stack(out_frames, axis=0),
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
    )


def main() -> None:
    if not DEFAULT_WORKSPACE.joinpath("scene.xml").exists():
        print("Run: python3 scripts/build_spider_ketchup_right.py", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Recording open-loop (FAIL)...")
    left_mp4 = _run_case(antislip=False, out_sub="open_loop")
    print("Recording anti-slip (PASS)...")
    right_mp4 = _run_case(antislip=True, out_sub="antislip")

    final = OUT_DIR / FINAL_NAME
    print("Composing side-by-side...")
    compose_side_by_side(left_mp4, right_mp4, final)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACTS / FINAL_NAME
    artifact.write_bytes(final.read_bytes())

    print(f"Done: {final}")
    print(f"Artifact: {artifact}")


if __name__ == "__main__":
    main()
