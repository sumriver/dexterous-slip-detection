#!/usr/bin/env python3
"""Record NN-1 closed-loop demo MP4s + friction÷2 open-loop vs NN side-by-side.

Outputs:
  data/slip_nn/videos/nn_baseline.mp4
  data/slip_nn/videos/nn_friction_div2.mp4
  data/slip_nn/videos/friction_div2_openloop_vs_nn.mp4
  /opt/cursor/artifacts/nn1_videos/*
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sim.spider_ketchup import DEFAULT_WORKSPACE  # noqa: E402
from sim.spider_replay import SpiderTaskConfig, replay_spider_task  # noqa: E402
from sim.slip_nn_detector import load_detector_from_dir  # noqa: E402

SPIDER = ROOT / "third_party" / "spider"
OUT_DIR = ROOT / "data" / "slip_nn" / "videos"
ARTIFACTS = Path("/opt/cursor/artifacts/nn1_videos")
MODEL_DIR = ROOT / "models" / "slip_nn"
FPS = 30
PANEL_W, PANEL_H = 960, 544  # H divisible by 16 for libx264


def _threshold() -> float:
    meta = MODEL_DIR / "train_meta.json"
    if meta.exists():
        return float(json.loads(meta.read_text()).get("default_threshold", 0.7))
    return 0.7


def _cfg() -> SpiderTaskConfig:
    return SpiderTaskConfig(
        dataset_dir=SPIDER / "example_datasets",
        dataset_name="arcticv2",
        robot_type="xhand",
        embodiment_type="right",
        task="s01-ketchup_use_01",
        workspace_root=DEFAULT_WORKSPACE,
    )


def _replay(
    *,
    out_sub: str,
    friction_scale: float,
    antislip_nn: bool,
    antislip: bool = False,
) -> Path:
    case_dir = OUT_DIR / out_sub
    case_dir.mkdir(parents=True, exist_ok=True)
    nn_detector = None
    if antislip_nn:
        nn_detector = load_detector_from_dir(MODEL_DIR, threshold=_threshold())
    result = replay_spider_task(
        _cfg(),
        case_dir,
        save_video=True,
        video_fps=FPS,
        post_lift_m=0.10,
        post_extend_s=2.0,
        post_mimic_s=1.0,
        friction_scale=friction_scale,
        log_energy=False,
        antislip=antislip and not antislip_nn,
        antislip_nn=antislip_nn,
        nn_detector=nn_detector,
    )
    assert result.video_path is not None, f"no video for {out_sub}"
    return result.video_path


def _load_font(size: int = 26) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        p = Path(name)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _annotate(
    frame: np.ndarray,
    *,
    status: str,
    status_color: tuple[int, int, int],
    line1: str,
    line2: str,
) -> np.ndarray:
    img = Image.fromarray(frame).resize((PANEL_W, PANEL_H), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    font_lg = _load_font(28)
    font_sm = _load_font(20)
    draw.rectangle([0, 0, img.width, 72], fill=(16, 16, 20))
    draw.text((14, 8), status, fill=status_color, font=font_lg)
    draw.text((14, 42), line1, fill=(220, 220, 220), font=font_sm)
    y0 = img.height - 34
    draw.rectangle([0, y0, img.width, img.height], fill=(16, 16, 20))
    draw.text((14, y0 + 6), line2, fill=status_color, font=font_sm)
    return np.asarray(img)


def compose_side_by_side(left: Path, right: Path, out: Path) -> None:
    L = iio.imread(left)
    R = iio.imread(right)
    if L.ndim == 3:
        L = L[None, ...]
    if R.ndim == 3:
        R = R[None, ...]
    n = min(len(L), len(R))
    frames = []
    for i in range(n):
        lf = _annotate(
            L[i],
            status="FAIL",
            status_color=(255, 90, 90),
            line1="friction ÷2 · open-loop (no anti-slip)",
            line2="物体滑落 · 接触丢失",
        )
        rf = _annotate(
            R[i],
            status="PASS",
            status_color=(90, 220, 120),
            line1="friction ÷2 · NN-1 TCN (--antislip-nn)",
            line2="y_event · τ=0.7 · latch+confirm",
        )
        frames.append(np.concatenate([lf, rf], axis=1))
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, np.stack(frames, axis=0), fps=FPS, codec="libx264", pixelformat="yuv420p")


def main() -> None:
    if not DEFAULT_WORKSPACE.joinpath("scene.xml").exists():
        print("Run: python3 scripts/build_spider_ketchup_right.py", file=sys.stderr)
        sys.exit(1)
    if not any(MODEL_DIR.glob("*.pt")):
        print(f"Missing checkpoint in {MODEL_DIR}", file=sys.stderr)
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("Recording NN baseline...")
    nn_base = _replay(out_sub="nn_baseline", friction_scale=1.0, antislip_nn=True)
    named_base = OUT_DIR / "nn_baseline.mp4"
    shutil.copy2(nn_base, named_base)

    print("Recording NN friction÷2...")
    nn_f2 = _replay(out_sub="nn_friction_div2", friction_scale=0.5, antislip_nn=True)
    named_nn = OUT_DIR / "nn_friction_div2.mp4"
    shutil.copy2(nn_f2, named_nn)

    print("Recording open-loop friction÷2 (FAIL)...")
    ol_f2 = _replay(
        out_sub="openloop_friction_div2",
        friction_scale=0.5,
        antislip_nn=False,
        antislip=False,
    )
    named_ol = OUT_DIR / "openloop_friction_div2.mp4"
    shutil.copy2(ol_f2, named_ol)

    compare = OUT_DIR / "friction_div2_openloop_vs_nn.mp4"
    print("Composing side-by-side...")
    compose_side_by_side(named_ol, named_nn, compare)

    for p in (named_base, named_nn, named_ol, compare):
        dest = ARTIFACTS / p.name
        shutil.copy2(p, dest)
        print(f"  artifact: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    print(f"Done. Videos in {OUT_DIR}")


if __name__ == "__main__":
    main()
