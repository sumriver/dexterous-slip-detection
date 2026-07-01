#!/usr/bin/env python3
"""Evaluate trained RL policy and record video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.xhand_grasp_env import XHandGraspEnv

DATA_DIR = ROOT / "data" / "rl"
DEFAULT_MODEL = ROOT / "models" / "rl" / "best" / "best_model.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-path", type=Path, default=DATA_DIR / "rl_grasp_eval.mp4")
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=250)
    args = parser.parse_args()

    if not args.model.exists():
        # fallback
        alt = ROOT / "models" / "rl" / "xhand_ppo_final.zip"
        if alt.exists():
            args.model = alt
        else:
            raise FileNotFoundError(f"No model at {args.model}. Run train_xhand_grasp_rl.py first.")

    model = PPO.load(args.model)
    env = XHandGraspEnv(
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_steps,
        randomize_reset=not args.deterministic,
        render_mode="rgb_array" if args.video else None,
    )

    best_z = 0.0
    best_ep = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_max_z = 0.0
        ep_reward = 0.0
        ep_frames: list[np.ndarray] = []
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            ep_max_z = max(ep_max_z, info.get("bottle_z", 0))
            steps += 1
            if args.video:
                frame = env.render()
                if frame is not None:
                    ep_frames.append(frame.copy())
            if term or trunc:
                break
        print(
            f"Episode {ep+1}: steps={steps} reward={ep_reward:.1f} "
            f"max_z={ep_max_z:.3f} contacts_end={info.get('n_contacts',0)} "
            f"lifted={info.get('lifted',0)} support={info.get('support_z',0):.2f}"
        )
        if ep_max_z > best_z:
            best_z = ep_max_z
            best_ep = ep + 1
            best_frames = ep_frames

    print(f"Best episode {best_ep}: max bottle z = {best_z:.3f} m (target lift +0.20)")

    out_frames = best_frames
    if args.video and out_frames:
        args.video_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.video_path, np.stack(out_frames), fps=25, codec="libx264", pixelformat="yuv420p")
        print(f"Video: {args.video_path} ({len(out_frames)} frames)")

    env.close()


if __name__ == "__main__":
    main()
