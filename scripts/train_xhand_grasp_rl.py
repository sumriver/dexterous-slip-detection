#!/usr/bin/env python3
"""Train PPO policy for XHAND bottle grasp + lift (no static phase script)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from sim.xhand_grasp_env import XHandGraspEnv

MODEL_DIR = ROOT / "models" / "rl"
TENSORBOARD_DIR = ROOT / "data" / "rl" / "tensorboard"


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO training for XHAND grasp-lift")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel envs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-freq", type=int, default=50_000)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    def _make():
        return Monitor(XHandGraspEnv(frame_skip=20, max_episode_steps=500))

    env = make_vec_env(_make, n_envs=args.n_envs, seed=args.seed)
    eval_env = Monitor(XHandGraspEnv(frame_skip=20, max_episode_steps=500))

    checkpoint = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(MODEL_DIR),
        name_prefix="xhand_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best"),
        log_path=str(ROOT / "data" / "rl" / "eval"),
        eval_freq=max(25_000 // args.n_envs, 1),
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=str(TENSORBOARD_DIR),
    )

    print(f"Training PPO for {args.timesteps} timesteps ({args.n_envs} envs)...")
    model.learn(total_timesteps=args.timesteps, callback=[checkpoint, eval_cb], progress_bar=True)
    final_path = MODEL_DIR / "xhand_ppo_final.zip"
    model.save(final_path)
    print(f"Saved: {final_path}")


if __name__ == "__main__":
    main()
