#!/usr/bin/env python3
"""Small PPO experiment: Policy-2 action on ketchup extend (div2-focused).

Freezes the demo arm trajectory; learns only (grip, Δwrist) per step.
Does **not** train the NN detect backbone — obs are physics features only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from sim.ketchup_extend_rl_env import ExtendCase, KetchupExtendRLEnv

OUT_DIR = ROOT / "models" / "rl" / "policy2_extend"
TB_DIR = ROOT / "data" / "rl" / "policy2_extend" / "tb"


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO smoke for ketchup extend Policy-2")
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--g-max", type=float, default=0.25)
    parser.add_argument("--d-max", type=float, default=0.25)
    parser.add_argument(
        "--cases",
        default="friction_div2,baseline,friction_s045,friction_s040",
        help="Comma list of case names",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    name_map = {
        "friction_div2": ExtendCase("friction_div2", friction_scale=0.50),
        "baseline": ExtendCase("baseline", friction_scale=1.0),
        "friction_s045": ExtendCase("friction_s045", friction_scale=0.45),
        "friction_s040": ExtendCase("friction_s040", friction_scale=0.40),
    }
    cases = tuple(name_map[n.strip()] for n in args.cases.split(",") if n.strip())

    args.out.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)

    def _make():
        return Monitor(
            KetchupExtendRLEnv(
                cases=cases,
                g_max=args.g_max,
                d_max=args.d_max,
                grip_ratchet=False,
            )
        )

    print(f"Building {args.n_envs} envs (traj snapshots) ...", flush=True)
    t0 = time.time()
    env = make_vec_env(_make, n_envs=args.n_envs, seed=args.seed)
    print(f"envs ready in {time.time() - t0:.1f}s", flush=True)

    checkpoint = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path=str(args.out),
        name_prefix="p2_extend_ppo",
    )

    if args.resume and args.resume.exists():
        model = PPO.load(args.resume, env=env, tensorboard_log=str(TB_DIR))
        print(f"Resumed {args.resume}", flush=True)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            ent_coef=0.01,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            tensorboard_log=str(TB_DIR),
        )

    print(f"Training PPO timesteps={args.timesteps} cases={[c.name for c in cases]}", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=[checkpoint], progress_bar=False)
    final = args.out / "p2_extend_ppo_final.zip"
    model.save(final)
    meta = {
        "arch": "ppo_policy2_extend",
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "g_max": args.g_max,
        "d_max": args.d_max,
        "cases": [c.name for c in cases],
        "final": str(final),
        "note": "Smoke RL: per-step grip+wrist on extend; detect frozen/out-of-loop",
    }
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out / "README.md").write_text(
        "# Policy-2 extend PPO (smoke)\n\n"
        "Learns `(grip, Δwrist)` during ketchup extend only.\n"
        "Trajectory / detect backbone not trained.\n\n"
        f"- timesteps: {args.timesteps}\n"
        f"- cases: {[c.name for c in cases]}\n"
        f"- checkpoint: `{final.name}`\n"
    )
    print(f"Saved {final}", flush=True)


if __name__ == "__main__":
    main()
