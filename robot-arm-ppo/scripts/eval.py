#!/usr/bin/env python3
"""
Evaluate a trained arm policy (custom PPO or Stable-Baselines3).

Usage:
    python scripts/eval.py --checkpoint outputs/ppo_arm.pt --episodes 50
    python scripts/eval.py --checkpoint outputs/sb3_arm.zip --sb3 --episodes 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from arm_env import ArmEnv


def make_agent(checkpoint: str, sb3: bool):
    if sb3:
        from stable_baselines3 import PPO
        model = PPO.load(checkpoint)
        return lambda obs: model.predict(obs, deterministic=True)[0]
    from ppo import PPO
    agent = PPO(obs_dim=6, act_dim=2)
    agent.load(checkpoint)
    return lambda obs: agent.act(obs)[0]


def evaluate(checkpoint: str, sb3: bool, episodes: int = 50, max_steps: int = 100, seed: int = 123):
    policy = make_agent(checkpoint, sb3)

    env = ArmEnv(max_steps=max_steps, seed=seed)
    successes, distances, lengths = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes.append(float(info["success"]))
        distances.append(info["distance"])
        lengths.append(env.step_count)

    print(f"Evaluated {episodes} episodes")
    print(f"Success rate: {np.mean(successes):.1%}")
    print(f"Final distance: {np.mean(distances):.4f} +- {np.std(distances):.4f}")
    print(f"Avg episode length: {np.mean(lengths):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sb3", action="store_true", help="load a Stable-Baselines3 model")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.sb3, args.episodes, args.max_steps)
