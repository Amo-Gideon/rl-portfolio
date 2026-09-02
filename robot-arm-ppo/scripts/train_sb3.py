#!/usr/bin/env python3
"""
Train the arm-reaching task with Stable-Baselines3 PPO (standard stack).

Usage:
    python scripts/train_sb3.py --timesteps 100000 --device cpu
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from arm_env import ArmEnv


class SuccessCallback(BaseCallback):
    """Track success rate over the last 100 episodes and save best model."""

    def __init__(self, save_path: str, window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.window = window
        self.successes = []
        self.best_rate = -1.0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "success" in info:
                self.successes.append(float(info["success"]))
        if len(self.successes) >= self.window:
            rate = sum(self.successes[-self.window:]) / self.window
            if rate > self.best_rate:
                self.best_rate = rate
                self.model.save(self.save_path)
                if self.verbose:
                    print(f"\nNew best success rate {rate:.2f} -> saved {self.save_path}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="outputs/sb3_arm.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = ArmEnv(seed=args.seed)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device=args.device,
        seed=args.seed,
    )

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    callback = SuccessCallback(args.save, window=100)

    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.save)
    print(f"Saved final model to {args.save}")


if __name__ == "__main__":
    main()
