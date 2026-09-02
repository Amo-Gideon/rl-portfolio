#!/usr/bin/env python3
"""
Visualize a trained episode of the 2-link arm as PNG frames.

Usage:
    python scripts/visualize.py --checkpoint outputs/sb3_arm.zip --sb3 --output_dir assets/viz
    python scripts/visualize.py --checkpoint outputs/ppo_arm.pt --output_dir assets/viz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arm_env import ArmEnv
from eval import make_agent


def render(env: ArmEnv, step: int, output_path: Path):
    shoulder = np.array([0.0, 0.0])
    elbow, ee = env._forward_kinematics()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Step {step} | distance {env.distance():.3f}")

    ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], "o-", lw=3, color="steelblue")
    ax.plot([elbow[0], ee[0]], [elbow[1], ee[1]], "o-", lw=3, color="steelblue")

    ax.plot(env.target[0], env.target[1], "r*", ms=20, label="target")
    circle = plt.Circle(env.target, env.success_radius, color="red", alpha=0.2)
    ax.add_patch(circle)

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sb3", action="store_true", help="load a Stable-Baselines3 model")
    parser.add_argument("--output_dir", type=str, default="assets/viz")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    policy = make_agent(args.checkpoint, args.sb3)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = ArmEnv(max_steps=args.max_steps, seed=args.seed)
    obs, _ = env.reset()
    render(env, 0, out / "step_000.png")

    done = False
    step = 0
    info = {}
    while not done and env.step_count < env.max_steps:
        action = policy(obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        render(env, step, out / f"step_{step:03d}.png")

    print(f"Saved {step + 1} frames to {out}")
    print(f"Success: {info.get('success')} | Final distance: {info.get('distance', -1):.4f}")


if __name__ == "__main__":
    main()
