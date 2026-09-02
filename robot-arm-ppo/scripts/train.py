#!/usr/bin/env python3
"""
Train PPO on the 2-link arm reaching task.

Usage:
    python scripts/train.py --episodes 300 --device cpu
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from arm_env import ArmEnv
from ppo import PPO


def collect_episode(env, agent):
    obs, _ = env.reset()
    data = {"obs": [], "act": [], "rew": [], "val": [], "logp": [], "done": []}
    done = False
    ep_ret, ep_len, success = 0.0, 0, False
    while not done:
        action, value, logp = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        data["obs"].append(obs)
        data["act"].append(action)
        data["rew"].append(reward)
        data["val"].append(value)
        data["logp"].append(logp)
        data["done"].append(float(terminated))
        obs = next_obs
        ep_ret += reward
        ep_len += 1
        success = info["success"]
    return data, ep_ret, ep_len, success


def episodes_to_batch(episodes, agent):
    """Compute GAE per episode and concatenate into one update batch."""
    batch = {"obs": [], "act": [], "logp": [], "adv": [], "ret": []}
    for data in episodes:
        adv, ret = agent.compute_gae(
            np.array(data["rew"]), np.array(data["val"]), np.array(data["done"])
        )
        batch["obs"].extend(data["obs"])
        batch["act"].extend(data["act"])
        batch["logp"].extend(data["logp"])
        batch["adv"].extend(adv)
        batch["ret"].extend(ret)
    return {k: np.array(v) for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--batch_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="outputs/ppo_arm.pt")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = ArmEnv(max_steps=args.max_steps, seed=args.seed)
    agent = PPO(obs_dim=6, act_dim=2, lr=args.lr, device=args.device)

    recent_success = []
    total_ep = 0
    while total_ep < args.episodes:
        episodes, rets, succs = [], [], []
        for _ in range(args.batch_episodes):
            data, ep_ret, ep_len, success = collect_episode(env, agent)
            episodes.append(data)
            rets.append(ep_ret)
            succs.append(float(success))
            recent_success.append(float(success))
            if len(recent_success) > 50:
                recent_success.pop(0)
            total_ep += 1

        batch = episodes_to_batch(episodes, agent)
        loss = agent.update(batch)

        print(
            f"Ep {total_ep:4d} | return {np.mean(rets):7.3f} | "
            f"loss {loss:.4f} | success(50ep) {np.mean(recent_success):.2f}"
        )

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.save)
    print(f"Saved model to {args.save}")


if __name__ == "__main__":
    main()
