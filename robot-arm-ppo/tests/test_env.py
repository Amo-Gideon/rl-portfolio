"""Tests for the arm environment and PPO GAE."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from arm_env import ArmEnv
from ppo import PPO


def test_fk_reach():
    env = ArmEnv(seed=0)
    # Fully stretched along x-axis
    env.joints = np.array([0.0, 0.0])
    _, ee = env._forward_kinematics()
    assert np.allclose(ee, [2.0, 0.0], atol=1e-5)
    assert abs(env.distance() - np.linalg.norm(ee - env.target)) < 1e-5


def test_episode_runs():
    env = ArmEnv(max_steps=50, seed=1)
    obs, _ = env.reset()
    assert obs.shape == (6,)
    done = False
    steps = 0
    while not done:
        obs, r, terminated, truncated, info = env.step(np.array([0.1, -0.1]))
        done = terminated or truncated
        steps += 1
    assert steps <= 50
    assert "distance" in info and "success" in info


def test_gae_shapes():
    agent = PPO(obs_dim=6, act_dim=2)
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.5, 0.5, 0.5])
    dones = np.array([0.0, 0.0, 1.0])
    adv, ret = agent.compute_gae(rewards, values, dones)
    assert adv.shape == (3,) and ret.shape == (3,)


if __name__ == "__main__":
    test_fk_reach()
    test_episode_runs()
    test_gae_shapes()
    print("arm-ppo tests passed")
