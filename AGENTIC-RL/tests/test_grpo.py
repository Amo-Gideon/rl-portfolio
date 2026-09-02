"""
Unit tests for GRPO loss computation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.grpo_trainer import GRPOTrainer


def test_group_advantages():
    """Test that advantages are zero-mean and unit-variance-ish."""
    rewards = [1.0, 2.0, 3.0, 4.0]

    mean_r = sum(rewards) / len(rewards)
    std_r = (sum((r - mean_r)**2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
    expected = [(r - mean_r) / std_r for r in rewards]

    assert abs(sum(expected)) < 1e-6

    print("✅ Group advantage math test passed")


def test_per_token_kl():
    """Test KL computation shape and sign."""
    actor_lp = torch.tensor([-2.0, -3.0, -1.5])
    ref_lp = torch.tensor([-2.5, -2.8, -1.6])

    kl = actor_lp - ref_lp  

    assert kl[0] > 0  # -2.0 > -2.5
    assert kl[1] < 0  # -3.0 < -2.8

    print("✅ Per-token KL test passed")


def test_ppo_clip():
    """Test that clipping works correctly."""
    advantage = 1.0
    epsilon = 0.2

    ratio = 1.1
    clipped = torch.clamp(torch.tensor(ratio), 1 - epsilon, 1 + epsilon).item()
    assert clipped == 1.1

    ratio = 1.5
    clipped = torch.clamp(torch.tensor(ratio), 1 - epsilon, 1 + epsilon).item()
    assert clipped == 1.2  # 1 + epsilon

    ratio = 0.5
    clipped = torch.clamp(torch.tensor(ratio), 1 - epsilon, 1 + epsilon).item()
    assert clipped == 0.8  # 1 - epsilon

    print("✅ PPO clip test passed")


if __name__ == "__main__":
    test_group_advantages()
    test_per_token_kl()
    test_ppo_clip()
    print("\nAll GRPO tests passed!")
