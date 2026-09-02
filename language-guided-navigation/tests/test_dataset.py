"""Tests for dataset generation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from env import LangNavEnv
from dataset import generate_expert_trajectory, generate_expert_dataset


def test_expert_reaches_target():
    env = LangNavEnv(size=5, max_steps=15, seed=1)
    task = env._sample_task()
    pairs = generate_expert_trajectory(env, task)
    final = json.loads(pairs[-1]["action"])
    assert "final_answer" in final
    assert final["final_answer"] == task["answer"]


def test_dataset_size():
    data = generate_expert_dataset(num_tasks=10, size=4, seed=0)
    assert len(data) > 0


if __name__ == "__main__":
    import json
    test_expert_reaches_target()
    test_dataset_size()
    print("dataset tests passed")
