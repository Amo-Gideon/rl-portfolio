"""Unit tests for the navigation environment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from env import LangNavEnv, reward_from_trajectory


def test_move_and_answer():
    task = {
        "instruction": "Go to the red house and report its color.",
        "answer": "red",
        "start": [0, 0],
        "start_dir": "N",
        "objects": {"red house": (0, 1)},
        "target": "red house",
    }
    env = LangNavEnv(size=3, max_steps=5)
    obs = env.reset(task)
    assert "red house" in obs
    obs, info, done = env.step('{"thought": "move", "action": "move_forward"}')
    assert env.position == (0, 1)
    assert not done
    obs, info, done = env.step('{"thought": "answer", "final_answer": "red"}')
    assert done
    traj = {"actions": [info["parsed"]], "infos": [info], "final_answer": "red"}
    rewards = reward_from_trajectory(traj, env)
    assert rewards["answer_accuracy_reward"] == 1.0


def test_invalid_action():
    env = LangNavEnv(size=3, max_steps=3, seed=0)
    env.reset()
    obs, info, done = env.step('not json')
    assert not info["format_valid"]
    obs, info, done = env.step('{"thought": "x", "action": "jump"}')
    assert info["format_valid"]
    assert not info["action_valid"]


if __name__ == "__main__":
    test_move_and_answer()
    test_invalid_action()
    print("env tests passed")
