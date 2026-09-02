"""Tests for math task generation and reward."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tasks import generate_tasks, extract_answer, reward


def test_generation():
    tasks = generate_tasks(num_tasks=50, seed=0)
    assert len(tasks) == 50
    for t in tasks:
        assert "prompt" in t and "answer" in t
        int(t["answer"])  # answer must parse as int


def test_extract_answer():
    assert extract_answer("... reasoning ...\nANSWER: 42") == "42"
    assert extract_answer("The answer is 7.") == "7"
    assert extract_answer("ANSWER: 1,234") == "1234"


def test_reward():
    r = reward("Some steps.\nANSWER: 42", "42")
    assert r["answer_reward"] == 1.0
    r = reward("ANSWER: 41", "42")
    assert r["answer_reward"] == 0.0
    r = reward("I think 42", "42")
    assert r["format_reward"] == 0.0
    assert r["answer_reward"] == 1.0  # fallback number extraction


if __name__ == "__main__":
    test_generation()
    test_extract_answer()
    test_reward()
    print("grpo-math tests passed")
