"""
Unit tests for MultiToolEnv.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.multi_tool_env import MultiToolEnv, ToolExecutor


def test_tool_executor():
    executor = ToolExecutor()

    result = executor.search("population of France")
    assert "68 million" in result or "No relevant" in result

    result = executor.calculate("15 * 23 + 7")
    assert "352" in result

    result = executor.calculate("import os")
    assert "Invalid characters" in result

    print("✅ ToolExecutor tests passed")


def test_env_lifecycle():
    env = MultiToolEnv(max_steps=3)
    obs = env.reset(task_idx=0)
    assert "Task:" in obs
    assert env.current_task is not None

    action = '{"thought": "test", "tool_name": "search", "tool_input": "population of France"}'
    obs, info, done = env.step(action)
    assert info["format_valid"] is True
    assert info["tool_valid"] is True
    assert "Tool 'search' result" in obs

    obs, info, done = env.step("not json")
    assert info["format_valid"] is False

    action = '{"thought": "done", "final_answer": "34000000"}'
    obs, info, done = env.step(action)
    assert done is True
    assert info["format_valid"] is True

    print("✅ Env lifecycle tests passed")


def test_env_max_steps():
    env = MultiToolEnv(max_steps=2)
    env.reset(task_idx=0)

    action = '{"thought": "test", "tool_name": "search", "tool_input": "test"}'
    _, _, done = env.step(action)
    assert done is False

    _, _, done = env.step(action)
    assert done is True  # Max steps reached

    print("✅ Max steps test passed")


if __name__ == "__main__":
    test_tool_executor()
    test_env_lifecycle()
    test_env_max_steps()
    print("\nAll env tests passed!")
