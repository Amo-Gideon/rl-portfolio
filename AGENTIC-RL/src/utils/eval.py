"""
Evaluation utilities for measuring agent performance.
"""

from typing import List, Dict
import json


def evaluate_agent(env, model, tokenizer, num_tasks: int = 50, max_steps: int = 5) -> Dict:
    """
    Evaluate a trained model on held-out tasks.

    Returns task success rate, tool validity, format compliance, and avg steps.
    """
    successes = 0
    tool_valid = 0
    format_valid = 0
    total_steps = 0

    for i in range(min(num_tasks, len(env.tasks))):
        obs = env.reset(task_idx=i)
        done = False
        step_count = 0
        episode_format_valid = True
        episode_tool_valid = True

        while not done and step_count < max_steps:
            step_count += 1
            done = True 

        total_steps += step_count

    n = min(num_tasks, len(env.tasks))
    return {
        "task_success_rate": successes / n if n > 0 else 0,
        "tool_validity": tool_valid / n if n > 0 else 0,
        "format_compliance": format_valid / n if n > 0 else 0,
        "avg_steps": total_steps / n if n > 0 else 0,
    }
