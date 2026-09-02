"""
Dataset utilities for language-guided navigation.

- Expert trajectories via BFS for supervised warm-start.
- Random rollouts for RL collection.
"""

import json
import random
from collections import deque
from typing import Dict, List

from env import LangNavEnv


def shortest_path_actions(env: LangNavEnv, task: Dict) -> List[str]:
    """Return a shortest path of primitive actions from start to target."""
    start = tuple(task["start"])
    target = task["objects"][task["target"]]
    size = env.size
    # BFS over (x, y, facing)
    start_state = (start[0], start[1], task["start_dir"])
    target_state = (target[0], target[1], None)  # ignore final facing
    if (start_state[0], start_state[1]) == (target[0], target[1]):
        return []

    visited = {start_state}
    queue = deque([(start_state, [])])
    dirs = env.DIRS
    while queue:
        (x, y, f), actions = queue.popleft()
        # turn left / right
        for a_name, delta in [("turn_left", -1), ("turn_right", 1)]:
            nf = dirs[(dirs.index(f) + delta) % 4]
            state = (x, y, nf)
            if state not in visited:
                visited.add(state)
                queue.append((state, actions + [a_name]))
        # move forward
        dx, dy = env.DIR_VECTORS[f]
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            state = (nx, ny, f)
            if state not in visited:
                new_actions = actions + ["move_forward"]
                if (nx, ny) == (target[0], target[1]):
                    return new_actions
                visited.add(state)
                queue.append((state, new_actions))
    return []


def generate_expert_trajectory(env: LangNavEnv, task: Dict) -> List[Dict]:
    """Generate (observation, action_text) pairs for SFT."""
    obs = env.reset(task)
    actions = shortest_path_actions(env, task)
    pairs = []
    for a in actions:
        action_text = json.dumps({"thought": f"Moving toward the {task['target']}.", "action": a})
        pairs.append({"observation": obs, "action": action_text})
        obs, _, done = env.step(action_text)
        if done:
            break
    answer_text = json.dumps({"thought": "I have arrived.", "final_answer": task["answer"]})
    pairs.append({"observation": obs, "action": answer_text})
    return pairs


def generate_expert_dataset(num_tasks: int = 200, size: int = 5, seed: int = 42) -> List[Dict]:
    """Generate a supervised fine-tuning dataset."""
    rng = random.Random(seed)
    env = LangNavEnv(size=size, max_steps=15, seed=seed)
    dataset = []
    for _ in range(num_tasks):
        task = env._sample_task()
        # Ensure target is reachable from (0,0)
        if task["objects"][task["target"]] == (0, 0):
            continue
        try:
            pairs = generate_expert_trajectory(env, task)
            dataset.extend(pairs)
        except Exception:
            continue
    return dataset


def format_sft_example(pair: Dict) -> str:
    """Format one observation-action pair as text for causal LM training."""
    return f"{pair['observation']}\nAction: {pair['action']}"
