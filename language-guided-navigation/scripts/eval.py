#!/usr/bin/env python3
"""
Greedy evaluation of a trained navigation agent.

Usage:
    python scripts/eval.py --checkpoint outputs/sft --num_tasks 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import random

from env import LangNavEnv, reward_from_trajectory
from model_utils import load_adapter, load_base_model, load_tokenizer, generate_action


def evaluate(checkpoint: str, base_model_name: str, num_tasks: int = 20, size: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = LangNavEnv(size=size, max_steps=15)

    if Path(checkpoint).exists() and (Path(checkpoint) / "adapter_config.json").exists():
        model, tokenizer = load_adapter(checkpoint, base_model_name)
    else:
        tokenizer = load_tokenizer(base_model_name)
        model = load_base_model(base_model_name)

    model.eval()

    rng = random.Random(123)
    tasks = []
    while len(tasks) < num_tasks:
        t = env._sample_task()
        if t["objects"][t["target"]] != tuple(t["start"]):
            tasks.append(t)

    successes = 0
    total_reward = 0.0
    for task in tasks:
        obs = env.reset(task)
        transitions = []
        done = False
        while not done and env.step_count < env.max_steps:
            response = generate_action(model, tokenizer, obs, do_sample=False)
            obs, info, done = env.step(response)
            transitions.append({"info": info})

        final_answer = ""
        if transitions and transitions[-1]["info"].get("parsed"):
            parsed = transitions[-1]["info"]["parsed"]
            if isinstance(parsed, dict) and "final_answer" in parsed:
                final_answer = str(parsed["final_answer"])

        traj = {
            "actions": [t["info"].get("parsed") for t in transitions],
            "infos": [t["info"] for t in transitions],
            "final_answer": final_answer,
        }
        reward_dict = reward_from_trajectory(traj, env)
        total_reward += reward_dict["total_reward"]
        if reward_dict["answer_accuracy_reward"] >= 0.9:
            successes += 1

    print(f"Evaluated {num_tasks} tasks")
    print(f"Success rate: {successes}/{num_tasks} = {successes/num_tasks:.1%}")
    print(f"Avg reward:   {total_reward/num_tasks:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num_tasks", type=int, default=20)
    parser.add_argument("--size", type=int, default=5)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.base_model, args.num_tasks, args.size)
