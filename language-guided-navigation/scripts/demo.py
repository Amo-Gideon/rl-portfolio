#!/usr/bin/env python3
"""
Interactive demo of one navigation episode.

Usage:
    python scripts/demo.py --checkpoint outputs/sft
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from env import LangNavEnv
from model_utils import load_adapter, load_base_model, load_tokenizer, generate_action


def demo(checkpoint: str, base_model_name: str, size: int = 5):
    if Path(checkpoint).exists() and (Path(checkpoint) / "adapter_config.json").exists():
        model, tokenizer = load_adapter(checkpoint, base_model_name)
    else:
        tokenizer = load_tokenizer(base_model_name)
        model = load_base_model(base_model_name)
    model.eval()

    env = LangNavEnv(size=size, max_steps=15, seed=7)
    task = env._sample_task()
    obs = env.reset(task)

    print("=" * 60)
    print(f"Task: {task['instruction']}")
    print(f"Objects: {task['objects']}")
    print("=" * 60)

    done = False
    while not done and env.step_count < env.max_steps:
        print(f"\n{obs}")
        response = generate_action(model, tokenizer, obs, do_sample=False)
        print(f"Agent: {response}")
        obs, info, done = env.step(response)
        if info.get("parsed") and "final_answer" in info["parsed"]:
            print(f"Final answer: {info['parsed']['final_answer']} (GT: {task['answer']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/sft")
    parser.add_argument("--base_model", type=str, default="./models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--size", type=int, default=5)
    args = parser.parse_args()

    demo(args.checkpoint, args.base_model, args.size)
