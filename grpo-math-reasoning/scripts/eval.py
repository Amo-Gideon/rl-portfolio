#!/usr/bin/env python3
"""
Greedy evaluation of a trained math model.

Usage:
    python scripts/eval.py --checkpoint outputs/final --num_tasks 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from tasks import generate_tasks, reward


def evaluate(checkpoint: str, base_model: str, num_tasks: int = 100, max_new_tokens: int = 256, seed: int = 999):
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.float32, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model.eval()

    tasks = generate_tasks(num_tasks=num_tasks, seed=seed)
    correct = 0
    total_reward = 0.0
    for task in tasks:
        messages = [
            {"role": "system", "content": "You are a careful math solver."},
            {"role": "user", "content": task["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        r = reward(text, task["answer"])
        total_reward += r["total_reward"]
        correct += int(r["answer_reward"] == 1.0)

    print(f"Evaluated {num_tasks} tasks")
    print(f"Accuracy: {correct}/{num_tasks} = {correct/num_tasks:.1%}")
    print(f"Avg reward: {total_reward/num_tasks:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="./models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num_tasks", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.base_model, args.num_tasks)
