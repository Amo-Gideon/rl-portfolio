"""
Evaluate a trained GRPO checkpoint on held-out tasks.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/grpo/final --num_tasks 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.envs.multi_tool_env import MultiToolEnv
from src.rewards.verifiable_reward import VerifiableReward


def run_episode(env, model, tokenizer, device, max_new_tokens: int = 128):
    """Run one full episode using greedy generation."""
    obs = env.reset()
    done = False
    trajectory = {"actions": [], "infos": [], "observations": [obs], "final_answer": ""}

    while not done and env.step_count < env.max_steps:
        inputs = tokenizer(obs, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        obs, info, done = env.step(response_text)
        trajectory["actions"].append(info.get("parsed_action"))
        trajectory["infos"].append(info)
        trajectory["observations"].append(obs)

        if done and info.get("final_answer"):
            trajectory["final_answer"] = str(info["final_answer"]).strip()

    return trajectory


def evaluate(checkpoint_path: str, num_tasks: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    env = MultiToolEnv(max_steps=5)
    reward_fn = VerifiableReward()

    successes = 0
    format_valids = 0
    tool_valids = 0
    total_reward = 0.0

    n = min(num_tasks, len(env.tasks))
    for i in range(n):
        trajectory = run_episode(env, model, tokenizer, device)
        reward_dict = reward_fn.compute(trajectory, env.get_task_answer())
        total_reward += reward_dict["total_reward"]

        if reward_dict["answer_accuracy_reward"] >= 0.9:
            successes += 1
        if reward_dict["format_reward"] > 0:
            format_valids += 1
        if reward_dict["tool_validity_reward"] > 0:
            tool_valids += 1

    print(f"\nEvaluation Results ({n} tasks):")
    print(f"  Task Success Rate: {successes / n * 100:.1f}%")
    print(f"  Format Compliance: {format_valids / n * 100:.1f}%")
    print(f"  Tool Validity:     {tool_valids / n * 100:.1f}%")
    print(f"  Avg Total Reward:  {total_reward / n:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_tasks", type=int, default=50)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.num_tasks)
