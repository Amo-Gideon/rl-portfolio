"""
Evaluate a trained GRPO on held-out tasks.

Usage:
    Python scripts/evaluate.py -- checkpoint checkpoints/grpo/final
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.multi_tool_env import MultiToolEnv
from src.rewards.verifiable_reward import VerifiableReward
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def evaluate(checkpoint_path: str, num_tasks: int= 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True).to(device)
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

    for i in range(min(num_tasks, len(env.tasks))):
        obs = env.reset(task_idx=i)
        task = env.get_current_task()

        inputs = tokenizer(obs, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = 128,
                do_sample = False,
                pad_token_id = tokenizer.pad_token_id
            )
        
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        parsed= env._parse_action(response_text)
        trajectory = {"actions": [parsed], "infos": [], "final_asnwer": ""}

        if parsed and "final_answer" in parsed:
            trajectory["final_asnwer"] = str(parsed["final_answer"])
            trajectory["infos"].append({"format_valid": True, "tool_valid": False})
        elif parsed and "tool_name" in parsed:
            _, info, _ = env.step(response_text)
            trajectory["infos"].append(info)
        else:
            trajectory["infos"].append({"format_valid": False, "tool_valid": False})
        
        reward_dict = reward_fn.compute(trajectory, env.get_task_answer())
        total_reward += reward_dict["total_reward"]
        

        if reward_dict["answer_accuracy_reward"] >= 0.9:
            successes += 1
        if reward_dict["format_reward"] > 0:
            format_valids += 1
        if reward_dict["tool_validity_reward"] > 0:
            tool_valids += 1
    
    n = min(num_tasks, len(env.tasks))
    print(f"\nEvaluation Results ({n} tasks):")
    print(f"  Task Success Rate: {successes/n*100:.1f}%")
    print(f"  Format Compliance: {format_valids/n*100:.1f}%")
    print(f"  Tool Validity:     {tool_valids/n*100:.1f}%")
    print(f"  Avg Total Reward:  {total_reward/n:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_tasks", type=int, default=50)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.num_tasks)