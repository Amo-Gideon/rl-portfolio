#!/usr/bin/env python3
"""Evaluation script for before/after comparison across all stages."""
import argparse
import sys
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlhf_pipeline.models.reward_model import NeuralRewardModel


def load_model_and_tokenizer(model_path, base_model_name, device="cuda"):
    """Load a model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=150, 
                      temperature=0.7, top_p=0.9, device="cuda"):
    """Generate a response for a prompt."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )


def evaluate_models(base_path, sft_path, ppo_path, rm_path, test_prompts, output_dir):
    """Evaluate base, SFT, and PPO models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # Load reward model for scoring
    rm_tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    reward_model = NeuralRewardModel(rm_path, rm_tokenizer, device=str(device))

    results = {
        "base": {"responses": [], "rewards": []},
        "sft": {"responses": [], "rewards": []},
        "ppo": {"responses": [], "rewards": []},
    }

    for model_name, model_path in [("base", base_path), ("sft", sft_path), ("ppo", ppo_path)]:
        print(f"\nEvaluating {model_name.upper()} model...")
        model, tokenizer = load_model_and_tokenizer(model_path, base_name, device)
        model.eval()

        for prompt in test_prompts:
            response = generate_response(model, tokenizer, prompt, device=device)
            reward = reward_model.score(prompt, response)
            results[model_name]["responses"].append(response)
            results[model_name]["rewards"].append(reward)
            print(f"  [{reward:.3f}] {response[:80]}...")

    # Compute statistics
    for model_name in ["base", "sft", "ppo"]:
        rewards = results[model_name]["rewards"]
        results[model_name]["avg_reward"] = sum(rewards) / len(rewards)
        results[model_name]["std_reward"] = (
            sum((r - results[model_name]["avg_reward"])**2 for r in rewards) / len(rewards)
        ) ** 0.5

    # Print comparison
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Base  Avg Reward: {results['base']['avg_reward']:.3f} ± {results['base']['std_reward']:.3f}")
    print(f"SFT   Avg Reward: {results['sft']['avg_reward']:.3f} ± {results['sft']['std_reward']:.3f}")
    print(f"PPO   Avg Reward: {results['ppo']['avg_reward']:.3f} ± {results['ppo']['std_reward']:.3f}")
    print(f"\nImprovement Base→SFT: {results['sft']['avg_reward'] - results['base']['avg_reward']:+.3f}")
    print(f"Improvement SFT→PPO:  {results['ppo']['avg_reward'] - results['sft']['avg_reward']:+.3f}")
    print(f"Improvement Base→PPO: {results['ppo']['avg_reward'] - results['base']['avg_reward']:+.3f}")

    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump({
            "test_prompts": test_prompts,
            "base_avg": results["base"]["avg_reward"],
            "sft_avg": results["sft"]["avg_reward"],
            "ppo_avg": results["ppo"]["avg_reward"],
            "base_std": results["base"]["std_reward"],
            "sft_std": results["sft"]["std_reward"],
            "ppo_std": results["ppo"]["std_reward"],
            "improvements": {
                "base_to_sft": results["sft"]["avg_reward"] - results["base"]["avg_reward"],
                "sft_to_ppo": results["ppo"]["avg_reward"] - results["sft"]["avg_reward"],
                "base_to_ppo": results["ppo"]["avg_reward"] - results["base"]["avg_reward"],
            },
            "detailed": results,
        }, f, indent=2)

    print(f"\nResults saved to: {os.path.join(output_dir, 'eval_results.json')}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLHF pipeline stages")
    parser.add_argument("--base-path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft-path", type=str, required=True)
    parser.add_argument("--ppo-path", type=str, required=True)
    parser.add_argument("--rm-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--prompts", type=str, nargs="+", default=[
        "Write a Python function to check if a number is even.",
        "Explain recursion with a real-life example.",
        "What are transformers in deep learning?",
        "Describe the difference between a stack and a queue.",
        "How does backpropagation work in neural networks?",
    ])
    args = parser.parse_args()

    evaluate_models(args.base_path, args.sft_path, args.ppo_path, 
                   args.rm_path, args.prompts, args.output_dir)


if __name__ == "__main__":
    main()
