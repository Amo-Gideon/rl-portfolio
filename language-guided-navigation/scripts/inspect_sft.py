#!/usr/bin/env python3
"""
Inspect what the SFT model outputs on a few training examples.

Usage:
    python scripts/inspect_sft.py --checkpoint outputs/sft --num_examples 5 > inspect_log.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import generate_expert_dataset, format_sft_example
from model_utils import load_adapter, load_base_model, load_tokenizer, generate_action


def inspect(checkpoint: str, base_model_name: str, num_examples: int = 5):
    if Path(checkpoint).exists() and (Path(checkpoint) / "adapter_config.json").exists():
        model, tokenizer = load_adapter(checkpoint, base_model_name)
    else:
        tokenizer = load_tokenizer(base_model_name)
        model = load_base_model(base_model_name)
    model.eval()

    examples = generate_expert_dataset(num_tasks=20, size=5, seed=42)
    print(f"Inspecting {num_examples} examples from checkpoint: {checkpoint}\n")
    for i, ex in enumerate(examples[:num_examples]):
        obs = ex["observation"]
        gt_action = ex["action"]
        pred_action = generate_action(model, tokenizer, obs, do_sample=False)
        print(f"--- Example {i+1} ---")
        print(f"Observation:\n{obs}")
        print(f"Ground truth action: {gt_action}")
        print(f"Predicted action:    {pred_action}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/sft")
    parser.add_argument("--base_model", type=str, default="./models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num_examples", type=int, default=5)
    args = parser.parse_args()

    inspect(args.checkpoint, args.base_model, args.num_examples)
