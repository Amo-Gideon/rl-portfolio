#!/usr/bin/env python3
"""
CLI entry point for Supervised Fine-Tuning (SFT).

Usage:
    python scripts/run_sft.py --config configs/sft.yaml
"""

import argparse
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sft_trainer import SFTTrainer, TrajectoryDataset


def main():
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument("--config", type=str, default="configs/sft.ymal", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    

    data_path = config["data"]["trajectory_file"]
    max_samples = config["data"].get("max_samples", None)


    if not Path(data_path).exists():
        print(f"Trajectory file {data_path} not found. Generating synthetic SFT data...")
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)


        import json
        synthetic = [
            {
                "prompt": "Task: What is the population of France divided by 2?\nRespond in JSON format.",
                "response": '{"thought": "I will search for the population of France and then divide by 2.", "tool_name": "search", "tool_input": "population of France"}'
            },
            {
                "prompt": "Task: Calculate the area of a circle with radius 5.\nRespond in JSON format.",
                "response": '{"thought": "I will use the calculate tool with pi * r^2.", "tool_name": "calculate", "tool_input": "3.14159 * 5 * 5"}'
            },
            {
                "prompt": "Task: What is the capital of Japan?\nRespond in JSON format.",
                "response": '{"thought": "I will search for the capital of Japan.", "tool_name": "search", "tool_input": "capital of Japan"}'
            },

        ] * 50

        with open(data_path, "w") as f:
            for item in synthetic:
                f.write(json.dumps(item) + "\n")
        print(f"Generated {len(synthetic)} synthetic trajectories.")
    
    data = SFTTrainer.load_trajectories(data_path, max_samples)
    print(f"Loaded {len(data)} training examples.")

    trainer = SFTTrainer(config)
    dataset = TrajectoryDataset(data, trainer.tokenizer, config["training"]["max_seq_length"])
    trainer.train(dataset)



if __name__ == "__main__":
    main()