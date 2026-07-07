#!/usr/bin/env python3
"""Entry point for Stage 2: Reward Model."""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlhf_pipeline.utils.config import load_config, setup_output_dir
from rlhf_pipeline.trainers.rm_trainer import run_reward_model


def main():
    parser = argparse.ArgumentParser(description="Run Reward Model Training (Stage 2)")
    parser.add_argument("--config", type=str, default="configs/reward_model.yaml", help="Path to config file")
    args = parser.parse_args()


    config = load_config(args.config)
    output_dir = setup_output_dir(config)

    print(f"Output directory: {output_dir}")
    run_reward_model(config, output_dir)

    print(f"\nDone! Check results in: {output_dir}")


if __name__ == "__main__":
    main()
