#!/usr/bin/env python3
"""Entry point for Stage 1: SFT."""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlhf_pipeline.utils.config import load_config, setup_output_dir
from rlhf_pipeline.trainers.sft_trainer import run_sft


def main():
    parser = argparse.ArgumentParser(description="Run Supervised Fine-Tuning (Stage 1)")
    parser.add_argument("--config", type=str, default="configs/sft.yaml", 
                       help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = setup_output_dir(config)

    print(f"Output directory: {output_dir}")
    run_sft(config, output_dir)
    print(f"\nDone! Check results in: {output_dir}")


if __name__ == "__main__":
    main()