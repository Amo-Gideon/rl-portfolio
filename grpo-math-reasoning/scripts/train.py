#!/usr/bin/env python3
"""
Train GRPO on verifiable math tasks.

Usage:
    python scripts/train.py --config configs/grpo.yaml
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from grpo import GRPOTrainer
from tasks import generate_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    tasks = generate_tasks(
        num_tasks=config["data"]["num_tasks"],
        seed=config["data"]["seed"],
    )

    trainer = GRPOTrainer(
        model_name=config["model"]["name"],
        lora_config=config["lora"],
        group_size=config["grpo"]["group_size"],
        kl_beta=config["grpo"]["kl_beta"],
        epsilon=config["grpo"]["epsilon"],
        lr=config["grpo"]["lr"],
        max_prompt_len=config["grpo"]["max_prompt_length"],
        max_response_len=config["grpo"]["max_response_length"],
        torch_dtype=config["model"].get("torch_dtype", "float32"),
        device_map=config["model"].get("device_map", "auto"),
    )
    trainer.actor.print_trainable_parameters()

    rng = random.Random(config["data"]["seed"])
    num_epochs = config["training"]["num_epochs"]
    steps_per_epoch = config["training"]["steps_per_epoch"]

    best_pass = -1.0
    for epoch in range(num_epochs):
        metrics = {"loss": [], "avg_reward": [], "pass_rate": []}
        for step in range(steps_per_epoch):
            task = rng.choice(tasks)
            m = trainer.train_step(task)
            for k in metrics:
                metrics[k].append(m[k])
            if (step + 1) % 5 == 0:
                print(
                    f"Ep {epoch+1}/{num_epochs} step {step+1}/{steps_per_epoch} | "
                    f"loss {m['loss']:.4f} | reward {m['avg_reward']:.3f} | "
                    f"pass {m['pass_rate']:.2f}"
                )

        avg_pass = sum(metrics["pass_rate"]) / len(metrics["pass_rate"])
        print(
            f"== Epoch {epoch+1}: avg reward {sum(metrics['avg_reward'])/len(metrics['avg_reward']):.3f} | "
            f"avg pass {avg_pass:.2f} =="
        )
        if avg_pass > best_pass:
            best_pass = avg_pass
            trainer.save(config["output"]["dir"] + "/best")
            print(f"Saved new best to {config['output']['dir']}/best")

    trainer.save(config["output"]["dir"] + "/final")
    print(f"Saved final to {config['output']['dir']}/final")


if __name__ == "__main__":
    main()
