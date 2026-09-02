#!/usr/bin/env python3
"""
RL fine-tuning (REINFORCE + KL) for the navigation agent.

Usage:
    python scripts/train_rl.py --config configs/rl.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import torch
import random

from env import LangNavEnv
from rl_trainer import NavRLTrainer
from model_utils import load_tokenizer, load_base_model, load_adapter, save_adapter


def load_tasks(config: dict) -> list:
    rng = random.Random(config["data"].get("seed", 42))
    env = LangNavEnv(size=config["env"]["size"], max_steps=config["env"]["max_steps"], seed=config["data"].get("seed", 42))
    tasks = []
    while len(tasks) < config["data"]["num_tasks"]:
        t = env._sample_task()
        if t["objects"][t["target"]] != tuple(t["start"]):
            tasks.append(t)
    return tasks


def train_rl(config: dict):
    model_name = config["model"]["name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(model_name)

    sft_path = config["model"].get("sft_checkpoint")
    if sft_path and Path(sft_path).exists():
        print(f"Loading SFT adapter from {sft_path}")
        actor, _ = load_adapter(sft_path, model_name, is_trainable=True)
        ref_model, _ = load_adapter(sft_path, model_name, is_trainable=False)
    else:
        print("No SFT checkpoint found, starting from base model")
        actor = load_base_model(
            model_name,
            torch_dtype=config["model"].get("torch_dtype", "float32"),
            device_map=config["model"].get("device_map", "auto"),
        )
        ref_model = load_base_model(
            model_name,
            torch_dtype=config["model"].get("torch_dtype", "float32"),
            device_map=config["model"].get("device_map", "auto"),
        )

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    tasks = load_tasks(config)
    env = LangNavEnv(size=config["env"]["size"], max_steps=config["env"]["max_steps"])

    trainer = NavRLTrainer(
        actor=actor,
        ref_model=ref_model,
        tokenizer=tokenizer,
        env=env,
        tasks=tasks,
        lr=config["training"]["learning_rate"],
        kl_beta=config["training"].get("kl_beta", 0.01),
        grad_clip=config["training"].get("grad_clip", 1.0),
        max_new_tokens=config["training"].get("max_new_tokens", 64),
    )

    for epoch in range(config["training"]["num_epochs"]):
        episodes = trainer.collect_rollouts(
            num_episodes=config["training"]["episodes_per_epoch"],
            do_sample=True,
        )
        metrics = trainer.update(episodes)
        print(
            f"Epoch {epoch+1}/{config['training']['num_epochs']} | "
            f"loss={metrics['loss']:.4f} | avg_reward={metrics['avg_reward']:.3f} | "
            f"success={metrics['success_rate']:.2%}"
        )

    output_dir = config["output"]["dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_adapter(actor, tokenizer, output_dir)
    print(f"Saved RL adapter to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rl.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_rl(config)
