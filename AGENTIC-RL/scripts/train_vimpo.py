#!/usr/bin/env python3
"""
CLI entry point for VIMPO training on agentic tasks.

Usage:
    python scripts/train_vimpo.py --config configs/grpo_multi_agent.yaml
"""

import argparse
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.multi_tool_env import MultiToolEnv
from src.rewards.verifiable_reward import VerifiableReward
from src.models.vimpo_trainer import VIMPOTrainer
from src.utils.checkpoint import CheckpointManager
from src.utils.logging_utils import ExperimentLogger

def main():
    parser = argparse.ArgumentParser(description="Train agent with VIMPO")
    parser.add_argument("--config", type=str, default="configs/grpo_multi_agent.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    

    env = MultiToolEnv(max_steps=config["env"]["max_steps"])

    reward_fn = VerifiableReward(config.get("reward"), {})

    trainer = VIMPOTrainer(config, env, reward_fn)

    logger = ExperimentLogger(
        use_wandb=config["logging"].get("use_wandb", False),
        project=config["logging"].get("project", "agentic-rl"),
        run_name=config["logging"].get("run_name", "vimpo_run"),
        config=config,
    )

    ckpt_manager = CheckpointManager(
        output_dir=config["checkpoint"]["output_dir"].replace("grpo", "vimpo"),
        keep_last_n=3,
    )

    prompts = []
    for task in env.tasks[:config["env"].get("num_tasks", 50)]:
        prompt = f"Task: {task['question']}\n\nYou have access to these tools:\n"
        prompt += "- search(query): Search for factual information\n"
        prompt += "- calculate(expression): Evaluate a mathematical expression\n"
        prompt += "- run_code(code): Execute Python code and return the result\n\n"
        prompt += "Respond in JSON format.\n"
        prompt += 'For tool calls: {"thought": "...", "tool_name": "...", "tool_input": "..."}\n'
        prompt += 'For final answer: {"thought": "...", "final_answer": "..."}'
        prompts.append(prompt)

    print(f"Training with VIMPO on {len(prompts)} tasks, group_size={config['grpo']['group_size']}")
    print("VIMPO uses per-token advantages via backward value DP (vs. flat GRPO advantages)")

    num_epochs = config["grpo"]["num_epochs"]
    save_every = config["checkpoint"].get("save_every_n_epochs", 10)

    for epoch in range(num_epochs):
        trainer.train(prompts, num_epochs=1)

        if (epoch + 1) % save_every == 0:
            ckpt_path = ckpt_manager.save(
                epoch=epoch + 1,
                model=trainer.actor,
                optimizer=trainer.optimizer,
                prefix="vimpo",
            )
            print(f"Checkpoint saved: {ckpt_path}")
    final_path = config["checkpoint"]["output_dir"].replace("grpo", "vimpo") + "/final"
    trainer.save(final_path)
    print(f"Final model saved to {final_path}")

    logger.finish()


if __name__ == "__main__":
    main()
