#!/usr/bin/env python3
"""
Visualize one navigation episode as a sequence of grid images.

Usage:
    python scripts/visualize.py --checkpoint outputs/rl --output_dir assets/episode_viz
    python scripts/visualize.py --checkpoint outputs/rl --output_dir assets/episode_viz --find_success
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from env import LangNavEnv
from model_utils import load_adapter, load_base_model, load_tokenizer, generate_action


def render_frame(env: LangNavEnv, step_idx: int, action_text: str, output_path: Path):
    """Render one episode step to a PNG file."""
    task = env.current_task
    size = env.size
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.grid(True)
    ax.set_title(f"Step {step_idx} | Facing {env.facing} | Action: {action_text[:40]}")

    # Draw objects
    for name, (x, y) in task["objects"].items():
        color = "lightblue"
        if name == task["target"]:
            color = "lightgreen"
        rect = plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, color=color, ec="black")
        ax.add_patch(rect)
        ax.text(x, y, name.replace(" ", "\n"), ha="center", va="center", fontsize=7)

    # Draw agent
    x, y = env.position
    arrow = {"N": (0, 0.3), "E": (0.3, 0), "S": (0, -0.3), "W": (-0.3, 0)}[env.facing]
    ax.arrow(x, y, arrow[0], arrow[1], head_width=0.15, head_length=0.15, fc="red", ec="red")
    ax.text(x, y - 0.25, "agent", ha="center", va="top", fontsize=8, color="red")

    # Instruction text below plot
    fig.text(0.5, 0.02, task["instruction"], ha="center", fontsize=9, wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def run_episode(model, tokenizer, env: LangNavEnv, task, out_path: Path):
    """Run one episode and save frames. Returns (success, num_frames)."""
    obs = env.reset(task)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    step_idx = 0
    render_frame(env, step_idx, "START", out_path / f"step_{step_idx:02d}.png")

    done = False
    info = {}
    while not done and env.step_count < env.max_steps:
        response = generate_action(model, tokenizer, obs, do_sample=False)
        obs, info, done = env.step(response)
        step_idx += 1
        render_frame(env, step_idx, response, out_path / f"step_{step_idx:02d}.png")

    final_answer = ""
    if info.get("parsed") and isinstance(info["parsed"], dict) and "final_answer" in info["parsed"]:
        final_answer = info["parsed"]["final_answer"]

    success = final_answer.strip().lower() == task["answer"].strip().lower()
    return success, step_idx + 1, final_answer


def visualize(checkpoint: str, base_model_name: str, output_dir: str, size: int = 5, seed: int = 0, find_success: bool = False):
    if Path(checkpoint).exists() and (Path(checkpoint) / "adapter_config.json").exists():
        model, tokenizer = load_adapter(checkpoint, base_model_name)
    else:
        tokenizer = load_tokenizer(base_model_name)
        model = load_base_model(base_model_name)
    model.eval()

    out_path = Path(output_dir)

    env = LangNavEnv(size=size, max_steps=15, seed=seed)
    task = env._sample_task()
    success, frames, final_answer = run_episode(model, tokenizer, env, task, out_path)

    if find_success:
        attempts = 1
        while not success and attempts < 50:
            env = LangNavEnv(size=size, max_steps=15, seed=seed + attempts)
            task = env._sample_task()
            success, frames, final_answer = run_episode(model, tokenizer, env, task, out_path)
            attempts += 1
        print(f"Found successful episode after {attempts} attempt(s)")

    print(f"Saved {frames} frames to {out_path}")
    print(f"Task: {task['instruction']}")
    print(f"Final answer: {final_answer} | Ground truth: {task['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/rl")
    parser.add_argument("--base_model", type=str, default="./models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="assets/episode_viz")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--find_success", action="store_true")
    args = parser.parse_args()

    visualize(args.checkpoint, args.base_model, args.output_dir, args.size, args.seed, args.find_success)
