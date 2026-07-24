#!/usr/bin/env python3
"""Generate plots from training results."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_ppo_comparison(ppo_comparison_path, save_path):
    """Plot before/after reward comparison."""
    with open(ppo_comparison_path) as f:
        data = json.load(f)

    avg_before = data.get("avg_before_reward", 0)
    avg_after = data.get("avg_after_reward", 0)
    improvement = data.get("improvement", 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Before PPO (SFT)", "After PPO (Aligned)"]
    values = [avg_before, avg_after]
    colors = ["#ff6b6b", "#51cf66"]

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.set_title(f"PPO Alignment: {avg_before:.3f} → {avg_after:.3f} ({improvement:+.3f})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add improvement arrow
    ax.annotate('', xy=(1, avg_after), xytext=(1, avg_before),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_reward_breakdown(ppo_comparison_path, save_path):
    """Plot per-prompt reward improvement."""
    with open(ppo_comparison_path) as f:
        data = json.load(f)

    comparisons = data.get("comparisons", [])
    if not comparisons:
        print("No comparison data found")
        return

    prompts = [c["prompt"][:40] + "..." for c in comparisons]
    before = [c["before_reward"] for c in comparisons]
    after = [c["after_reward"] for c in comparisons]

    x = np.arange(len(prompts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, before, width, label='Before PPO', color='#ff6b6b', alpha=0.8)
    bars2 = ax.bar(x + width/2, after, width, label='After PPO', color='#51cf66', alpha=0.8)

    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Per-Prompt Reward: Before vs After PPO', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_sft_improvement(sft_results_path, save_path):
    """Visualize SFT before/after responses."""
    with open(sft_results_path) as f:
        data = json.load(f)

    prompts = data.get("test_prompts", [])
    before = data.get("before_responses", [])
    after = data.get("after_responses", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (prompt, b, a) in enumerate(zip(prompts[:4], before[:4], after[:4])):
        ax = axes[idx]
        ax.axis('off')

        # Truncate for display
        b_short = b[:200] + "..." if len(b) > 200 else b
        a_short = a[:200] + "..." if len(a) > 200 else a

        text = f"Q: {prompt[:60]}...\n\nBEFORE (SFT):\n{b_short}\n\nAFTER (PPO):\n{a_short}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title(f"Example {idx+1}", fontsize=10, fontweight='bold')

    fig.suptitle("SFT: Before vs After Fine-Tuning", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)

    results_dir = Path("outputs")

    # Find latest results
    ppo_dirs = sorted(results_dir.glob("*_ppo_*"))
    sft_dirs = sorted(results_dir.glob("*_sft_*"))

    if ppo_dirs:
        ppo_dir = ppo_dirs[-1]
        if (ppo_dir / "ppo_comparison.json").exists():
            plot_ppo_comparison(ppo_dir / "ppo_comparison.json", output_dir / "ppo_comparison.png")
            plot_reward_breakdown(ppo_dir / "ppo_comparison.json", output_dir / "ppo_reward_breakdown.png")

    if sft_dirs:
        sft_dir = sft_dirs[-1]
        if (sft_dir / "sft_results.json").exists():
            plot_sft_improvement(sft_dir / "sft_results.json", output_dir / "sft_examples.png")

    print(f"\nAll plots saved to {output_dir}/")
    print("Files:")
    for f in output_dir.glob("*.png"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
