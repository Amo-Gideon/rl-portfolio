"""Evaluation metrics and plotting utilities."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Curves"
):
    """Plot training metrics over time."""
    n_metrics = len(metrics)
    if n_metrics == 0:
        return

    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for idx, (key, values) in enumerate(metrics.items()):
        ax = axes[idx]
        ax.plot(values, "o-", markersize=3, linewidth=1.5, alpha=0.8)
        ax.set_title(key.replace("_", " ").title())
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_reward_distribution(
    chosen_scores: List[float],
    rejected_scores: List[float],
    save_path: str
):
    """Plot chosen vs rejected reward distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = range(len(chosen_scores))
    axes[0].scatter(x, chosen_scores, color="green", label="Chosen", alpha=0.7, s=60)
    axes[0].scatter(x, rejected_scores, color="red", label="Rejected", alpha=0.7, s=60)
    axes[0].axhline(y=np.mean(chosen_scores), color="green", linestyle="--", alpha=0.5)
    axes[0].axhline(y=np.mean(rejected_scores), color="red", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Reward Score")
    axes[0].set_title("Chosen vs Rejected Scores")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(chosen_scores, bins=15, alpha=0.6, color="green", label="Chosen", density=True)
    axes[1].hist(rejected_scores, bins=15, alpha=0.6, color="red", label="Rejected", density=True)
    axes[1].set_xlabel("Reward Score")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Score Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_pairwise_accuracy(chosen_scores: List[float], rejected_scores: List[float]) -> float:
    """Compute pairwise accuracy: fraction where chosen > rejected."""
    correct = sum(c > r for c, r in zip(chosen_scores, rejected_scores))
    return correct / len(chosen_scores) if chosen_scores else 0.0


def compute_margin(chosen_scores: List[float], rejected_scores: List[float]) -> float:
    """Compute average margin between chosen and rejected scores."""
    margins = [c - r for c, r in zip(chosen_scores, rejected_scores)]
    return np.mean(margins) if margins else 0.0


def save_comparison_results(
    test_prompts: List[str],
    before_responses: List[str],
    after_responses: List[str],
    before_rewards: List[float],
    after_rewards: List[float],
    output_path: str
):
    """Save before/after comparison to JSON."""
    results = []
    for i, prompt in enumerate(test_prompts):
        results.append({
            "prompt": prompt,
            "before_response": before_responses[i],
            "after_response": after_responses[i],
            "before_reward": before_rewards[i],
            "after_reward": after_rewards[i],
            "improvement": after_rewards[i] - before_rewards[i],
        })

    save_json({
        "avg_before_reward": float(np.mean(before_rewards)),
        "avg_after_reward": float(np.mean(after_rewards)),
        "improvement": float(np.mean(after_rewards) - np.mean(before_rewards)),
        "comparisons": results,
    }, output_path)


def save_json(data: dict, path: str):
    """Save a dictionary as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
