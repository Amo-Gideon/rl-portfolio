"""Reward model training stage."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from rlhf_pipeline.data.preference_data import load_preference_data
from rlhf_pipeline.models.reward_model import RewardModel
from rlhf_pipeline.utils.logging_utils import setup_logger
from rlhf_pipeline.utils.checkpoint import save_json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_reward_model(model, dataloader, optimizer, device, epochs=5):
    model.train()
    all_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            r_chosen = model(chosen_ids, chosen_mask)
            r_rejected = model(rejected_ids, rejected_mask)

            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (r_chosen > r_rejected).sum().item()
            total += r_chosen.shape[0]

        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0
        all_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")

    return all_losses

def evaluate_reward_model(model, pairs, tokenizer, device, max_length=256):
    model.eval()
    correct = 0
    chosen_scores = []
    rejected_scores = []

    with torch.no_grad():
        for pair in pairs:
            for key in ["chosen", "rejected"]:
                messages = [
                    {"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": pair[key]},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                enc = tokenizer(text, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
                score = model(enc["input_ids"].to(device), enc["attention_mask"].to(device)).item()
                if key == "chosen":
                    chosen_scores.append(score)
                else:
                    rejected_scores.append(score)

            if chosen_scores[-1] > rejected_scores[-1]:
                correct += 1

    accuracy = correct / len(pairs)
    return chosen_scores, rejected_scores, accuracy


def plot_distributions(chosen_scores, rejected_scores, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(chosen_scores))
    axes[0].scatter(x, chosen_scores, color="green", label="Chosen", alpha=0.7, s=60)
    axes[0].scatter(x, rejected_scores, color="red", label="Rejected", alpha=0.7, s=60)
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Reward Score")
    axes[0].set_title("Chosen vs Rejected")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(chosen_scores, bins=10, alpha=0.6, color="green", label="Chosen")
    axes[1].hist(rejected_scores, bins=10, alpha=0.6, color="red", label="Rejected")
    axes[1].set_xlabel("Reward Score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_reward_model(config, output_dir: str):
    logger = setup_logger("RM", log_file=os.path.join(output_dir, "rm.log"))
    logger.info("=" * 60)
    logger.info("Stage 2: Reward Model Training")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name, torch_dtype=getattr(torch, config.model.torch_dtype)
    )
    hidden_size = config.model.hidden_size or base_model.config.hidden_size

    reward_model = RewardModel(config.model.name, hidden_size, freeze_base=config.model.freeze_base).to(device)
    logger.info(f"Reward model built with hidden_size={hidden_size}")

    train_dataset, test_pairs = load_preference_data(config, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    logger.info(f"Train: {len(train_dataset)} samples, Test: {len(test_pairs)} pairs")

    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=config.training.learning_rate)

    logger.info("Training reward model...")
    train_losses = train_reward_model(reward_model, train_loader, optimizer, device, epochs=config.training.epochs)

    logger.info("Evaluating reward model...")
    chosen_scores, rejected_scores, accuracy = evaluate_reward_model(
        reward_model, test_pairs, tokenizer, device, config.data.max_length
    )
    logger.info(f"Test accuracy: {accuracy:.2%}")

    # Save
    value_head_path = os.path.join(output_dir, "value_head.pt")
    torch.save(reward_model.value_head.state_dict(), value_head_path)
    logger.info(f"Value head saved to: {value_head_path}")

    save_json({
        "accuracy": accuracy,
        "train_losses": train_losses,
        "test_chosen_scores": chosen_scores,
        "test_rejected_scores": rejected_scores,
    }, os.path.join(output_dir, "rm_results.json"))

    if config.evaluation.get("visualize", True):
        plot_path = os.path.join(output_dir, "reward_distribution.png")
        plot_distributions(chosen_scores, rejected_scores, plot_path)
        logger.info(f"Plot saved to: {plot_path}")

    logger.info("Reward model training complete!")
    return value_head_path
