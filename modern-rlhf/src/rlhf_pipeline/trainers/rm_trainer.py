"""Reward model training stage."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from rlhf_pipeline.data.preference_data import load_preference_data
from rlhf_pipeline.utils.config import set_seed
from rlhf_pipeline.utils.logging_utils import setup_logger, WandbTracker
from rlhf_pipeline.utils.checkpoint import save_json, create_symlink
from rlhf_pipeline.utils.metrics import (
    plot_reward_distribution,
    compute_pairwise_accuracy,
    compute_margin,
)


def train_reward_model_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        r_chosen = model(chosen_ids, chosen_mask).logits.squeeze(-1)
        r_rejected = model(rejected_ids, rejected_mask).logits.squeeze(-1)

        loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-8).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (r_chosen > r_rejected).sum().item()
        total += r_chosen.shape[0]

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate_reward_model(model, test_data, tokenizer, device, max_length=512):
    """Evaluate reward model on test pairs."""
    model.eval()
    chosen_scores = []
    rejected_scores = []

    with torch.no_grad():
        for pair in test_data:
            for key in ["chosen", "rejected"]:
                messages = [
                    {"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": pair[key]},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                score = model(**enc).logits.squeeze().item()

                if key == "chosen":
                    chosen_scores.append(score)
                else:
                    rejected_scores.append(score)

    accuracy = compute_pairwise_accuracy(chosen_scores, rejected_scores)
    margin = compute_margin(chosen_scores, rejected_scores)

    return chosen_scores, rejected_scores, accuracy, margin


def run_reward_model(config, output_dir: str):
    """Train reward model with LoRA."""
    logger = setup_logger("RM", log_file=os.path.join(output_dir, "rm.log"))
    logger.info("=" * 60)
    logger.info("Stage 2: Reward Model Training")
    logger.info("=" * 60)

    set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tracker = WandbTracker(
        project=config.training.report_to if config.training.report_to != "none" else "none",
        config={
            "stage": "reward_model",
            "model": config.model.name,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.num_train_epochs,
        },
        run_name=f"{config.experiment.name}_rm",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading base model: {config.model.name}")
    torch_dtype = getattr(torch, config.model.torch_dtype)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name,
        num_labels=1,
        torch_dtype=torch_dtype,
        trust_remote_code=config.model.trust_remote_code,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if config.model.use_lora:
        logger.info("Applying LoRA to reward model...")
        lora_config = LoraConfig(
            r=config.model.lora.r,
            lora_alpha=config.model.lora.lora_alpha,
            lora_dropout=config.model.lora.lora_dropout,
            target_modules=config.model.lora.target_modules,
            bias=config.model.lora.bias,
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    logger.info(f"Loading preference data from: {config.data.source}")
    train_dataset, test_data = load_preference_data(config, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    logger.info(f"Train: {len(train_dataset)} samples, Test: {len(test_data)} pairs")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=0.01,
    )

    logger.info("Training reward model...")
    best_accuracy = 0.0
    train_losses = []

    for epoch in range(config.training.num_train_epochs):
        avg_loss, train_acc = train_reward_model_epoch(
            model, train_loader, optimizer, device, epoch
        )
        train_losses.append(avg_loss)

        logger.info(f"Epoch {epoch+1}/{config.training.num_train_epochs} | "
                   f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2%}")
        tracker.log({
            "rm/epoch": epoch + 1,
            "rm/train_loss": avg_loss,
            "rm/train_accuracy": train_acc,
        })

        if test_data:
            chosen_scores, rejected_scores, accuracy, margin = evaluate_reward_model(
                model, test_data, tokenizer, device, config.data.max_length
            )
            logger.info(f"  Test Accuracy: {accuracy:.2%} | Margin: {margin:.4f}")
            tracker.log({
                "rm/test_accuracy": accuracy,
                "rm/test_margin": margin,
            })

            if accuracy > best_accuracy:
                best_accuracy = accuracy

    model_path = os.path.join(output_dir, "reward_model")
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Reward model saved to: {model_path}")

    if test_data:
        chosen_scores, rejected_scores, accuracy, margin = evaluate_reward_model(
            model, test_data, tokenizer, device, config.data.max_length
        )

        plot_path = os.path.join(output_dir, "reward_distribution.png")
        plot_reward_distribution(chosen_scores, rejected_scores, plot_path)
        logger.info(f"Distribution plot saved to: {plot_path}")

        save_json({
            "test_accuracy": accuracy,
            "test_margin": margin,
            "best_accuracy": best_accuracy,
            "train_losses": train_losses,
            "test_chosen_scores": chosen_scores,
            "test_rejected_scores": rejected_scores,
        }, os.path.join(output_dir, "rm_results.json"))

    create_symlink(output_dir, config.experiment.name, config.experiment.output_dir)

    tracker.finish()
    logger.info("Reward model training complete!")
    return model_path