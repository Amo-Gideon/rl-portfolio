#!/usr/bin/env python3
"""
Supervised fine-tuning on expert trajectories for language-guided navigation.

Usage:
    python scripts/sft.py --config configs/sft.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import generate_expert_dataset, format_sft_example
from model_utils import load_actor, load_tokenizer, save_adapter


class SFTDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        pair = self.examples[idx]
        prefix = f"{pair['observation']}\nAction: "
        text = format_sft_example(pair)

        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_ids)

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[:prefix_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_sft(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]["name"]

    tokenizer = load_tokenizer(model_name)
    model = load_actor(
        model_name,
        config["lora"],
        torch_dtype=config["model"].get("torch_dtype", "float32"),
        device_map=config["model"].get("device_map", "auto"),
    )
    model.print_trainable_parameters()

    print("Generating expert dataset...")
    examples = generate_expert_dataset(
        num_tasks=config["data"].get("num_tasks", 200),
        size=config["env"].get("size", 5),
        seed=config["data"].get("seed", 42),
    )
    print(f"Dataset size: {len(examples)}")

    dataset = SFTDataset(examples, tokenizer, max_length=config["training"].get("max_length", 512))
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    model.train()
    for epoch in range(config["training"]["num_epochs"]):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"SFT epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"].get("grad_clip", 1.0))
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    output_dir = config["output"]["dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_adapter(model, tokenizer, output_dir)
    print(f"Saved SFT adapter to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_sft(config)
