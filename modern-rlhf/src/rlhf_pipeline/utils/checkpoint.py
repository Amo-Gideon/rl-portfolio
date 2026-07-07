"""Checkpoint saving and loading utilities."""
import os
import json
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def save_model_and_tokenizer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    subfolder: str = ""
):
    """Save a model and tokenizer to disk."""
    save_path = os.path.join(output_dir, subfolder) if subfolder else output_dir
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    return save_path


def save_json(data: dict, path: str):
    """Save a dictionary as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
