"""Reward model implementation."""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RewardModel(nn.Module):
    """
    Reward model based on a pretrained language model with classification head.
    Outputs a scalar reward score.
    Trained with Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected)).
    """

    def __init__(self, base_model_name: str, hidden_size: int, freeze_base: bool = False):
        super().__init__()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
        )
        if freeze_base:
            for param in self.base_model.base_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits.squeeze(-1)


class NeuralRewardModel:
    """Wrapper for trained neural reward model with tokenizer."""

    def __init__(self, model_path: str, tokenizer: AutoTokenizer, device: str = "cuda"):
        self.tokenizer = tokenizer
        self.device = device

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.model.eval()

    def score(self, prompt: str, response: str) -> float:
        """Score a prompt-response pair."""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            score = self.model(**enc).logits.squeeze().item()

        return score