"""Reward model implementation."""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class RewardModel(nn.Module):
    """
    Reward model based on a pretrained language model.
    Outputs a scalar reward score using a linear value head on the last hidden state.
    Trained with Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected)).
    """

    def __init__(self, base_model_name: str, hidden_size: int, freeze_base: bool = False):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float32
        )
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden)

        # Extract last valid token's hidden state
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        last_token_hidden = last_hidden[torch.arange(batch_size), seq_lengths]

        reward = self.value_head(last_token_hidden).squeeze(-1)
        return reward


class SimpleRewardModel:
    """
    Rule-based reward model fallback for quick PPO demos without a trained RM.
    Scores responses on length, structure, tone, and relevance.
    """

    def __init__(self, tokenizer, backbone_model=None, value_head_path: str = None):
        self.tokenizer = tokenizer
        self.backbone_model = backbone_model
        self.value_head = None

        if backbone_model is not None:
            hidden_size = backbone_model.config.hidden_size
            self.value_head = nn.Linear(hidden_size, 1)
            if value_head_path and __import__("os").path.exists(value_head_path):
                self.value_head.load_state_dict(
                    torch.load(value_head_path, map_location="cpu")
                )

    def score(self, prompt: str, response: str) -> float:
        if self.backbone_model is not None and self.value_head is not None:
            return self._neural_score(prompt, response)
        return self._rule_based_score(prompt, response)

    def _neural_score(self, prompt, response):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = self.tokenizer(text, truncation=True, max_length=256, padding=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.backbone_model(**enc, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            seq_len = enc["attention_mask"].sum(dim=1) - 1
            last_token = last_hidden[0, seq_len[0]]
            neural_reward = self.value_head(last_token).item()

        rule_reward = self._rule_based_score(prompt, response)
        return 0.5 * neural_reward + 0.5 * rule_reward

    def _rule_based_score(self, prompt, response):
        score = 0.0
        length = len(response)

        if length < 10:
            score -= 2.0
        elif length < 30:
            score -= 0.5
        elif 50 <= length <= 300:
            score += 1.5
        elif length > 500:
            score -= 0.5

        if any(m in response for m in ["1.", "2.", "3.", "(1)", "(2)"]):
            score += 1.0
        if "```" in response:
            score += 1.0
        if any(m in response for m in [":\n", "step", "method"]):
            score += 0.5

        positive = ["please", "suggest", "can help", "below", "of course", "sure"]
        negative = ["not my problem", "search yourself", "whatever", "don't want"]
        for w in positive:
            if w in response:
                score += 0.3
        for w in negative:
            if w in response:
                score -= 1.0

        p_words = set(prompt.replace("?", "").replace(",", "").split())
        r_words = set(response.replace(",", "").replace(".", "").split())
        overlap = len(p_words & r_words)
        if overlap > 0:
            score += min(overlap * 0.2, 1.0)

        return score
