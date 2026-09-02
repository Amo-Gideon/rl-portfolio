"""
GRPO (Group Relative Policy Optimization) for verifiable math tasks.

Follows the DeepSeek-R1 recipe:
  - sample a group of responses per prompt
  - compute group-relative advantages (reward - mean) / std
  - clipped surrogate PPO objective per token
  - per-token KL penalty against a frozen reference model
  - no critic network
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Dict

from tasks import reward


class GRPOTrainer:
    def __init__(
        self,
        model_name: str,
        lora_config: dict,
        group_size: int = 8,
        kl_beta: float = 0.01,
        epsilon: float = 0.2,
        lr: float = 1e-5,
        max_prompt_len: int = 128,
        max_response_len: int = 256,
        torch_dtype: str = "float32",
        device_map: str = "auto",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = getattr(torch, torch_dtype)
        self.actor = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config["target_modules"],
            task_type=TaskType.CAUSAL_LM,
        )
        self.actor = get_peft_model(self.actor, peft_config)

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.group_size = group_size
        self.kl_beta = kl_beta
        self.epsilon = epsilon
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len

    def _chat_prompt(self, task_prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a careful math solver."},
            {"role": "user", "content": task_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_group(self, task_prompt: str) -> List[Dict]:
        prompt = self._chat_prompt(task_prompt)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_prompt_len
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        group = []
        for _ in range(self.group_size):
            with torch.no_grad():
                outputs = self.actor.generate(
                    **inputs,
                    max_new_tokens=self.max_response_len,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            full_ids = outputs[0]
            response_ids = full_ids[prompt_len:]
            text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            group.append({
                "text": text,
                "full_ids": full_ids.cpu(),
                "prompt_len": prompt_len,
            })
        return group

    def compute_log_probs(self, model, full_ids, prompt_len):
        full_ids = full_ids.unsqueeze(0).to(self.device)
        logits = model(input_ids=full_ids).logits[:, :-1, :]
        targets = full_ids[:, 1:]
        response_logits = logits[0, prompt_len - 1:, :]
        response_targets = targets[0, prompt_len - 1:]
        log_probs = F.log_softmax(response_logits, dim=-1)
        return log_probs.gather(dim=-1, index=response_targets.unsqueeze(-1)).squeeze(-1)

    def train_step(self, task: Dict) -> Dict:
        group = self.generate_group(task["prompt"])

        rewards = [reward(g["text"], task["answer"])["total_reward"] for g in group]
        r_tensor = torch.tensor(rewards, dtype=torch.float32)
        std = r_tensor.std()
        advantages = ((r_tensor - r_tensor.mean()) / (std + 1e-8)).tolist()

        with torch.no_grad():
            old_lps = [self.compute_log_probs(self.actor, g["full_ids"], g["prompt_len"]) for g in group]
            ref_lps = [self.compute_log_probs(self.ref_model, g["full_ids"], g["prompt_len"]) for g in group]

        total_loss = 0.0
        for i, g in enumerate(group):
            actor_lp = self.compute_log_probs(self.actor, g["full_ids"], g["prompt_len"])
            ratio = torch.exp(actor_lp - old_lps[i])
            adv = advantages[i]

            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
            surrogate = torch.min(unclipped, clipped)

            kl = actor_lp - ref_lps[i]
            loss = -torch.mean(surrogate) + self.kl_beta * torch.mean(kl)
            total_loss = total_loss + loss

        total_loss = total_loss / self.group_size
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "pass_rate": (r_tensor > 0.5).float().mean().item(),
        }

    def save(self, path: str):
        self.actor.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
