"""
Supervised Fine-Tuning (SFT) Trainer with LoRA.

Prepares the base model for RLHF by training on high-quality trajectories.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from typing import List, Dict, Optional
import json


class TrajectoryDataset(Dataset):
    """Dataset of (prompt, response) pairs for SFT."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        response = item["response"]

        # Format as chat / instruction
        text = f"{prompt}\n{response}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: mask prompt tokens with -100
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Only compute loss on response tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SFTTrainer:
    """Simple SFT trainer with LoRA."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        model_name = config["model"]["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
        dtype = getattr(torch, config["model"].get("torch_dtype", "float32"))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    
        if config["training"].get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
    
        lora_cfg = config["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
        self.model.to(self.device)
    
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
        )
    
        self.output_dir = config["training"]["output_dir"]

    def train(self, dataset: TrajectoryDataset):
        """Run SFT training."""
        cfg = self.config["training"]
        dataloader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
        )

        self.model.train()
        global_step = 0

        for epoch in range(cfg["num_epochs"]):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch+1}/{cfg['num_epochs']}")

            for batch_idx, batch in enumerate(pbar):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                accum_steps = cfg.get("gradient_accumulation_steps", 1)
                if accum_steps > 1:
                    loss = loss / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    global_step += 1

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Save final model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model saved to {self.output_dir}")

    @staticmethod
    def load_trajectories(path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load trajectory data from JSONL."""
        data = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data.append(json.loads(line))
        return data
