"""
Group Relative Policy Optimization (GRPO) Trainer for Agentic RL.

Implements:
  - Group sampling: G responses per prompt
  - Per-token log probability computation
  - Group-relative advantage normalization
  - Per-token KL divergence penalty (not sequence-level)
  - Clipped surrogate PPO objective
  - No critic model (GRPO removes the value network)

References:
  - DeepSeek-R1: GRPO for reasoning (2025)
  - Dr. GRPO: Token-level penalization to prevent collapse
  - VIMPO: Per-token value recurrence (future extension point)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import copy
import os

class GRPOTrainer:
    """
    GRPO trainer for fine-tuning LLMs on agentic tasks with verifiable rewards
    """

    def __init__(self, config:Dict, env, reward_fn):
        self.config = config
        self.env = env
        self.reward_fn = reward_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = config["model"]["name"]
        sft_ckpt = config["model"].get("sft_checkpoint", None)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_path = sft_ckpt if sft_ckpt and os.path.exists(sft_ckpt) else model_name
        
        self.actor = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=getattr(torch, config["model"].get("torch_dtype", "float32")),
            device_map=config["model"].get("device_map", "auto"),
            trust_remote_code=True,
        )

        lora_cfg = config["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            task_type=TaskType.CAUSAL_LM,
        )
        self.actor = get_peft_model(self.actor, peft_config)

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=getattr(torch, config["model"].get("torch_dtype", "float32")),
            device_map=config["model"].get("device_map", "auto"),
            trust_remote_code=True,
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=config["grpo"]["actor_lr"],
        )

        self.group_size = config["grpo"]["group_size"]
        self.kl_beta = config["grpo"]["kl_beta"]
        self.epsilon = config["grpo"]["epsilon"]
        self.grad_clip_norm = config["grpo"]["grad_clip_norm"]
        self.entropy_coef = config["grpo"].get("entropy_coef", 0.01)
        self.max_prompt_len = config["grpo"]["max_prompt_length"]
        self.max_response_len = config["grpo"]["max_response_length"]
    
    def generate_group(self, prompt: str, group_size: int) -> List[Dict]:
        """
        Generate a group of responses for a single prompt.

        Returns list of dicts with:
          - text: generated text
          - token_ids: response token IDs
          - prompt_ids: prompt token IDs
          - full_ids: prompt + response token IDs
        """
        prompt_encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_len,
        ).to(self.device)

        prompt_ids = prompt_encoding["input_ids"]
        prompt_len = prompt_ids.shape[1]

        group = []
        for _ in range(group_size):
            with torch.no_grad():
                outputs = self.actor.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=self.max_response_len,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            full_ids = outputs[0]  # [seq_len]
            response_ids = full_ids[prompt_len:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            group.append({
                "text": response_text,
                "token_ids": response_ids.cpu(),
                "prompt_ids": prompt_ids[0].cpu(),
                "full_ids": full_ids.cpu(),
                "prompt_len": prompt_len,
            })

        return group
    
    def compute_log_probs(self, model, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """
        Compute per-token log probabilities for the response portion.

        Args:
            model: LLM model
            full_ids: [batch, seq_len] full sequence (prompt + response)
            prompt_len: int, length of prompt portion

        Returns:
            log_probs: [batch, response_len] per-token log probs

        """
        outputs = model(input_ids=full_ids.unsqueeze(0).to(self.device))
        logits = outputs.logits


        logits = logits[:, :-1, :]
        target_ids = full_ids[1:].to(self.device)

        response_logits = logits[0, prompt_len-1:,:]
        response_targets = target_ids[prompt_len-1:]


        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=response_targets.unsqueeze(-1)).squeeze(-1)

        return token_log_probs
    

    def compute_per_token_kl(self, actor_log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token KL divergence: KL(P_actor || P_ref) at each position.

        Uses the approximation: KL ≈ log P_actor - log P_ref (sample estimate)
        """
        return actor_log_probs - ref_log_probs

    def compute_group_advantages(self, rewards: List[float]) -> List[float]:
        """
        Compute group-relative advantages.

        A_i = (r_i - mean(r)) / (std(r) + epsilon)
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8

        advantages = (rewards_tensor - mean_reward) / std_reward
        return advantages.tolist()
    
    def run_episode(self, prompt: str, response_text: str) -> Tuple[float, Dict]:
        """
        Run a full episode in the environment with a generated response.

        Returns total reward and trajectory info.
        """
        obs = self.env.reset()
        done = False
        step_count = 0
        trajectory = {
            "actions": [],
            "infos": [],
            "observations": [obs],
            "final_answer": "",
        }

        parsed = self.env._parse_action(response_text)
        trajectory["actions"].append(parsed)

        if parsed and "final_answer" in parsed:
            trajectory["final_answer"] = str(parsed["final_answer"])
            info = {"format_valid": True, "tool_valid": False, "final_answer": trajectory["final_answer"]}
            trajectory["infos"].append(info)
        elif parsed and "tool_name" in parsed:
            obs, info, done = self.env.step(response_text)
            trajectory["infos"].append(info)
            trajectory["observations"].append(obs)
        else:
            info = {"format_valid": False, "tool_valid": False}
            trajectory["infos"].append(info)

        ground_truth = self.env.get_task_answer()
        reward_dict = self.reward_fn.compute(trajectory, ground_truth)


        return reward_dict["total_reward"], trajectory
    

    def train_step(self, prompt: str) -> Dict:
        """
        Execute one GRPO training step for a single prompt.

        Returns metrics dict.
        """
        group = self.generate_group(prompt, self.group_size)

        rewards = []
        trajectories = []
        for response in group:
            reward, traj = self.run_episode(prompt, response["text"])
            rewards.append(reward)
            trajectories.append(traj)

        advantages = self.compute_group_advantages(rewards)
        old_log_probs_list = []
        ref_log_probs_list = []
        for response in group:
            full_ids = response["full_ids"].to(self.device)
            prompt_len = response["prompt_len"]

            with torch.no_grad():
                old_lp = self.compute_log_probs(self.actor, full_ids, prompt_len)
                ref_lp = self.compute_log_probs(self.ref_model, full_ids, prompt_len)
            old_log_probs_list.append(old_lp)
            ref_log_probs_list.append(ref_lp)

        total_loss = 0.0
        total_kl = 0.0

        for i in range(self.group_size):
            full_ids = group[i]["full_ids"].to(self.device)
            prompt_len = group[i]["prompt_len"]
            actor_lp = self.compute_log_probs(self.actor, full_ids, prompt_len)
            ref_lp = ref_log_probs_list[i]
            advantage = advantages[i]

            old_lp = old_log_probs_list[i].detach()
            ratio = torch.exp(actor_lp - old_lp)

            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            surrogate = torch.min(unclipped, clipped)

            kl_per_token = self.compute_per_token_kl(actor_lp, ref_lp)
            kl_penalty = self.kl_beta * kl_per_token

            entropy = -torch.mean(actor_lp) * self.entropy_coef

            # Policy loss: negative because we maximize reward
            # Average over response tokens
            policy_loss = -torch.mean(surrogate - kl_penalty) - entropy

            total_loss += policy_loss
            total_kl += kl_per_token.mean().item()

        # Average over group
        total_loss = total_loss / self.group_size

        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "avg_reward": sum(rewards) / len(rewards),
            "avg_kl": total_kl / self.group_size,
            "advantage_std": torch.tensor(advantages).std().item(),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
        }

    def train(self, prompts: List[str], num_epochs: int):
        """
        Full GRPO training loop.

        Args:
            prompts: list of task prompts
            num_epochs: number of training epochs
        """
        self.actor.train()

        for epoch in range(num_epochs):
            epoch_metrics = {
                "loss": [], "avg_reward": [], "avg_kl": [],
                "advantage_std": [], "max_reward": [], "min_reward": [],
            }

            pbar = tqdm(enumerate(prompts), total=len(prompts), desc=f"GRPO Epoch {epoch+1}/{num_epochs}")

            for step_idx, prompt in pbar:
                metrics = self.train_step(prompt)

                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

                pbar.set_postfix({
                    "reward": f"{metrics['avg_reward']:.3f}",
                    "kl": f"{metrics['avg_kl']:.4f}",
                    "loss": f"{metrics['loss']:.4f}",
                })

            # Epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            for k, v in epoch_metrics.items():
                if v:
                    print(f"  {k}: {sum(v)/len(v):.4f}")

        print("\nTraining complete.")

    def save(self, path: str):
        """Save actor model."""
        self.actor.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        """Load actor model."""
        self.actor = AutoModelForCausalLM.from_pretrained(path)
        self.actor.to(self.device)
