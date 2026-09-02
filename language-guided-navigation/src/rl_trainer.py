"""
REINFORCE-style fine-tuning for the navigation agent.

After SFT warm-start, this collects rollouts and updates the LoRA policy
with a KL penalty against a frozen reference model.
"""

import random
import torch
from typing import List, Dict

from env import LangNavEnv, reward_from_trajectory
from model_utils import generate_action, compute_log_probs


class NavRLTrainer:
    def __init__(
        self,
        actor,
        ref_model,
        tokenizer,
        env: LangNavEnv,
        tasks: List[Dict],
        lr: float = 1e-5,
        kl_beta: float = 0.01,
        grad_clip: float = 1.0,
        max_new_tokens: int = 64,
    ):
        self.actor = actor
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.env = env
        self.tasks = tasks
        self.optimizer = torch.optim.AdamW(actor.parameters(), lr=lr)
        self.kl_beta = kl_beta
        self.grad_clip = grad_clip
        self.max_new_tokens = max_new_tokens

    def sample_rollout(self, task: Dict, do_sample: bool = True):
        """Run one episode and record transitions."""
        obs = self.env.reset(task)
        transitions = []
        done = False
        while not done and self.env.step_count < self.env.max_steps:
            response_text = generate_action(
                self.actor,
                self.tokenizer,
                obs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
            )
            next_obs, info, done = self.env.step(response_text)
            transitions.append({
                "observation": obs,
                "action": response_text,
                "info": info,
            })
            obs = next_obs

        final_answer = ""
        if transitions and transitions[-1]["info"].get("parsed"):
            parsed = transitions[-1]["info"]["parsed"]
            if isinstance(parsed, dict) and "final_answer" in parsed:
                final_answer = str(parsed["final_answer"])

        trajectory = {
            "actions": [t["info"].get("parsed") for t in transitions],
            "infos": [t["info"] for t in transitions],
            "final_answer": final_answer,
        }
        reward_dict = reward_from_trajectory(trajectory, self.env)
        return {
            "transitions": transitions,
            "reward": reward_dict["total_reward"],
            "answer_reward": reward_dict["answer_accuracy_reward"],
            "final_answer": final_answer,
        }

    def collect_rollouts(self, num_episodes: int, do_sample: bool = True) -> List[Dict]:
        episodes = []
        for _ in range(num_episodes):
            task = random.choice(self.tasks)
            ep = self.sample_rollout(task, do_sample=do_sample)
            episodes.append(ep)
        return episodes

    def update(self, episodes: List[Dict]):
        """One policy-gradient update with mean baseline and KL penalty."""
        rewards = torch.tensor([ep["reward"] for ep in episodes], dtype=torch.float32)
        baseline = rewards.mean()

        total_loss = 0.0
        num_tokens = 0

        self.actor.train()
        for ep in episodes:
            advantage = ep["reward"] - baseline.item()
            for trans in ep["transitions"]:
                obs = trans["observation"]
                action = trans["action"]
                actor_lp = compute_log_probs(self.actor, self.tokenizer, obs, action)
                with torch.no_grad():
                    ref_lp = compute_log_probs(self.ref_model, self.tokenizer, obs, action)

                kl = actor_lp - ref_lp
                loss = -torch.mean(actor_lp) * advantage + self.kl_beta * torch.mean(kl)
                total_loss += loss
                num_tokens += actor_lp.numel()

        total_loss = total_loss / max(1, num_tokens)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "avg_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "success_rate": (rewards > 0.5).float().mean().item(),
        }

    def save(self, path: str):
        self.actor.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
