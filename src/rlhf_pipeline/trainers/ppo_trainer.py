"""PPO alignment training stage."""
import os
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from rlhf_pipeline.models.reward_model import SimpleRewardModel
from rlhf_pipeline.models.reference_model import create_reference_model
from rlhf_pipeline.utils.logging_utils import setup_logger
from rlhf_pipeline.utils.checkpoint import save_json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_response(model, tokenizer, prompt, max_new_tokens=80, temperature=0.7):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=0.9
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response, outputs[0]


def compute_log_probs(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    shift_mask = attention_mask[:, 1:]
    token_log_probs = token_log_probs * shift_mask
    return token_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1)


def compute_kl_divergence(policy_model, ref_model, input_ids, attention_mask):
    with torch.no_grad():
        p_out = policy_model(input_ids=input_ids, attention_mask=attention_mask)
        p_logits = p_out.logits[:, :-1, :]
        p_log_probs = F.log_softmax(p_logits, dim=-1)
        p_probs = torch.softmax(p_logits, dim=-1)

        r_out = ref_model(input_ids=input_ids, attention_mask=attention_mask)
        r_logits = r_out.logits[:, :-1, :]
        r_log_probs = F.log_softmax(r_logits, dim=-1)

        kl_per_token = (p_probs * (p_log_probs - r_log_probs)).sum(dim=-1)
        shift_mask = attention_mask[:, 1:]
        kl = (kl_per_token * shift_mask).sum() / shift_mask.sum()
    return kl.item()


class PPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, tokenizer, config):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.kl_coef = config.ppo.kl_coef
        self.clip_range = config.ppo.clip_range
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.ppo.learning_rate)
        self.stats = {
            "rewards": [], "kl_divergences": [], "policy_losses": [],
            "total_losses": [], "response_lengths": [],
        }

    def train_step(self, prompts):
        self.policy_model.train()
        batch_rewards, batch_kl, batch_lengths = [], [], []
        all_input_ids, all_masks, all_old_log_probs = [], [], []

        for prompt in prompts:
            response, full_ids = generate_response(
                self.policy_model, self.tokenizer, prompt,
                max_new_tokens=60, temperature=0.8
            )
            input_ids = full_ids.unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)

            reward = self.reward_model.score(prompt, response)
            batch_rewards.append(reward)
            batch_lengths.append(len(response))

            kl = compute_kl_divergence(self.policy_model, self.ref_model, input_ids, attention_mask)
            batch_kl.append(kl)

            with torch.no_grad():
                old_log_prob = compute_log_probs(self.policy_model, input_ids, attention_mask)
            all_old_log_probs.append(old_log_prob)
            all_input_ids.append(input_ids)
            all_masks.append(attention_mask)

        rewards_t = torch.tensor(batch_rewards, dtype=torch.float32)
        advantages = rewards_t - rewards_t.mean()
        advantages = advantages / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        for i, (ids, mask, old_log_p) in enumerate(zip(all_input_ids, all_masks, all_old_log_probs)):
            new_log_prob = compute_log_probs(self.policy_model, ids, mask)
            ratio = torch.exp(new_log_prob - old_log_p)
            advantage = advantages[i]
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantage
            policy_loss = -torch.min(surr1, surr2)
            total_policy_loss += policy_loss

        avg_policy_loss = total_policy_loss / len(prompts)
        avg_kl = sum(batch_kl) / len(batch_kl)
        kl_penalty = self.kl_coef * avg_kl
        total_loss = avg_policy_loss + kl_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        self.stats["rewards"].append(sum(batch_rewards) / len(batch_rewards))
        self.stats["kl_divergences"].append(avg_kl)
        self.stats["policy_losses"].append(avg_policy_loss.item())
        self.stats["total_losses"].append(total_loss.item())
        self.stats["response_lengths"].append(sum(batch_lengths) / len(batch_lengths))

        return {
            "avg_reward": self.stats["rewards"][-1],
            "avg_kl": avg_kl,
            "policy_loss": avg_policy_loss.item(),
            "total_loss": total_loss.item(),
        }


def plot_ppo_stats(stats, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = [
        ("rewards", "Average Reward", "Reward Value"),
        ("kl_divergences", "KL Divergence", "KL"),
        ("policy_losses", "Policy Loss", "Loss"),
        ("response_lengths", "Avg Response Length", "Chars"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        ax.plot(stats[key], "o-", markersize=4)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_ppo(config, output_dir: str):
    logger = setup_logger("PPO", log_file=os.path.join(output_dir, "ppo.log"))
    logger.info("=" * 60)
    logger.info("Stage 3: PPO Alignment Training")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load policy model (SFT checkpoint or base)
    sft_path = config.model.policy_model_path
    if os.path.exists(sft_path):
        logger.info(f"Loading SFT model from: {sft_path}")
        policy_model = AutoModelForCausalLM.from_pretrained(sft_path, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(sft_path)
    else:
        logger.info(f"SFT model not found. Loading base: {config.model.base_model_name}")
        policy_model = AutoModelForCausalLM.from_pretrained(config.model.base_model_name, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy_model = policy_model.to(device)

    # Reference model
    logger.info("Creating frozen reference model...")
    ref_model = create_reference_model(policy_model)

    # Reward model
    logger.info("Initializing reward model...")
    rm_backbone = None
    if os.path.exists(config.model.reward_model_path):
        rm_backbone = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_name, torch_dtype=torch.float32
        )
        reward_model = SimpleRewardModel(tokenizer, rm_backbone, config.model.reward_model_path)
        logger.info("Loaded trained neural reward model.")
    else:
        reward_model = SimpleRewardModel(tokenizer)
        logger.info("Using rule-based reward model fallback.")

    # Test prompts
    test_prompts = config.data.test_prompts
    logger.info("Testing before PPO alignment...")
    before_responses, before_rewards = [], []
    for prompt in test_prompts:
        response, _ = generate_response(policy_model, tokenizer, prompt, max_new_tokens=80, temperature=0.7)
        reward = reward_model.score(prompt, response)
        before_responses.append(response)
        before_rewards.append(reward)
        logger.info(f"  [{reward:.3f}] {response[:60]}...")

    # PPO training
    ppo = PPOTrainer(policy_model, ref_model, reward_model, tokenizer, config)

    train_pool = [
        "Explain what deep learning is.",
        "Write a bubble sort in Python.",
        "How do I learn programming?",
        "What is artificial intelligence?",
        "Recommend technical books.",
        "How to prepare for a technical interview?",
        "Explain RESTful API.",
        "Write an encouraging message for a student.",
    ]

    logger.info("Starting PPO training...")
    for step in range(config.ppo.num_steps):
        step_prompts = random.sample(train_pool, min(config.ppo.batch_size, len(train_pool)))
        stats = ppo.train_step(step_prompts)
        logger.info(f"Step {step+1}/{config.ppo.num_steps} | "
                    f"Reward: {stats['avg_reward']:.3f} | "
                    f"KL: {stats['avg_kl']:.4f} | "
                    f"Loss: {stats['policy_loss']:.4f}")

    # Test after
    logger.info("Testing after PPO alignment...")
    policy_model.eval()
    after_responses, after_rewards = [], []
    for prompt in test_prompts:
        response, _ = generate_response(policy_model, tokenizer, prompt, max_new_tokens=80, temperature=0.7)
        reward = reward_model.score(prompt, response)
        after_responses.append(response)
        after_rewards.append(reward)

    avg_before = sum(before_rewards) / len(before_rewards)
    avg_after = sum(after_rewards) / len(after_rewards)
    logger.info(f"Average reward: {avg_before:.3f} → {avg_after:.3f} ({avg_after - avg_before:+.3f})")

    # Save
    model_path = os.path.join(output_dir, "aligned_model")
    policy_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Aligned model saved to: {model_path}")

    save_json({
        "stats": ppo.stats,
        "test_prompts": test_prompts,
        "before_rewards": before_rewards,
        "after_rewards": after_rewards,
        "before_responses": before_responses,
        "after_responses": after_responses,
    }, os.path.join(output_dir, "ppo_results.json"))

    plot_path = os.path.join(output_dir, "ppo_training_stats.png")
    plot_ppo_stats(ppo.stats, plot_path)
    logger.info(f"Training plot saved to: {plot_path}")

    logger.info("PPO alignment complete!")
    return model_path
