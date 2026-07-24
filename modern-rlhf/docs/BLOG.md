# From Tutorial to Production: Building a Modular RLHF Pipeline

> **TL;DR**: I took three tutorial scripts (SFT → Reward Model → PPO) and transformed them into a production-grade, configurable, and blog-worthy RLHF framework. This post walks through the architecture, the code, and the lessons learned.

---

## 1. Why RLHF Matters (And Why Most Tutorials Fall Short)

Reinforcement Learning from Human Feedback (RLHF) is the secret sauce behind ChatGPT, Claude, and Llama 2 Chat. It transforms a base language model from a "text completer" into a "helpful assistant."

The standard pipeline has three stages:
1. **SFT** — Teach the model to follow instructions
2. **Reward Model** — Learn to score "good" vs "bad" responses
3. **PPO** — Optimize the policy to maximize reward while staying close to the original

Most tutorials give you three monolithic scripts with hardcoded paths, toy data, and no way to experiment. That works for learning, but not for:
- Swapping in real datasets (Alpaca, HH-RLHF, ShareGPT)
- Trying different base models (Qwen, Llama, Mistral)
- Tracking experiments (WandB, TensorBoard)
- Reproducing results

This post shows how to bridge that gap.

---

## 2. The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        STAGE 1: SFT                          │
│  Instruction Data → SFTTrainer → Fine-tuned Model              │
│                                                              │
│  Goal: Teach the model to answer questions in a helpful way   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 2: Reward Model                     │
│  Preference Pairs (prompt, chosen, rejected)                 │
│  → Bradley-Terry Loss → Value Head                           │
│                                                              │
│  Goal: Learn to predict human preference rankings           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     STAGE 3: PPO Alignment                   │
│  Policy Model + Reference Model + Reward Model               │
│  → Generate → Score → Clip → Update                          │
│                                                              │
│  Goal: Optimize responses for higher reward scores          │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1: Supervised Fine-Tuning (SFT)

### The Intuition

Before RLHF, the model needs to know *how* to answer questions. SFT is like giving a student a textbook with question-answer pairs. The model learns the format, tone, and structure of helpful responses.

### Key Code

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        learning_rate=2e-5,
        num_train_epochs=2,
        per_device_train_batch_size=2,
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### What I Learned

- **SFT is not optional**. Skipping it and going straight to PPO is like trying to teach chess strategy to someone who doesn't know how the pieces move.
- **Data quality > quantity**. 1,000 high-quality examples beat 100,000 mediocre ones.
- **Chat templates matter**. Using the model's native chat template (e.g., Qwen's) is critical for good results.

---

## 4. Stage 2: Reward Model (RM)

### The Intuition

Humans are good at comparing two answers and saying "A is better than B," but terrible at assigning absolute scores. The Reward Model learns from these pairwise comparisons.

The **Bradley-Terry model** formalizes this:

```
P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
Loss = -log(sigmoid(r_chosen - r_rejected))
```

When `r_chosen` is much larger than `r_rejected`, the sigmoid approaches 1 and the loss approaches 0.

### Key Code

```python
class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]
        # Extract last valid token's hidden state
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_token = last_hidden[torch.arange(batch_size), seq_lengths]
        return self.value_head(last_token).squeeze(-1)
```

### What I Learned

- **The value head is tiny**. It's just one linear layer on top of the frozen (or fine-tuned) base model. The heavy lifting is already done by the LM.
- **Accuracy is a good sanity check**. If your RM can't beat random chance (50%) on held-out pairs, something is wrong.
- **Visualize distributions**. Plotting chosen vs rejected scores reveals whether your model has actually learned to distinguish quality.

---

## 5. Stage 3: PPO Alignment

### The Intuition

PPO is like a student (the **Actor**) taking a test, a teacher (the **Reward Model**) grading it, and a parent (the **Reference Model**) saying "don't change too much from how you used to answer."

The **clipped objective** prevents the student from changing too radically in one step:

```
ratio = π_new(a|s) / π_old(a|s)
L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

The **KL penalty** keeps the new policy close to the reference:

```
Total Loss = -L_CLIP + β * KL(π_new || π_ref)
```

### Key Code

```python
# Generate response
response = policy_model.generate(prompt)

# Score it
reward = reward_model.score(prompt, response)

# Compute advantage (simplified: reward - mean)
advantages = rewards - rewards.mean()

# PPO update
ratio = torch.exp(new_log_prob - old_log_prob)
surr1 = ratio * advantage
surr2 = clip(ratio, 1-ε, 1+ε) * advantage
policy_loss = -min(surr1, surr2)

# Add KL penalty
total_loss = policy_loss + β * kl_divergence
```

### What I Learned

- **PPO is notoriously unstable**. Start with tiny models (0.5B parameters) and small step counts (10-50).
- **KL divergence is your safety net**. Without it, the model can collapse into repetitive high-reward gibberish.
- **Reward hacking is real**. A rule-based reward model might give high scores for long responses; the model learns to ramble.

---

## 6. Making It Production-Ready

### Config-Driven Experiments

Instead of editing Python files, everything lives in YAML:

```yaml
# configs/ppo.yaml
model:
  policy_model_path: "outputs/sft_model"
  reward_model_path: "outputs/value_head.pt"

ppo:
  learning_rate: 1.0e-6
  kl_coef: 0.1
  clip_range: 0.2
  num_steps: 10
```

### Modular Architecture

Each component is swappable:
- **Data**: `toy` → `huggingface` → `json` with one line
- **Model**: Qwen → Llama → Mistral with one line
- **Trainer**: SFTTrainer → Custom trainer with one import

### Reproducibility

- Fixed random seeds
- Timestamped output directories (`2026-07-05_10-30-00_sft/`)
- Pinned dependencies in `pyproject.toml`

---

## 7. Results & Observations

Running the full pipeline on **Qwen2.5-0.5B-Instruct** (RTX 4090, 24GB):

| Stage | Metric | Value | Notes |
|-------|--------|-------|-------|
| **SFT** | Final Loss | **1.235** | 10K Alpaca samples, 1 epoch, LoRA r=16 |
| **SFT** | Trainable Params | **8.8M / 502.8M** | 1.75% of total parameters |
| **RM** | Test Accuracy | **100%** | 4/4 pairs correct (toy data) |
| **PPO** | Avg Reward Before | **0.213** | Baseline (SFT model) |
| **PPO** | Avg Reward After | **0.675** | After 300 PPO steps |
| **PPO** | Improvement | **+0.462** | **+217% relative improvement** |

### Before/After Examples

| Prompt | Before (SFT) | After (PPO) | Reward Δ |
|--------|-----------|------------|---------|
| "Write a Python function to find the maximum value in a list." | `def max_list_val(lst): ...` | `def max_value_in_list(nums): ...` | **+0.71** |
| "Explain what machine learning is in simple terms." | 4-sentence paragraph | 2-sentence concise | **-0.91** |
| "Describe the difference between a stack and a queue." | Incorrect (LIFO/FIFO swapped) | Correct (LIFO vs FIFO) | **+0.81** |
| "How does backpropagation work in neural networks?" | General description | More technical, gradient descent focus | **+0.18** |
| "Write a SQL query to find the top 5 highest-paid employees." | `SELECT * FROM Employees ORDER BY Salary DESC LIMIT 5;` | Same | **0.00** |
| "Explain the concept of attention in transformer models." | Vague description | Technical, attention map focus | **+1.98** |

### Key Observations

- **Code generation improved**: SFT → PPO responses became more concise and correct
- **Technical explanations improved**: Attention and backpropagation answers became more precise
- **Some degradation**: The "machine learning" prompt became overly concise (negative reward)
- **KL divergence stayed low**: Policy did not drift far from SFT initialization

---

## 8. What's Next?

### Try These Experiments

1. **Swap the base model**: Use `meta-llama/Llama-2-7b-hf` with `bfloat16`
2. **Use real data**: Load `tatsu-lab/alpaca` for SFT and `Anthropic/hh-rlhf` for RM
3. **Add LoRA**: Use `peft` to train adapters instead of full fine-tuning
4. **Try DPO**: Direct Preference Optimization skips the reward model entirely; compare it to PPO
5. **Try GRPO**: Group Relative Policy Optimization (no critic model needed) — see below

### Read These Papers

- **InstructGPT**: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **DPO**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **GRPO**: [DeepSeekMath](https://arxiv.org/abs/2402.03300) — no value network needed

---

## 9. Conclusion

RLHF is not magic, it's a systematic pipeline of three well-understood stages. The hard part is not the math; it's the engineering: data curation, distributed training, and hyperparameter tuning.

This project gives you a clean foundation to experiment with. Start with toy data, understand the mechanics, then scale up.

**GitHub**: [github.com/Amo-Gideon/modern-rlhf](https://github.com/Amo-Gideon/modern-rlhf)

---

*Written by [Appau Gideon Kofi Amo](mailto:gideonamoappau@gmail.com). Built with the `hands-on-modern-rl` tutorial series and a lot of debugging.*
