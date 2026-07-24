# Modern RLHF: From SFT to PPO Alignment

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A **production-grade** implementation of the RLHF three-stage pipeline: **SFT → Reward Model → PPO Alignment**. Built with Hugging Face TRL, LoRA, and real datasets.

[📖 Blog Post](docs/BLOG.md) · [🚀 Quick Start](#quick-start) · [📊 Results](#results)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔧 **Modular Design** | Each stage is self-contained; swap data, models, or trainers independently |
<<<<<<< HEAD
| ⚙️ **Config-Driven** | All hyperparameters via YAML zero code changes needed |
=======
| ⚙️ **Config-Driven** | All hyperparameters via YAML — zero code changes needed |
>>>>>>> 5c29891 (Update blog, results, and add new files)
| 🧠 **TRL Integration** | Uses battle-tested TRL library for SFT and PPO |
| 🎯 **LoRA Support** | Memory-efficient training with PEFT adapters |
| 📚 **Real Datasets** | Alpaca for SFT, HH-RLHF for reward model |
| 📈 **Full Evaluation** | Before/after comparison with reward model scoring |
| 🔬 **Reproducible** | Fixed seeds, pinned dependencies, timestamped outputs |

---

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/Amo-Gideon/modern-rlhf.git
cd modern-rlhf
pip install -e ".[dev]"
```

### Run the Pipeline

```bash
# Stage 1: Supervised Fine-Tuning
python scripts/run_sft.py --config configs/sft.yaml

# Stage 2: Reward Model Training
python scripts/run_reward_model.py --config configs/reward_model.yaml

# Stage 3: PPO Alignment
python scripts/run_ppo.py --config configs/ppo.yaml

# Evaluation: Compare all stages
python scripts/eval.py \
  --sft-path outputs/sft_latest/sft_model \
  --ppo-path outputs/ppo_latest/aligned_model \
  --rm-path outputs/rm_latest/reward_model
```

Or run everything at once:

```bash
bash scripts/run_full_pipeline.sh
```

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Stage 1   │ ──▶ │    Stage 2      │ ──▶ │    Stage 3      │
│    SFT      │     │  Reward Model   │     │  PPO Alignment  │
│             │     │                 │     │                 │
│   Alpaca    │     │  HH-RLHF pairs  │     │  Actor + Ref    │
│  + LoRA     │     │  Bradley-Terry  │     │  + Reward Model │
│             │     │   Loss          │     │  TRL PPOTrainer │
└─────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 📊 Results

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

### Training Curves

<<<<<<< HEAD
*Plots generated from training logs see `assets/` directory.*
=======
*Plots generated from training logs — see `assets/` directory.*
>>>>>>> 5c29891 (Update blog, results, and add new files)

---

## 📁 Project Structure

```
modern-rlhf/
├── configs/                  # YAML configuration files
│   ├── sft.yaml
│   ├── reward_model.yaml
│   └── ppo.yaml
├── src/rlhf_pipeline/
│   ├── data/                 # Data loading (Alpaca, HH-RLHF)
│   ├── models/               # Reward model, reference model
│   ├── trainers/             # SFT, RM, and PPO trainers
│   ├── utils/                # Config, logging, metrics, checkpointing
│   └── cli.py                # Unified CLI entry point
├── scripts/                  # Executable training scripts
│   ├── run_sft.py
│   ├── run_reward_model.py
│   ├── run_ppo.py
│   ├── run_full_pipeline.sh
│   └── eval.py               # Before/after evaluation
├── tests/                    # Sanity checks
├── docs/                     # Blog post & technical write-ups
├── assets/                   # Training curves, plots, diagrams
└── README.md
```

---

## 🎛️ Customization

### Use Your Own Data

Edit `configs/sft.yaml`:
```yaml
data:
  source: "huggingface"
  dataset_name: "HuggingFaceH4/ultrachat_200k"
  split: "train_sft"
  max_samples: 10000
```

### Change the Base Model

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  torch_dtype: "bfloat16"
```

### Adjust PPO Hyperparameters

```yaml
ppo:
  learning_rate: 1.0e-5
  kl_coef: 0.2
  clip_range: 0.2
  num_steps: 300
  batch_size: 16
  ppo_epochs: 2
```

---

## 📝 Blog Post

See [`docs/BLOG.md`](docs/BLOG.md) for a deep dive into the architecture, lessons learned, and scaling strategies.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 Citation

```bibtex
@software{modern_rlhf,
  title = {Modern RLHF: A Modular Pipeline from SFT to PPO},
  author = {Appau Gideon Kofi Amo},
  year = {2026},
  url = {https://github.com/Amo-Gideon/modern-rlhf}
}
```

---

## 📧 Contact

- **Author:** Appau Gideon Kofi Amo
- **Email:** gideonamoappau@gmail.com
<<<<<<< HEAD
- **GitHub:** [@Amo-Gideon](https://github.com/Amo-Gideon)
=======
- **GitHub:** [@Amo-Gideon](https://github.com/Amo-Gideon)
>>>>>>> 5c29891 (Update blog, results, and add new files)
