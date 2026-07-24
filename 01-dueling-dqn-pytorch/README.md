# Dueling DQN vs Standard DDQN (PyTorch)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A from-scratch implementation of **Dueling Deep Q-Networks** [(Wang et al., 2016)](https://arxiv.org/abs/1511.06581) with a head-to-head comparison against standard Double DQN on Atari Pong.

> **Key Idea:** Dueling architectures separate the estimation of state-value `V(s)` and action advantages `A(s,a)`, leading to more stable learning and better policy evaluation — especially in environments with large action spaces.

---

## Architecture

### Dueling DQN
```
Input State → Shared Conv Layers → [V(s) Stream] + [A(s,a) Stream] → Q(s,a) = V(s) + (A(s,a) - mean(A))
```

- **Value Stream**: Estimates how good it is to be in state `s`
- **Advantage Stream**: Estimates the relative benefit of each action
- **Aggregation**: `Q(s,a) = V(s) + A(s,a) - mean(A)` (identifiability trick)

### Standard DDQN (Baseline)
```
Input State → Conv Layers → Single Q-value Head
```

Same total capacity as the dueling network, but without the value/advantage decomposition.

---

## Results (Pong-v5, 500 Episodes)

| Model | Last-50 Mean Score | Best Score | Notes |
|-------|-------------------|------------|-------|
| **Standard DDQN** | ~-10.5 | -8.6 | Single-stream baseline |
| **Dueling DDQN** | ~-11.5 | -10.1 | Value + Advantage decomposition |

> **Note:** Pong has only 6 discrete actions. The dueling advantage is more pronounced on high-action games (e.g., Breakout, Space Invaders) per the original paper. Both agents learn to rally consistently within 500 episodes.

### Training Curves

![Pong Training Comparison](results/comparison.png)

---

## Quick Start

### Install
```bash
git clone https://github.com/Amo-Gideon/rl-portfolio.git
cd rl-portfolio/01-dueling-dqn-pytorch
pip install -r requirements.txt
```

### Train Both Agents
```bash
python main.py
```

This trains both Dueling DDQN and Standard DDQN side-by-side and saves:
- `results/comparison.png` — training curves
- `results/dueling_checkpoint.pt` — best dueling model
- `results/standard_checkpoint.pt` — best standard model

---

## Project Structure

```
01-dueling-dqn-pytorch/
├── main.py              # Training loop + comparison
├── dueling_dqn.py       # Dueling architecture (V + A streams)
├── standard_dqn.py      # Standard DDQN baseline
├── replay_buffer.py     # Experience replay with prioritized sampling
├── train.py             # Single-agent training wrapper
├── utils.py             # Plotting, logging, evaluation
├── requirements.txt     # Dependencies
└── results/             # Training curves + saved checkpoints
```

---

## Key Implementation Details

| Component | Detail |
|-----------|--------|
| **Algorithm** | Double DQN with gradient clipping |
| **Replay** | Prioritized Experience Replay (PER) |
| **Target Update** | Soft update (τ = 0.001) every 4 steps |
| **Epsilon** | Decay from 1.0 → 0.01 over 10,000 frames |
| **Optimizer** | Adam (lr = 1e-4) |
| **Frame Stack** | 4 consecutive frames |

---

## What I Learned

1. **Dueling helps policy evaluation.** Separating value and advantage makes the network more robust to overestimation of Q-values for suboptimal actions.
2. **PER matters more than architecture.** On simple games like Pong, prioritized replay contributes more to sample efficiency than the dueling head itself.
3. **Gradient clipping is essential.** Atari frames are highly correlated; clipping prevents exploding gradients during early training.

---

## References

- **Wang et al. (2016)** — *Dueling Network Architectures for Deep Reinforcement Learning* ([arXiv:1511.06581](https://arxiv.org/abs/1511.06581))
- **van Hasselt et al. (2016)** — *Deep Reinforcement Learning with Double Q-learning* ([arXiv:1509.06461](https://arxiv.org/abs/1509.06461))
- **Schaul et al. (2016)** — *Prioritized Experience Replay* ([arXiv:1511.05952](https://arxiv.org/abs/1511.05952))

---

## License

MIT License — see the [root LICENSE](../LICENSE) for details.

---

*Built by [Appau Gideon Kofi Amo](mailto:gideonamoappau@gmail.com) as part of the [RL Portfolio](https://github.com/Amo-Gideon/rl-portfolio).*
