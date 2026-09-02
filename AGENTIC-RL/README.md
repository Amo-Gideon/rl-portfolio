# AGENTIC-RL

Train a small LLM (Qwen2.5-0.5B) to use tools via RL with verifiable rewards.

## What it does

The agent learns to emit JSON tool calls for `search`, `calculate`, and `run_code`, then produce a final answer. Rewards are fully programmatic:

- **Format reward**: valid JSON with required keys
- **Tool validity reward**: calls an allowed tool
- **Answer accuracy reward**: numeric or exact-string match against ground truth

Two trainers are included:

- **GRPO** — group-relative advantages, no critic model
- **VIMPO** — per-token value assignment via backward DP

## Quick start

```bash
# From inside AGENTIC-RL/
pip install -r requirements.txt
python scripts/run_sft.py --config configs/sft.yaml
python scripts/train_grpo.py --config configs/grpo_4gb_gpu.yaml
```

CPU-only machine? Use `configs/grpo_cpu_only.yaml`.

## Project structure

```
AGENTIC-RL/
├── configs/            # training configs for different setups
├── data/               # SFT trajectories (small sample included)
├── notebooks/          # demo notebooks
├── scripts/            # training/evaluation entry points
├── src/
│   ├── envs/           # multi-tool environment
│   ├── models/         # GRPO / VIMPO / SFT trainers
│   ├── rewards/        # verifiable reward decomposition
│   └── utils/          # checkpointing, logging, eval
└── tests/              # unit tests
```

## Tests

```bash
python tests/test_env.py
python tests/test_grpo.py
```

## Safety note

`run_code` is restricted (no imports, no file access), but it still executes Python. Do not expose it to untrusted inputs without additional sandboxing.
