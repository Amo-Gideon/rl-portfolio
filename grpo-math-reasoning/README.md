# GRPO on Verifiable Math Reasoning

Group Relative Policy Optimization (the DeepSeek-R1 recipe) applied to programmatically generated arithmetic problems with exact-match rewards — no learned reward model needed.

A 0.5B instruct model is fine-tuned with LoRA. For each problem, the trainer samples a **group** of responses, scores them with a verifiable reward, normalizes advantages within the group, and takes clipped surrogate PPO steps with a per-token KL penalty against a frozen reference model.

## Setup

```bash
pip install torch transformers peft accelerate pyyaml tqdm
```

Uses the same local `Qwen2.5-0.5B-Instruct` copy as the other projects in this repo (place it under `./models/Qwen2.5-0.5B-Instruct`, or point `model.name` in `configs/grpo.yaml` elsewhere).

Sanity checks (no model needed):

```bash
python tests/test_tasks.py
```

## Train

```bash
python scripts/train.py --config configs/grpo.yaml
```

Default run: 5 epochs x 40 GRPO steps, group size 8, on 300 generated problems.

## Evaluate

Greedy decoding on a held-out seed:

```bash
python scripts/eval.py --checkpoint outputs/grpo_math/final --num_tasks 100
```

## How it works

- **Tasks** (`src/tasks.py`): add / subtract / multiply and two-step compositions (`a + b*c`), generated with answers for exact verification.
- **Reward**: 0.8 x exact answer match + 0.2 x using the required `ANSWER:` format. Exact match via numeric comparison, with a fallback last-number extractor.
- **Trainer** (`src/grpo.py`): group sampling -> group-relative advantage `(r - mean) / std` -> clipped surrogate per response token -> KL penalty per token (Dr. GRPO style) -> no value network.

## Notes

- Group-relative advantages mean the model learns from *which of its own samples solved the problem*, not from absolute difficulty — this is what makes GRPO stable without a critic.
- KL per token (rather than per sequence) avoids the length bias that lets models game sequence-level KL penalties.
