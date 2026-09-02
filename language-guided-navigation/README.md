# Language-Guided Navigation Agent

A minimal embodied-AI portfolio project: a small LLM learns to follow natural-language navigation instructions on a grid, then answers a question about the target object.

**Why this exists:**
- Demonstrates **LLM + RL** in a spatial/embodied setting.
- Fits Prof. Yan Xia’s SPIN Lab interests (spatial intelligence, embodied AI, navigation).
- Runs end-to-end on a **CPU** or small GPU instance (e.g., PAI-DSW 8-core/32GB or the AMD GPU quota).

## Pipeline

1. **Environment** (`src/env.py`): Grid world with landmarks, JSON actions, verifiable rewards.
2. **Expert data** (`src/dataset.py`): Shortest-path trajectories generated with BFS.
3. **Supervised warm-start** (`scripts/sft.py`): Fine-tune `Qwen2.5-0.5B-Instruct` with LoRA on expert demos.
4. **RL fine-tuning** (`scripts/train_rl.py`): REINFORCE + KL penalty against the SFT checkpoint.
5. **Evaluation** (`scripts/eval.py`) and **demo** (`scripts/demo.py`).

## Quick start in PAI-DSW / Online VS Code

The image in the screenshot (`ubuntu22.04-py312-torch2.3.1-1.39.0`) already has PyTorch. Install the remaining dependencies:

```bash
git clone https://github.com/Amo-Gideon/rl-portfolio.git
cd rl-portfolio/language-guided-navigation
pip install transformers peft accelerate pyyaml tqdm
```

### Download the base model

PAI-DSW sometimes cannot reach HuggingFace directly. Use the ModelScope downloader (ModelScope is preinstalled in the image):

```bash
python scripts/download_model.py --source modelscope
```

This prints a local path like `models/qwen/Qwen2.5-0.5B-Instruct`. Set that path as `model.name` in `configs/sft.yaml` and `configs/rl.yaml`, or keep the default `./models/Qwen2.5-0.5B-Instruct` if you symlink/copy the folder there.

If ModelScope is also unreachable, try the HuggingFace mirror:

```bash
python scripts/download_model.py --source hf-mirror
```

If that keeps failing too, download the model locally and upload it manually:

1. **On your local machine**, download with huggingface_hub:
   ```bash
   pip install huggingface_hub
   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='Qwen2.5-0.5B-Instruct')"
   ```
2. **Zip the folder**:
   ```bash
   zip -r Qwen2.5-0.5B-Instruct.zip Qwen2.5-0.5B-Instruct
   ```
3. **Upload** the zip into PAI-DSW’s VS Code file explorer under:
   ```
   rl-portfolio/language-guided-navigation/models/
   ```
4. **Unzip** in the PAI-DSW terminal:
   ```bash
   cd rl-portfolio/language-guided-navigation/models
   unzip Qwen2.5-0.5B-Instruct.zip
   ```
5. Make sure `model.name: "./models/Qwen2.5-0.5B-Instruct"` is set in `configs/sft.yaml` and `configs/rl.yaml`.

### SFT warm-start

```bash
python scripts/sft.py --config configs/sft.yaml
```

### RL fine-tuning

```bash
python scripts/train_rl.py --config configs/rl.yaml
```

### Evaluate / demo

```bash
python scripts/eval.py --checkpoint outputs/sft --num_tasks 20
python scripts/demo.py --checkpoint outputs/rl
```

## Local development

No GPU required for the environment tests:

```bash
python tests/test_env.py
python tests/test_dataset.py
```

## Project structure

```
language-guided-navigation/
├── configs/          # YAML configs for SFT and RL
├── scripts/          # sft.py, train_rl.py, eval.py, demo.py
├── src/              # env.py, dataset.py, model_utils.py, rl_trainer.py
└── tests/            # unit tests for env + dataset
```

## Extending

- Replace the grid world with a **discrete 3D scene graph** or Matterport-like connectivity graph.
- Swap REINFORCE for **GRPO** (group-relative advantage) from the `AGENTIC-RL` subproject.
- Add a vision encoder so the agent reads rendered views instead of text observations.
