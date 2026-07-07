#!/bin/bash
set -e

echo "=========================================="
echo "Modern RLHF: Full Pipeline"
echo "=========================================="

# Stage 1: SFT
echo ""
echo "[Stage 1] Supervised Fine-Tuning..."
python scripts/run_sft.py --config configs/sft.yaml

# Stage 2: Reward Model
echo ""
echo "[Stage 2] Reward Model Training..."
python scripts/run_reward_model.py --config configs/reward_model.yaml

# Stage 3: PPO
echo ""
echo "[Stage 3] PPO Alignment..."
python scripts/run_ppo.py --config configs/ppo.yaml

echo ""
echo "=========================================="
echo "All stages complete!"
echo "=========================================="
