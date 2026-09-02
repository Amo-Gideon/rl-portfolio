"""
Checkpoint management for training resumption and model saving.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Optional


class CheckpointManager:
    """Handles saving/loading of model, optimizer, and training state."""

    def __init__(self, output_dir: str, keep_last_n: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.saved_checkpoints = []

    def save(self, 
             epoch: int,
             model,
             optimizer,
             scheduler=None,
             metrics: Optional[Dict] = None,
             prefix: str = "grpo") -> str:
        """Save a checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        path = self.output_dir / f"{prefix}_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        self.saved_checkpoints.append(path)

        if len(self.saved_checkpoints) > self.keep_last_n:
            old = self.saved_checkpoints.pop(0)
            if old.exists():
                old.unlink()

        return str(path)

    def load(self, model, optimizer, scheduler=None, checkpoint_path: Optional[str] = None):
        """Load from latest or specified checkpoint."""
        if checkpoint_path is None:
            checkpoints = sorted(self.output_dir.glob("*.pt"), key=os.path.getmtime)
            if not checkpoints:
                return 0
            checkpoint_path = str(checkpoints[-1])

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint.get("epoch", 0)
