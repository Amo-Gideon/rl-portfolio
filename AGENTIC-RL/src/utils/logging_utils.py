"""
Logging utilities for tracking experiments.
"""

import wandb
from typing import Dict, Optional


class ExperimentLogger:
    """Wrapper for wandb and console logging."""

    def __init__(self, use_wandb: bool, project: str, run_name: str, config: Optional[Dict] = None):
        self.use_wandb = use_wandb
        self.config = config or {}

        if self.use_wandb:
            wandb.init(project=project, name=run_name, config=config)

    def log(self, metrics: Dict, step: int):
        """Log metrics to wandb and console."""
        if self.use_wandb:
            wandb.log(metrics, step=step)

        # Console print key metrics
        msg = f"Step {step} | " + " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        print(msg)

    def finish(self):
        if self.use_wandb:
            wandb.finish()
