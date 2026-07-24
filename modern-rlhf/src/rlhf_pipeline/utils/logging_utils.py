"""Logging and experiment tracking utilities."""
import logging
import sys
import os
from typing import Optional, Dict, Any


def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Set up a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class WandbTracker:
    """Weights & Biases experiment tracker."""

    def __init__(self, project: str, config: Dict[str, Any], enabled: bool = True, run_name: Optional[str] = None):
        self.enabled = enabled and project not in ("none", "", None)
        self._wandb = None

        if self.enabled:
            try:
                import wandb
                self._wandb = wandb
                self._run = wandb.init(
                    project=project,
                    config=config,
                    name=run_name,
                    reinit=True
                )
            except ImportError:
                self.enabled = False
                print("WARNING: wandb not installed. Tracking disabled.")
            except Exception as e:
                self.enabled = False
                print(f"WARNING: wandb init failed: {e}. Tracking disabled.")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.enabled and self._wandb:
            self._wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled and self._wandb:
            self._wandb.finish()
