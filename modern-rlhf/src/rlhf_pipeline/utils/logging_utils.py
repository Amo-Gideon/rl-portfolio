"""Logging and experiment tracking utilities."""
import logging
import sys
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Set up a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class WandbTracker:
    """Optional Weights & Biases tracker."""

    def __init__(self, project: str, config: dict, enabled: bool = True):
        self.enabled = enabled and project != "none"
        if self.enabled:
            try:
                import wandb
                wandb.init(project=project, config=config)
                self._wandb = wandb
            except ImportError:
                self.enabled = False

    def log(self, metrics: dict, step: Optional[int] = None):
        if self.enabled:
            self._wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            self._wandb.finish()
