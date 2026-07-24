"""Configuration loading utilities."""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=list)
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    hidden_size: Optional[int] = None
    freeze_base: bool = False
    use_lora: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    num_labels: int = 1
    policy_model_path: Optional[str] = None
    base_model_name: Optional[str] = None
    reward_model_path: Optional[str] = None


@dataclass
class DataConfig:
    source: str = "huggingface"
    dataset_name: Optional[str] = None
    split: str = "train"
    max_samples: Optional[int] = None
    max_length: int = 512
    train_test_split: float = 0.9
    text_column: Optional[str] = None
    prompt_column: Optional[str] = None
    chosen_column: Optional[str] = None
    rejected_column: Optional[str] = None
    json_path: Optional[str] = None
    test_prompts: List[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    per_device_train_batch_size: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    logging_steps: int = 10
    save_strategy: str = "epoch"
    report_to: str = "wandb"
    fp16: bool = False
    bf16: bool = True
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    loss_type: str = "bt_pairwise"
    eval_strategy: str = "steps"
    eval_steps: int = 100


@dataclass
class PPOConfig:
    learning_rate: float = 1e-5
    kl_coef: float = 0.2
    clip_range: float = 0.2
    num_steps: int = 300
    batch_size: int = 16
    mini_batch_size: int = 4
    ppo_epochs: int = 2
    gradient_clip: float = 1.0
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    target_kl: float = 0.02
    log_interval: int = 1
    save_interval: int = 50


@dataclass
class TrackingConfig:
    log_interval: int = 1
    save_plots: bool = True
    compare_before_after: bool = True
    project_name: str = "modern-rlhf"


@dataclass
class ExperimentConfig:
    name: str = "rlhf_experiment"
    output_dir: str = "outputs"
    seed: int = 42


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    evaluation: Dict[str, Any] = field(default_factory=dict)


def _dict_to_dataclass(cls, data: Dict[str, Any]) -> Any:
    """Recursively convert dict to dataclass, handling nested dataclasses."""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            if hasattr(field_type, '__dataclass_fields__'):
                kwargs[key] = _dict_to_dataclass(field_type, value)
            else:
                kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: str) -> Config:
    """Load a YAML config file into a Config dataclass."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _dict_to_dataclass(Config, raw)


def setup_output_dir(config: Config) -> str:
    """Create a timestamped output directory for an experiment."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        config.experiment.output_dir,
        f"{timestamp}_{config.experiment.name}"
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
