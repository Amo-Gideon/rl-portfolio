"""Configuration loading utilities."""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "float32"
    trust_remote_code: bool = True
    hidden_size: Optional[int] = None
    freeze_base: bool = False
    policy_model_path: Optional[str] = None
    base_model_name: Optional[str] = None
    reward_model_path: Optional[str] = None


@dataclass
class DataConfig:
    source: str = "toy"  # toy | huggingface | json
    dataset_name: Optional[str] = None
    split: str = "train"
    max_samples: Optional[int] = None
    json_path: Optional[str] = None
    max_length: int = 512
    train_test_split: float = 0.8
    text_column: Optional[str] = None
    prompt_column: Optional[str] = None
    chosen_column: Optional[str] = None
    rejected_column: Optional[str] = None
    test_prompts: List[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-5
    num_train_epochs: int = 2
    logging_steps: int = 1
    save_strategy: str = "epoch"
    report_to: str = "none"
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    batch_size: int = 2
    epochs: int = 5
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    gradient_clip: float = 1.0


@dataclass
class PPOConfig:
    learning_rate: float = 1e-6
    kl_coef: float = 0.1
    clip_range: float = 0.2
    num_steps: int = 10
    batch_size: int = 4
    max_new_tokens: int = 80
    temperature: float = 0.7
    top_p: float = 0.9
    gradient_clip: float = 1.0


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
    evaluation: dict = field(default_factory=dict)
    tracking: dict = field(default_factory=dict)


def load_config(config_path: str) -> Config:
    """Load a YAML config file into a Config dataclass."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        experiment=ExperimentConfig(**raw.get("experiment", {})),
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        training=TrainingConfig(**raw.get("training", {})),
        ppo=PPOConfig(**raw.get("ppo", {})),
        evaluation=raw.get("evaluation", {}),
        tracking=raw.get("tracking", {}),
    )


def setup_output_dir(config: Config) -> str:
    """Create a timestamped output directory for an experiment."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(
        config.experiment.output_dir,
        f"{timestamp}_{config.experiment.name}"
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
