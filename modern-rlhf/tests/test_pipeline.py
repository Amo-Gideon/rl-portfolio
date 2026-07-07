"""Sanity tests for the RLHF pipeline components."""
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rlhf_pipeline.data.sft_data import generate_toy_sft_data, format_sft_dataset
from rlhf_pipeline.data.preference_data import generate_toy_preference_data, PreferenceDataset
from rlhf_pipeline.models.reward_model import RewardModel, SimpleRewardModel
from rlhf_pipeline.models.reference_model import create_reference_model
from rlhf_pipeline.utils.config import Config, ModelConfig, DataConfig


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class TestSFTData:
    """Tests for SFT data generation and formatting."""

    def test_toy_data_length(self):
        data = generate_toy_sft_data()
        assert len(data) == 10
        assert all("instruction" in d and "response" in d for d in data)

    def test_format_sft_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        data = generate_toy_sft_data()
        dataset = format_sft_dataset(data, tokenizer)
        assert len(dataset) == 10
        assert "text" in dataset.features


class TestPreferenceData:
    """Tests for preference data and dataset class."""

    def test_toy_preference_data(self):
        data = generate_toy_preference_data()
        assert len(data) == 10
        assert all("prompt" in d and "chosen" in d and "rejected" in d for d in data)

    def test_preference_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        data = generate_toy_preference_data()
        ds = PreferenceDataset(data, tokenizer, max_length=128)
        assert len(ds) == 10
        sample = ds[0]
        assert "chosen_input_ids" in sample
        assert "rejected_input_ids" in sample
        assert sample["chosen_input_ids"].shape == sample["rejected_input_ids"].shape


class TestRewardModel:
    """Tests for reward model components."""

    def test_reward_model_forward(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = RewardModel(MODEL_NAME, hidden_size=896, freeze_base=True)

        text = "Hello world"
        enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            reward = model(enc["input_ids"], enc["attention_mask"])

        assert reward.shape == torch.Size([1])
        assert torch.isfinite(reward).all()

    def test_simple_reward_model(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        rm = SimpleRewardModel(tokenizer)

        score_good = rm.score("What is 2+2?", "2+2 equals 4. Here is the step-by-step calculation: 1. Start with 2. 2. Add another 2. 3. Result is 4.")
        score_bad = rm.score("What is 2+2?", "idk")

        assert score_good > score_bad

    def test_simple_reward_structure_bonus(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        rm = SimpleRewardModel(tokenizer)

        structured = rm.score("Explain", "1. First point\n2. Second point")
        unstructured = rm.score("Explain", "Some random text without structure")

        assert structured > unstructured


class TestReferenceModel:
    """Tests for reference model creation."""

    def test_reference_model_frozen(self):
        policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        ref = create_reference_model(policy)

        assert ref.training == False
        assert all(p.requires_grad == False for p in ref.parameters())

    def test_reference_model_same_architecture(self):
        policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        ref = create_reference_model(policy)

        assert type(policy) == type(ref)
        assert policy.config.vocab_size == ref.config.vocab_size


class TestConfig:
    """Tests for configuration loading."""

    def test_default_config(self):
        config = Config()
        assert config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert config.data.source == "toy"
        assert config.experiment.seed == 42

    def test_config_dataclass_types(self):
        config = Config()
        assert isinstance(config.training.learning_rate, float)
        assert isinstance(config.ppo.kl_coef, float)
        assert isinstance(config.data.max_length, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
