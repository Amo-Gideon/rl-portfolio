"""Sanity tests for the RLHF pipeline components."""
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from rlhf_pipeline.data.sft_data import generate_toy_sft_data, load_sft_data
from rlhf_pipeline.data.preference_data import generate_toy_preference_data, PreferenceDataset
from rlhf_pipeline.models.reward_model import RewardModel
from rlhf_pipeline.models.reference_model import create_reference_model
from rlhf_pipeline.utils.config import Config, load_config, set_seed


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class TestSFTData:
    """Tests for SFT data generation and formatting."""

    def test_toy_data_length(self):
        data = generate_toy_sft_data()
        assert len(data) == 10
        assert all("instruction" in d and "response" in d for d in data)

    def test_format_chat(self):
        from rlhf_pipeline.data.sft_data import format_chat_example
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        ex = {"instruction": "Test", "output": "Answer"}
        text = format_chat_example(ex, tokenizer)
        assert isinstance(text, str)
        assert len(text) > 0


class TestPreferenceData:
    """Tests for preference data and dataset class."""

    def test_toy_preference_data(self):
        data = generate_toy_preference_data()
        assert len(data) == 10
        assert all("prompt" in d and "chosen" in d and "rejected" in d for d in data)

    def test_preference_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        data = generate_toy_preference_data()
        ds = PreferenceDataset(data, tokenizer, max_length=128)
        assert len(ds) == 10
        sample = ds[0]
        assert "chosen_input_ids" in sample
        assert "rejected_input_ids" in sample


class TestRewardModel:
    """Tests for reward model components."""

    def test_reward_model_forward(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = RewardModel(MODEL_NAME, hidden_size=896, freeze_base=True)
        text = "Hello world"
        enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            reward = model(enc["input_ids"], enc["attention_mask"])

        assert reward.shape == torch.Size([1])
        assert torch.isfinite(reward).all()

    def test_reward_model_pairwise(self):
        """Test that model can score pairs."""
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = RewardModel(MODEL_NAME, hidden_size=896, freeze_base=True)

        prompt = "What is 2+2?"
        chosen = "2+2 equals 4."
        rejected = "idk"

        messages_c = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]
        messages_r = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]

        text_c = tokenizer.apply_chat_template(messages_c, tokenize=False, add_generation_prompt=False)
        text_r = tokenizer.apply_chat_template(messages_r, tokenize=False, add_generation_prompt=False)

        enc_c = tokenizer(text_c, return_tensors="pt", padding=True, truncation=True)
        enc_r = tokenizer(text_r, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            score_c = model(enc_c["input_ids"], enc_c["attention_mask"])
            score_r = model(enc_r["input_ids"], enc_r["attention_mask"])

        assert score_c.shape == score_r.shape == torch.Size([1])


class TestReferenceModel:
    """Tests for reference model creation."""

    def test_reference_model_frozen(self):
        policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        ref = create_reference_model(policy)

        assert ref.training == False
        assert all(p.requires_grad == False for p in ref.parameters())

    def test_reference_model_same_outputs(self):
        """Reference model should produce same outputs as policy at init."""
        policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        ref = create_reference_model(policy)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        text = "Hello world"
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            out_policy = policy(**inputs)
            out_ref = ref(**inputs)

        assert torch.allclose(out_policy.logits, out_ref.logits, atol=1e-5)


class TestConfig:
    """Tests for configuration loading."""

    def test_default_config(self):
        config = Config()
        assert config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert config.data.source == "huggingface"
        assert config.experiment.seed == 42

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
experiment:
  name: test_exp
  seed: 123
model:
  name: test-model
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        config = load_config(str(yaml_path))
        assert config.experiment.name == "test_exp"
        assert config.experiment.seed == 123
        assert config.model.name == "test-model"

    def test_set_seed(self):
        set_seed(42)
        a = torch.rand(10)
        set_seed(42)
        b = torch.rand(10)
        assert torch.allclose(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])