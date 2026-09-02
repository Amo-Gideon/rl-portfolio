"""
Model loading / helpers for the navigation agent.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_name: str, torch_dtype: str = "float32", device_map: str = "auto"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=getattr(torch, torch_dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    return model


def load_actor(model_name: str, lora_config: dict, torch_dtype: str = "float32", device_map: str = "auto"):
    model = load_base_model(model_name, torch_dtype, device_map)
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    return model


def save_adapter(model, tokenizer, path: str):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_adapter(path: str, base_model_name: str, torch_dtype: str = "float32", device_map: str = "auto", is_trainable: bool = False):
    """Load base model + LoRA adapter."""
    from peft import PeftModel

    base = load_base_model(base_model_name, torch_dtype, device_map)
    model = PeftModel.from_pretrained(base, path, is_trainable=is_trainable)
    tokenizer = load_tokenizer(path)
    return model, tokenizer


def compute_log_probs(model, tokenizer, observation: str, action_text: str):
    """
    Compute per-token log probabilities of action_text given observation.
    Returns tensor of shape [action_len].
    """
    prefix = f"{observation}\nAction: "
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_text = prefix + action_text
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    prefix_len = prefix_ids.shape[1]

    outputs = model(input_ids=full_ids)
    logits = outputs.logits[:, :-1, :]
    targets = full_ids[:, 1:]

    response_logits = logits[:, prefix_len - 1 :, :]
    response_targets = targets[:, prefix_len - 1 :]

    log_probs = torch.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=response_targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.squeeze(0)


def generate_action(model, tokenizer, observation: str, max_new_tokens: int = 64, do_sample: bool = True, temperature: float = 0.7):
    """Generate one action / final answer JSON string."""
    prompt = f"{observation}\nAction: "
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)
