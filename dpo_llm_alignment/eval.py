"""
Evaluation Script
=================
Computes preference accuracy and length bias.
This is NOT in the original tutorial — research-grade addition.
"""
import os
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from data_utils import load_preference_dataset

def compute_log_prob(model, tokenizer, prompt, response, max_length=512):
    """Compute average log P(response | prompt). Higher = better."""
    text = prompt + response
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    labels = inputs["input_ids"].clone()
    
    # Mask prompt tokens: only compute loss on response tokens
    prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    prompt_len = prompt_tokens.shape[1]
    labels[:, :prompt_len] = -100
    
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    
    # outputs.loss is mean CE over non-ignored tokens; negate to get log-likelihood
    if outputs.loss is not None:
        return -outputs.loss.item()
    return -999.0

def evaluate(model, tokenizer, dataset, n_samples=200):
    accs = []
    margins = []
    chosen_lens = []
    rejected_lens = []
    
    model.eval()
    for i, ex in enumerate(dataset):
        if i >= n_samples:
            break
        
        c_lp = compute_log_prob(model, tokenizer, ex["prompt"], ex["chosen"])
        r_lp = compute_log_prob(model, tokenizer, ex["prompt"], ex["rejected"])
        
        accs.append(1.0 if c_lp > r_lp else 0.0)
        margins.append(c_lp - r_lp)
        chosen_lens.append(len(tokenizer.encode(ex["chosen"])))
        rejected_lens.append(len(tokenizer.encode(ex["rejected"])))
    
    return {
        "preference_accuracy": np.mean(accs) * 100,
        "mean_reward_margin": np.mean(margins),
        "chosen_avg_length": np.mean(chosen_lens),
        "rejected_avg_length": np.mean(rejected_lens),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./output/dpo_results/final_model")
    parser.add_argument("--base_model", default="./Qwen2.5-0.5B-Instruct",
                        help="Base model path if loading LoRA adapter")
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()
    
    print(f"Loading model from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Detect LoRA adapter vs full model
    is_adapter = os.path.exists(os.path.join(args.model, "adapter_config.json"))
    
    if is_adapter:
        if not os.path.exists(args.base_model):
            print(f"[ERROR] Base model not found: {args.base_model}")
            print("Please run: python download_model.py")
            return
        print(f"[INFO] Loading LoRA adapter + base model...")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
    
    _, eval_ds = load_preference_dataset(n_train=0, n_test=args.n_samples)
    
    print(f"\nEvaluating on {min(args.n_samples, len(eval_ds))} samples ...")
    metrics = evaluate(model, tokenizer, eval_ds, n_samples=args.n_samples)
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.3f}")
    print("="*40)

if __name__ == "__main__":
    main()