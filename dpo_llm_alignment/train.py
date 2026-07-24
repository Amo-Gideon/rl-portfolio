"""
DPO Training Script
===================
Core DPOTrainer structure adapted from:
    "Hands-On Modern RL" (Walking Labs, 2025)
    https://walkinglabs.github.io/hands-on-modern-rl/en/chapter02_dpo/intro

Modifications:
    - Real preference dataset (Ultrafeedback fallback to mock)
    - LoRA for parameter-efficient training
    - Auto GPU/CPU detection with optional 4-bit quantization
    - WandB logging (optional)
    - Gradient checkpointing for memory efficiency
"""
import os
import sys
import shutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False
    print("[WARN] wandb not installed. Logging disabled.")

from data_utils import load_preference_dataset, load_mock_data
from download_model import download_model

# ==================== CONFIG ====================
LOCAL_MODEL_DIR = "./Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./output/dpo_results"
USE_REAL_DATA = True
N_TRAIN = 2000
N_EVAL = 200
LORA_R = 16
LORA_ALPHA = 32

# ==================== GPU AUTO-CONFIG ====================
def get_device_config():
    if not torch.cuda.is_available():
        print("[INFO] No GPU detected. Training on CPU (very slow).")
        return {"torch_dtype": torch.float32, "device_map": None}
    
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
    
    # Try 4-bit only if bitsandbytes is available (often broken on Windows)
    try:
        from transformers import BitsAndBytesConfig
        if gpu_mem < 6:
            print("[INFO] Attempting 4-bit QLoRA for <6GB VRAM...")
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            return {
                "quantization_config": bnb,
                "torch_dtype": torch.float16,
                "device_map": "auto",
            }
    except Exception as e:
        print(f"[INFO] 4-bit quant skipped ({e}).")
    
    # Standard fp16 + LoRA (should fit 0.5B on 4GB with batch=1, grad_accum=4)
    print("[INFO] Using fp16 + LoRA.")
    return {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }

# ==================== CALLBACKS ====================
class NanStopCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            if loss != loss or abs(loss) > 10:
                print(f"\n[EMERGENCY STOP] Loss unstable: {loss}. Aborting.")
                control.should_training_stop = True
        return control

# ==================== MAIN ====================
def main():
    # 1. Model
    model_path = download_model()
    
    # 2. Data
    if USE_REAL_DATA:
        try:
            train_ds, eval_ds = load_preference_dataset(n_train=N_TRAIN, n_test=N_EVAL)
        except Exception as e:
            print(f"[WARN] Real dataset failed ({e}). Falling back to mock data.")
            USE_REAL_DATA = False
    
    if not USE_REAL_DATA:
        from datasets import Dataset
        raw = load_mock_data(100)
        train_ds = Dataset.from_dict({
            "prompt": [d["prompt"] for d in raw],
            "chosen": [d["chosen"] for d in raw],
            "rejected": [d["rejected"] for d in raw],
        })
        eval_ds = None
    
    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4. Model
    load_kwargs = get_device_config()
    print(f"[INFO] Loading base model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        **load_kwargs,
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False  # Required for gradient checkpointing
    
    # LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 5. Clean output dir
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 6. WandB
    if USE_WANDB:
        wandb.init(project="dpo-llm-alignment", name="qwen0.5b-lora")
        wandb.config.update({
            "model": "Qwen2.5-0.5B-Instruct",
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "n_train": len(train_ds),
            "dataset": "ultrafeedback_binarized" if USE_REAL_DATA else "mock",
        })
    
    # 7. Training args
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        eval_steps=50 if eval_ds else 999999,
        evaluation_strategy="steps" if eval_ds else "no",
        beta=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=load_kwargs.get("torch_dtype") == torch.float16 and torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to=["wandb"] if USE_WANDB else [],
        remove_unused_columns=False,
    )
    
    # 8. Trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=[NanStopCallback()],
    )
    
    # 9. Train
    print("\n" + "="*50)
    print("Starting DPO training...")
    print("="*50)
    trainer.train()
    
    # 10. Save
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
    
    print(f"\n[INFO] Saving model to {final_path} ...")
    trainer.save_model(final_path)
    print(f"[INFO] Saving LoRA adapter to {adapter_path} ...")
    model.save_pretrained(adapter_path)
    
    # Verify
    has_nan = any(
        p.isnan().any() 
        for p in model.parameters() 
        if p.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )
    if has_nan:
        print("\n[FAIL] Model has NaN weights!")
    else:
        print(f"\n[SUCCESS] Saved to {final_path}")
        print(f"[SUCCESS] Adapter saved to {adapter_path}")
        print("\nNext steps:")
        print(f"  python eval.py --model {final_path}")
        print(f"  python inference.py --model {final_path}")

if __name__ == "__main__":
    main()