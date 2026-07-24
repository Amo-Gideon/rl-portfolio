"""SFT training stage using TRL."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from rlhf_pipeline.data.sft_data import load_sft_data
from rlhf_pipeline.utils.config import set_seed
from rlhf_pipeline.utils.logging_utils import setup_logger, WandbTracker
from rlhf_pipeline.utils.checkpoint import save_model_and_tokenizer, save_json, create_symlink


def run_sft(config, output_dir: str):
    """Run supervised fine-tuning with LoRA."""
    logger = setup_logger("SFT", log_file=os.path.join(output_dir, "sft.log"))
    logger.info("=" * 60)
    logger.info("Stage 1: Supervised Fine-Tuning")
    logger.info("=" * 60)

    set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tracker = WandbTracker(
        project=config.training.report_to if config.training.report_to != "none" else "none",
        config={
            "stage": "sft",
            "model": config.model.name,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.num_train_epochs,
            "batch_size": config.training.per_device_train_batch_size,
            "gradient_accumulation": config.training.gradient_accumulation_steps,
        },
        run_name=f"{config.experiment.name}_sft",
    )

    logger.info(f"Loading tokenizer: {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading model: {config.model.name}")
    torch_dtype = getattr(torch, config.model.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch_dtype,
        trust_remote_code=config.model.trust_remote_code,
        device_map="auto",
    )

    if config.model.use_lora:
        logger.info("Applying LoRA...")
        lora_config = LoraConfig(
            r=config.model.lora.r,
            lora_alpha=config.model.lora.lora_alpha,
            lora_dropout=config.model.lora.lora_dropout,
            target_modules=config.model.lora.target_modules,
            bias=config.model.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    logger.info(f"Loading data from source: {config.data.source}")
    train_dataset = load_sft_data(config, tokenizer)
    logger.info(f"Dataset size: {len(train_dataset)}")

    test_prompts = config.evaluation.get("test_prompts", []) if hasattr(config, 'evaluation') else []

    before_responses = []
    if test_prompts:
        logger.info("Generating responses BEFORE fine-tuning...")
        model.eval()
        for prompt in test_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.evaluation.get("max_new_tokens", 100),
                    do_sample=True,
                    temperature=config.evaluation.get("temperature", 0.7),
                    top_p=config.evaluation.get("top_p", 0.9),
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )
            before_responses.append(response)
            logger.info(f"  Q: {prompt[:50]}...")
            logger.info(f"  A: {response[:80]}...")

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        max_seq_length=config.data.max_length,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        report_to="none",
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        max_grad_norm=config.training.max_grad_norm,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
    )

    logger.info("Starting SFT training...")
    train_result = trainer.train()

    final_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else 0.0
    logger.info(f"Training complete. Final loss: {final_loss:.4f}")
    tracker.log({"sft/final_loss": final_loss})

    after_responses = []
    if test_prompts:
        logger.info("Generating responses AFTER fine-tuning...")
        model.eval()
        for i, prompt in enumerate(test_prompts):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.evaluation.get("max_new_tokens", 100),
                    do_sample=True,
                    temperature=config.evaluation.get("temperature", 0.7),
                    top_p=config.evaluation.get("top_p", 0.9),
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )
            after_responses.append(response)
            logger.info(f"  Q: {prompt[:50]}...")
            logger.info(f"  A: {response[:80]}...")
            logger.info(f"  Before: {before_responses[i][:60]}...")

    model_path = os.path.join(output_dir, "sft_model")
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Model saved to: {model_path}")

    try:
        merged_model = model.merge_and_unload()
        merged_path = os.path.join(output_dir, "sft_model_merged")
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        logger.info(f"Merged model saved to: {merged_path}")
    except Exception as e:
        logger.warning(f"Could not save merged model: {e}")

    save_json({
        "final_loss": final_loss,
        "test_prompts": test_prompts,
        "before_responses": before_responses,
        "after_responses": after_responses,
    }, os.path.join(output_dir, "sft_results.json"))

    create_symlink(output_dir, config.experiment.name, config.experiment.output_dir)

    tracker.finish()
    logger.info("SFT stage complete!")
    return model_path