"""SFT training stage."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from rlhf_pipeline.data.sft_data import load_sft_data
from rlhf_pipeline.utils.logging_utils import setup_logger
from rlhf_pipeline.utils.checkpoint import save_model_and_tokenizer, save_json


def run_sft(config, output_dir: str):
    """Run supervised fine-tuning."""
    logger = setup_logger("SFT", log_file=os.path.join(output_dir, "sft.log"))
    logger.info("=" * 60)
    logger.info("Stage 1: Supervised Fine-Tuning")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer and model
    logger.info(f"Loading model: {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FIX: Move model to device immediately after loading
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.torch_dtype),
    ).to(device)

    # Load data
    logger.info(f"Loading data from source: {config.data.source}")
    train_dataset = load_sft_data(config, tokenizer)
    logger.info(f"Dataset size: {len(train_dataset)}")

    # Test before fine-tuning
    test_prompts = config.data.test_prompts or config.evaluation.get("test_prompts", [])
    before_responses = []
    if test_prompts:
        logger.info("Testing before fine-tuning...")
        for prompt in test_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            before_responses.append(response)
            logger.info(f"  Q: {prompt}")
            logger.info(f"  A (before): {response[:80]}...")

    # Configure training
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        max_length=config.data.max_length,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        report_to=config.training.report_to,
        fp16=config.training.fp16,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_ratio=config.training.warmup_ratio,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    train_result = trainer.train()
    logger.info(f"Training complete. Final loss: {train_result.training_loss:.4f}")

    # Test after fine-tuning
    after_responses = []
    if test_prompts:
        logger.info("Testing after fine-tuning...")
        model.eval()
        for i, prompt in enumerate(test_prompts):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            after_responses.append(response)
            logger.info(f"  Q: {prompt}")
            logger.info(f"  A (after): {response[:80]}...")
            logger.info(f"  Before: {before_responses[i][:60]}...")

    # Save
    model_path = save_model_and_tokenizer(model, tokenizer, output_dir, "sft_model")
    logger.info(f"Model saved to: {model_path}")

    save_json({
        "final_loss": train_result.training_loss,
        "test_prompts": test_prompts,
        "before_responses": before_responses,
        "after_responses": after_responses,
    }, os.path.join(output_dir, "sft_results.json"))

    logger.info("SFT stage complete!")
    return model_path