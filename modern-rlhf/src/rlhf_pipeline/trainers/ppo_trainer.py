"""PPO alignment training stage using TRL."""
import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import PPOTrainer as TRLPPOTrainer, PPOConfig as TRLPPOConfig
from rlhf_pipeline.models.reward_model import NeuralRewardModel
from rlhf_pipeline.data.sft_data import load_sft_data
from rlhf_pipeline.utils.config import set_seed
from rlhf_pipeline.utils.logging_utils import setup_logger, WandbTracker
from rlhf_pipeline.utils.checkpoint import save_json, create_symlink
from rlhf_pipeline.utils.metrics import plot_training_curves, save_comparison_results


def generate_responses(model, tokenizer, prompts, max_new_tokens=128, 
                       temperature=0.8, top_p=0.9, top_k=50, device="cuda"):
    """Generate responses for a list of prompts."""
    responses = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=10,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        responses.append(response)

    return responses


def run_ppo(config, output_dir: str):
    """Run PPO alignment using TRL's PPOTrainer."""
    logger = setup_logger("PPO", log_file=os.path.join(output_dir, "ppo.log"))
    logger.info("=" * 60)
    logger.info("Stage 3: PPO Alignment Training")
    logger.info("=" * 60)

    set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tracker = WandbTracker(
        project=config.tracking.project_name if config.tracking.project_name != "none" else "none",
        config={
            "stage": "ppo",
            "model": config.model.base_model_name,
            "learning_rate": config.ppo.learning_rate,
            "kl_coef": config.ppo.kl_coef,
            "num_steps": config.ppo.num_steps,
            "batch_size": config.ppo.batch_size,
        },
        run_name=f"{config.experiment.name}_ppo",
    )

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_name,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    from trl import AutoModelForCausalLMWithValueHead
    sft_path = config.model.policy_model_path
    if os.path.exists(sft_path):
        logger.info(f"Loading SFT model from: {sft_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            sft_path,
            torch_dtype=getattr(torch, config.model.torch_dtype),
            trust_remote_code=config.model.trust_remote_code,
            device_map="auto",
        )
    else:
        logger.warning(f"SFT model not found at {sft_path}. Loading base model.")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_name,
            torch_dtype=getattr(torch, config.model.torch_dtype),
            trust_remote_code=config.model.trust_remote_code,
            device_map="auto",
        )
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

    # Note: LoRA not applied for PPO with ValueHead model

    # Create reference model (frozen copy)
    logger.info("Creating frozen reference model...")
    from trl import AutoModelForCausalLMWithValueHead
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load reward model
    logger.info("Loading reward model...")
    rm_path = config.model.reward_model_path
    if os.path.exists(rm_path):
        reward_model = NeuralRewardModel(rm_path, tokenizer, device=str(device))
        logger.info("Loaded trained neural reward model.")
    else:
        logger.warning("Trained reward model not found. Using fallback scoring.")
        from transformers import AutoModelForSequenceClassification
        fallback_rm = AutoModelForSequenceClassification.from_pretrained(
            config.model.base_model_name,
            num_labels=1,
            torch_dtype=getattr(torch, config.model.torch_dtype),
        ).to(device)
        reward_model = NeuralRewardModel(
            config.model.base_model_name, tokenizer, device=str(device)
        )
        reward_model.model = fallback_rm

    # Load prompts for training
    logger.info("Loading training prompts...")
    if config.data.source == "huggingface":
        from datasets import load_dataset
        dataset = load_dataset(config.data.dataset_name, split=config.data.split)
        if config.data.max_samples:
            dataset = dataset.shuffle(seed=config.experiment.seed).select(
                range(min(config.data.max_samples, len(dataset)))
            )
        prompts = [ex[config.data.prompt_column or "instruction"] for ex in dataset]
    else:
        prompts = [
            "Explain what deep learning is.",
            "Write a bubble sort in Python.",
            "How do I learn programming?",
            "What is artificial intelligence?",
            "Recommend technical books.",
            "How to prepare for a technical interview?",
            "Explain RESTful API.",
            "Write an encouraging message for a student.",
        ] * 100

    # Test prompts
    test_prompts = config.data.test_prompts

    # Generate BEFORE PPO
    logger.info("Generating responses BEFORE PPO...")
    policy_model.eval()
    before_responses = generate_responses(
        policy_model, tokenizer, test_prompts,
        max_new_tokens=config.ppo.max_new_tokens,
        temperature=config.ppo.temperature,
        top_p=config.ppo.top_p,
        top_k=config.ppo.top_k,
        device=device,
    )
    before_rewards = [
        reward_model.score(p, r) for p, r in zip(test_prompts, before_responses)
    ]
    for prompt, response, reward in zip(test_prompts, before_responses, before_rewards):
        logger.info(f"  [{reward:.3f}] {prompt[:50]}... -> {response[:60]}...")

    # Setup TRL PPOConfig
    ppo_config = TRLPPOConfig(
        model_name=config.model.base_model_name,
        learning_rate=config.ppo.learning_rate,
        batch_size=config.ppo.batch_size,
        mini_batch_size=config.ppo.mini_batch_size,
        gradient_accumulation_steps=1,
        ppo_epochs=config.ppo.ppo_epochs,
        cliprange=config.ppo.clip_range,
        cliprange_value=0.2,
        gamma=1.0,
        lam=0.95,
        seed=config.experiment.seed,
        log_with="wandb" if tracker.enabled else None,
    )

    # Prepare dataset for TRL
    from datasets import Dataset
    ppo_data = [{"query": p} for p in prompts[:config.ppo.batch_size * config.ppo.num_steps]]
    ppo_dataset = Dataset.from_list(ppo_data)

    ppo_trainer = TRLPPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=ppo_dataset,
    )

    # Training loop
    logger.info(f"Starting PPO training for {config.ppo.num_steps} steps...")
    stats_history = {
        "rewards": [],
        "kl_divergences": [],
        "policy_losses": [],
        "values": [],
        "response_lengths": [],
    }

    for step in range(config.ppo.num_steps):
        batch_start = (step * config.ppo.batch_size) % len(ppo_dataset)
        batch_end = min(batch_start + config.ppo.batch_size, len(ppo_dataset))
        batch_queries = [ppo_dataset[i]["query"] for i in range(batch_start, batch_end)]

        # Generate responses with current policy
        policy_model.eval()
        batch_responses = generate_responses(
            ppo_trainer.model, tokenizer, batch_queries,
            max_new_tokens=config.ppo.max_new_tokens,
            temperature=config.ppo.temperature,
            top_p=config.ppo.top_p,
            top_k=config.ppo.top_k,
            device=device,
        )

        # Compute rewards
        batch_rewards = [
            reward_model.score(q, r) for q, r in zip(batch_queries, batch_responses)
        ]
        batch_rewards_tensor = [torch.tensor(r, dtype=torch.float32) for r in batch_rewards]

        # Prepare queries for TRL
        batch_query_tensors = [
            tokenizer.encode(q, return_tensors="pt").squeeze() for q in batch_queries
        ]
        batch_response_tensors = [
            tokenizer.encode(r, return_tensors="pt").squeeze() for r in batch_responses
        ]

        # Filter out empty responses before PPO step
        valid_indices = [i for i, r in enumerate(batch_response_tensors) if r.numel() > 0]
        if len(valid_indices) < len(batch_response_tensors):
            logger.warning(f"Filtered {len(batch_response_tensors) - len(valid_indices)} empty responses")
        batch_query_tensors = [batch_query_tensors[i] for i in valid_indices]
        batch_response_tensors = [batch_response_tensors[i] for i in valid_indices]
        batch_rewards_tensor = [batch_rewards_tensor[i] for i in valid_indices]
        
        if len(batch_query_tensors) == 0:
            logger.warning("All responses empty, skipping step")
            continue
        
        # Run PPO step
        stats = ppo_trainer.step(batch_query_tensors, batch_response_tensors, batch_rewards_tensor)

        # Log stats
        avg_reward = sum(batch_rewards) / len(batch_rewards)
        kl = stats.get("objective/kl", 0.0)
        policy_loss = stats.get("ppo/loss/policy", 0.0)
        value_loss = stats.get("ppo/loss/value", 0.0)

        stats_history["rewards"].append(avg_reward)
        stats_history["kl_divergences"].append(float(kl) if hasattr(kl, 'item') else float(kl))
        stats_history["policy_losses"].append(float(policy_loss) if hasattr(policy_loss, 'item') else float(policy_loss))
        stats_history["values"].append(float(value_loss) if hasattr(value_loss, 'item') else float(value_loss))
        stats_history["response_lengths"].append(
            sum(len(r) for r in batch_responses) / len(batch_responses)
        )

        if (step + 1) % config.ppo.log_interval == 0:
            kl_float = float(kl) if hasattr(kl, 'item') else float(kl)
            policy_loss_float = float(policy_loss) if hasattr(policy_loss, 'item') else float(policy_loss)
            value_loss_float = float(value_loss) if hasattr(value_loss, 'item') else float(value_loss)
            logger.info(f"Step {step+1}/{config.ppo.num_steps} | "
                       f"Reward: {avg_reward:.3f} | KL: {kl_float:.4f} | "
                       f"Policy Loss: {policy_loss_float:.4f}")
            tracker.log({
                "ppo/step": step + 1,
                "ppo/reward": avg_reward,
                "ppo/kl": kl_float,
                "ppo/policy_loss": policy_loss_float,
                "ppo/value_loss": value_loss_float,
            })

        # Save checkpoint
        if (step + 1) % config.ppo.save_interval == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint-{step+1}")
            os.makedirs(ckpt_path, exist_ok=True)
            ppo_trainer.model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            logger.info(f"Checkpoint saved to: {ckpt_path}")

    # Generate AFTER PPO
    logger.info("Generating responses AFTER PPO...")
    ppo_trainer.model.eval()
    after_responses = generate_responses(
        ppo_trainer.model, tokenizer, test_prompts,
        max_new_tokens=config.ppo.max_new_tokens,
        temperature=config.ppo.temperature,
        top_p=config.ppo.top_p,
        top_k=config.ppo.top_k,
        device=device,
    )
    after_rewards = [
        reward_model.score(p, r) for p, r in zip(test_prompts, after_responses)
    ]
    for prompt, response, reward in zip(test_prompts, after_responses, after_rewards):
        logger.info(f"  [{reward:.3f}] {prompt[:50]}... -> {response[:60]}...")

    # Compute improvement
    avg_before = sum(before_rewards) / len(before_rewards)
    avg_after = sum(after_rewards) / len(after_rewards)
    improvement = avg_after - avg_before
    logger.info(f"Average reward: {avg_before:.3f} -> {avg_after:.3f} ({improvement:+.3f})")

    # Save final model
    model_path = os.path.join(output_dir, "aligned_model")
    os.makedirs(model_path, exist_ok=True)
    ppo_trainer.model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Aligned model saved to: {model_path}")

    # Save results
    save_comparison_results(
        test_prompts, before_responses, after_responses,
        before_rewards, after_rewards,
        os.path.join(output_dir, "ppo_comparison.json"),
    )

    save_json({
        "stats": stats_history,
        "avg_before_reward": avg_before,
        "avg_after_reward": avg_after,
        "improvement": improvement,
    }, os.path.join(output_dir, "ppo_results.json"))

    # Plot training curves
    plot_path = os.path.join(output_dir, "ppo_training_stats.png")
    plot_training_curves(stats_history, plot_path, title="PPO Training Curves")
    logger.info(f"Training plot saved to: {plot_path}")

    create_symlink(output_dir, config.experiment.name, config.experiment.output_dir)

    tracker.finish()
    logger.info("PPO alignment complete!")
    return model_path