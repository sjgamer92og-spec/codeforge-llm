#!/usr/bin/env python3
"""
CodeForge-LLM Training Script
Fine-tune large language models for code generation using QLoRA.
Supports 4-bit quantization for training on consumer GPUs (8-24GB VRAM).

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --base_model codellama/CodeLlama-7b-Python
    accelerate launch --num_processes 1 train.py --config config.yaml
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    """Custom callback for logging training progress."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"Step {state.global_step}: {logs}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def setup_quantization_config(config: Dict) -> BitsAndBytesConfig:
    """Configure 4-bit quantization for memory-efficient training."""
    q_config = config.get("quantization", {})
    
    if not q_config.get("load_in_4bit", True):
        return None
    
    compute_dtype = getattr(torch, q_config.get("bnb_4bit_compute_dtype", "bfloat16"))
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=q_config.get("bnb_4bit_use_double_quant", True),
    )
    logger.info(f"4-bit quantization enabled (type: nf4, compute: {compute_dtype})")
    return bnb_config


def load_model_and_tokenizer(config: Dict, bnb_config: Optional[BitsAndBytesConfig]):
    """Load base model and tokenizer with quantization."""
    model_config = config["model"]
    model_name = model_config["base_model"]
    
    logger.info(f"Loading base model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", True),
        use_fast=True,
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with optional quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_config.get("trust_remote_code", True),
        torch_dtype=torch.bfloat16 if config["training"].get("bf16") else torch.float16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model, tokenizer


def setup_lora_config(config: Dict) -> LoraConfig:
    """Configure LoRA for parameter-efficient fine-tuning."""
    lora_config = config["lora"]
    
    lora = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )
    
    logger.info(
        f"LoRA config: r={lora_config['r']}, alpha={lora_config['lora_alpha']}, "
        f"target_modules={lora_config['target_modules']}"
    )
    return lora


def format_alpaca_prompt(example: Dict) -> str:
    """Format example in Alpaca instruction format."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return prompt


def format_chatml_prompt(example: Dict) -> str:
    """Format example in ChatML format."""
    messages = example.get("messages", [])
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted


def prepare_dataset(config: Dict, tokenizer):
    """Load and prepare training and evaluation datasets."""
    dataset_config = config["dataset"]
    template = dataset_config.get("template", "alpaca")
    preprocessing = config["preprocessing"]
    max_length = preprocessing.get("max_seq_length", 2048)
    
    logger.info(f"Loading datasets with template: {template}")
    
    # Load training datasets
    train_datasets = []
    for ds_config in dataset_config.get("train_datasets", []):
        logger.info(f"Loading dataset: {ds_config['name']}")
        ds = load_dataset(ds_config["name"], split=ds_config["split"])
        
        # Sample if specified
        sample_size = ds_config.get("sample_size", -1)
        if sample_size > 0 and len(ds) > sample_size:
            ds = ds.shuffle(seed=42).select(range(sample_size))
            logger.info(f"Sampled {sample_size} examples from {ds_config['name']}")
        
        train_datasets.append(ds)
    
    # Combine training datasets
    if len(train_datasets) > 1:
        train_dataset = concatenate_datasets(train_datasets)
    else:
        train_dataset = train_datasets[0]
    
    logger.info(f"Combined training set size: {len(train_dataset)}")
    
    # Load evaluation dataset
    eval_dataset = None
    if dataset_config.get("eval_datasets"):
        eval_configs = dataset_config["eval_datasets"]
        eval_datasets = []
        for ds_config in eval_configs:
            ds = load_dataset(ds_config["name"], split=ds_config["split"])
            eval_datasets.append(ds)
        eval_dataset = concatenate_datasets(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]
        logger.info(f"Evaluation set size: {len(eval_dataset)}")
    
    # Format function based on template
    def format_prompt(example):
        if template == "alpaca":
            text = format_alpaca_prompt(example)
        elif template == "chatml":
            text = format_chatml_prompt(example)
        elif template == "deepseek":
            # DeepSeek Coder format
            text = example.get("prompt", format_alpaca_prompt(example))
        else:
            text = example.get("text", example.get("prompt", str(example)))
        return {"text": text}
    
    # Apply formatting
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    if eval_dataset:
        eval_dataset = eval_dataset.map(format_prompt, remove_columns=eval_dataset.column_names)
    
    # Tokenization function
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        # Set labels for language modeling (shifted inputs)
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train dataset",
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing eval dataset",
        )
    
    return train_dataset, eval_dataset


def create_training_arguments(config: Dict) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config."""
    t = config["training"]
    
    args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", t["per_device_train_batch_size"]),
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        logging_steps=t["logging_steps"],
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 500),
        evaluation_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 500),
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=t.get("greater_is_better", False),
        bf16=t.get("bf16", True),
        fp16=t.get("fp16", False),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        optim=t.get("optim", "paged_adamw_8bit"),
        group_by_length=t.get("group_by_length", True),
        report_to=t.get("report_to", "none"),
        run_name=t.get("run_name", "codeforge-llm"),
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )
    
    return args


def main():
    parser = argparse.ArgumentParser(description="Train CodeForge-LLM with QLoRA")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--base_model", type=str, default=None, help="Override base model")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.base_model:
        config["model"]["base_model"] = args.base_model
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directory
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    logger.info("=" * 60)
    logger.info("  CodeForge-LLM Training")
    logger.info(f"  Model: {config['model']['base_model']}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)
    
    # Setup quantization
    bnb_config = setup_quantization_config(config)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, bnb_config)
    
    # Prepare model for k-bit training
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)
    
    # Create training arguments
    training_args = create_training_arguments(config)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[LoggingCallback()],
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_output_dir = output_dir / "final"
    logger.info(f"Saving final model to {final_output_dir}")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    # Save merged model (optional, for easier inference)
    logger.info("Saving merged model...")
    merged_output_dir = output_dir / "merged"
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    
    logger.info("Training complete!")
    logger.info(f"LoRA adapter saved to: {final_output_dir}")
    logger.info(f"Merged model saved to: {merged_output_dir}")


if __name__ == "__main__":
    main()
