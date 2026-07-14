#!/usr/bin/env python3
"""
CodeForge-LLM Model Merge Script
Merge LoRA adapter weights with the base model for easier deployment.

Usage:
    python merge.py --adapter ./checkpoints/codeforge-llm/final --output ./models/codeforge-merged
    python merge.py --adapter ./checkpoints/codeforge-llm/final --output ./models/codeforge-merged --push sjgamer92og-spec/CodeForge-7B
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_lora_weights(adapter_path: str, output_path: str, push_to_hub: str = None):
    """Merge LoRA adapter with base model and save."""
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("  CodeForge-LLM Model Merger")
    logger.info("=" * 60)
    
    # Load adapter config to get base model name
    import json
    adapter_config_path = adapter_path / "adapter_config.json"
    
    if not adapter_config_path.exists():
        logger.error(f"No adapter_config.json found in {adapter_path}")
        logger.error("Make sure you're pointing to a LoRA adapter directory")
        sys.exit(1)
    
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        logger.error("Could not find base_model_name_or_path in adapter config")
        sys.exit(1)
    
    logger.info(f"Base model: {base_model_name}")
    logger.info(f"Adapter: {adapter_path}")
    logger.info(f"Output: {output_path}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model (in fp16 for merging)
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    
    # Merge and unload
    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    
    # Save merged model
    logger.info(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save model info
    model_info = {
        "base_model": base_model_name,
        "adapter_path": str(adapter_path),
        "merge_date": str(logging.getLogger().handlers[0].baseFilename if logging.getLogger().handlers else "unknown"),
        "torch_dtype": "float16",
    }
    
    with open(output_path / "merge_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Calculate model size
    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Merged model size: {model_size:.0f}M parameters")
    
    # Push to Hub if requested
    if push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub: {push_to_hub}")
        from huggingface_hub import HfApi
        
        api = HfApi()
        api.create_repo(repo_id=push_to_hub, exist_ok=True)
        
        model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)
        logger.info("Model pushed to HuggingFace Hub!")
    
    logger.info("=" * 60)
    logger.info("  Merge complete!")
    logger.info(f"  Merged model saved to: {output_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", type=str, required=True, help="Output directory for merged model")
    parser.add_argument("--push", type=str, default=None, help="Push to HuggingFace Hub (format: username/model-name)")
    args = parser.parse_args()
    
    merge_lora_weights(args.adapter, args.output, args.push)


if __name__ == "__main__":
    main()
