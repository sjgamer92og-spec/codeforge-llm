#!/usr/bin/env python3
"""
CodeForge-LLM Dataset Preparation
Prepare and format datasets for training.

Usage:
    # Prepare from HuggingFace datasets
    python data/prepare.py --output ./data/processed

    # Prepare custom JSON/JSONL files
    python data/prepare.py --custom_data ./data/raw/my_data.jsonl --output ./data/processed

    # Preview dataset without saving
    python data/prepare.py --preview 10
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default code-focused datasets
DEFAULT_DATASETS = [
    {
        "name": "iamtarun/python_code_instructions_18k_alpaca",
        "split": "train",
        "weight": 1.0,
    },
    {
        "name": "TokenBender/code_instructions_122k_alpaca_style",
        "split": "train",
        "weight": 0.5,
        "sample_size": 30000,
    },
    {
        "name": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "weight": 1.0,
    },
    {
        "name": "HuggingFaceH4/Code-Feedback",
        "split": "train",
        "weight": 0.8,
        "sample_size": 20000,
    },
]


def format_alpaca(example: Dict) -> Dict[str, str]:
    """Format example in Alpaca instruction format."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return {"text": text}


def format_code_feedback(example: Dict) -> Dict[str, str]:
    """Format Code-Feedback dataset."""
    question = example.get("question", "")
    answer = example.get("answer", "")
    text = f"### Instruction:\n{question}\n\n### Response:\n{answer}"
    return {"text": text}


def format_custom_jsonl(file_path: str) -> Dataset:
    """Load and format custom JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            # Try common formats
            if "instruction" in item:
                formatted = format_alpaca(item)
            elif "prompt" in item:
                text = item["prompt"]
                if "completion" in item:
                    text += item["completion"]
                formatted = {"text": text}
            elif "text" in item:
                formatted = {"text": item["text"]}
            elif "messages" in item:
                # Chat format
                messages = item["messages"]
                text = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    text += f"### {role.capitalize()}:\n{content}\n\n"
                formatted = {"text": text.strip()}
            else:
                logger.warning(f"Unknown format in line: {line[:100]}")
                continue
            data.append(formatted)
    
    return Dataset.from_list(data)


def load_and_format_dataset(ds_config: Dict) -> Dataset:
    """Load a dataset and format it."""
    name = ds_config["name"]
    split = ds_config.get("split", "train")
    
    logger.info(f"Loading dataset: {name}")
    
    try:
        ds = load_dataset(name, split=split)
    except Exception as e:
        logger.warning(f"Failed to load {name}: {e}")
        return None
    
    # Sample if needed
    sample_size = ds_config.get("sample_size")
    if sample_size and len(ds) > sample_size:
        ds = ds.shuffle(seed=42).select(range(sample_size))
        logger.info(f"Sampled {sample_size} examples from {name}")
    
    # Format based on dataset type
    if "code_feedback" in name.lower():
        ds = ds.map(format_code_feedback, remove_columns=ds.column_names)
    elif "instruction" in name.lower() or "alpaca" in name.lower():
        ds = ds.map(format_alpaca, remove_columns=ds.column_names)
    else:
        # Try to auto-detect format
        if "instruction" in ds.column_names and "output" in ds.column_names:
            ds = ds.map(format_alpaca, remove_columns=ds.column_names)
        elif "prompt" in ds.column_names:
            ds = ds.rename_column("prompt", "text")
        else:
            logger.warning(f"Could not auto-format {name}, columns: {ds.column_names}")
            return None
    
    logger.info(f"Loaded {len(ds)} examples from {name}")
    return ds


def prepare_datasets(
    output_dir: str,
    custom_data: str = None,
    datasets_config: List[Dict] = None,
    test_split: float = 0.05,
    preview: int = 0,
):
    """Prepare and combine all datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_train = []
    all_eval = []
    
    # Load default datasets
    if datasets_config is None:
        datasets_config = DEFAULT_DATASETS
    
    for ds_config in datasets_config:
        ds = load_and_format_dataset(ds_config)
        if ds is None:
            continue
        
        # Split into train/eval
        if test_split > 0:
            split = ds.train_test_split(test_size=test_split, seed=42)
            all_train.append(split["train"])
            all_eval.append(split["test"])
        else:
            all_train.append(ds)
    
    # Load custom data if provided
    if custom_data:
        logger.info(f"Loading custom data: {custom_data}")
        custom_ds = format_custom_jsonl(custom_data)
        
        if test_split > 0:
            split = custom_ds.train_test_split(test_size=test_split, seed=42)
            all_train.append(split["train"])
            all_eval.append(split["test"])
        else:
            all_train.append(custom_ds)
    
    if not all_train:
        logger.error("No datasets were loaded!")
        sys.exit(1)
    
    # Combine all datasets
    logger.info("Combining datasets...")
    train_dataset = concatenate_datasets(all_train)
    
    if preview > 0:
        logger.info(f"\n{'='*60}")
        logger.info("Preview of training data:")
        logger.info(f"{'='*60}")
        for i in range(min(preview, len(train_dataset))):
            text = train_dataset[i]["text"]
            logger.info(f"\n--- Example {i+1} ---")
            logger.info(text[:500] + ("..." if len(text) > 500 else ""))
        logger.info(f"{'='*60}\n")
        return
    
    # Shuffle combined dataset
    train_dataset = train_dataset.shuffle(seed=42)
    
    # Save datasets
    logger.info(f"Saving training dataset: {len(train_dataset)} examples")
    train_dataset.save_to_disk(output_dir / "train")
    
    if all_eval:
        eval_dataset = concatenate_datasets(all_eval).shuffle(seed=42)
        logger.info(f"Saving evaluation dataset: {len(eval_dataset)} examples")
        eval_dataset.save_to_disk(output_dir / "eval")
    
    # Save as JSONL for easy inspection
    train_dataset.to_json(output_dir / "train.jsonl")
    if all_eval:
        eval_dataset.to_json(output_dir / "eval.jsonl")
    
    # Save dataset info
    info = {
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset) if all_eval else 0,
        "avg_length": sum(len(ex["text"]) for ex in train_dataset) / len(train_dataset),
        "datasets": [ds["name"] for ds in datasets_config],
        "custom_data": custom_data,
    }
    
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"{'='*60}")
    logger.info("Dataset preparation complete!")
    logger.info(f"  Train: {info['train_size']} examples")
    logger.info(f"  Eval: {info['eval_size']} examples")
    logger.info(f"  Avg length: {info['avg_length']:.0f} chars")
    logger.info(f"  Saved to: {output_dir}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    parser.add_argument("--output", type=str, default="./data/processed", help="Output directory")
    parser.add_argument("--custom_data", type=str, help="Path to custom JSONL file")
    parser.add_argument("--preview", type=int, default=0, help="Preview N examples without saving")
    parser.add_argument("--test_split", type=float, default=0.05, help="Fraction for evaluation")
    args = parser.parse_args()
    
    prepare_datasets(
        output_dir=args.output,
        custom_data=args.custom_data,
        test_split=args.test_split,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
