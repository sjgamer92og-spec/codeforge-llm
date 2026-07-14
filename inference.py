#!/usr/bin/env python3
"""
CodeForge-LLM Inference Script
Run inference on your fine-tuned model.

Usage:
    # Load base model + LoRA adapter
    python inference.py --model ./checkpoints/codeforge-llm/final --prompt "Write a Python function to sort a list"

    # Load merged model
    python inference.py --model ./checkpoints/codeforge-llm/merged --prompt "Explain recursion"

    # Interactive mode
    python inference.py --model ./checkpoints/codeforge-llm/final --interactive

    # With system prompt
    python inference.py --model ./checkpoints/codeforge-llm/final --prompt "Hello" --system "You are an expert Python programmer."
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel
from threading import Thread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class StopOnTokens(StoppingCriteria):
    """Stop generation on specific tokens."""
    
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def load_model(model_path: str, load_4bit: bool = True, device_map: str = "auto"):
    """Load model and tokenizer."""
    logger.info(f"Loading model from: {model_path}")
    
    # Detect if it's a LoRA adapter or merged model
    is_lora = (Path(model_path) / "adapter_config.json").exists()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Setup quantization
    bnb_config = None
    if load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    if is_lora:
        # Load base model first, then merge LoRA
        import json
        with open(Path(model_path) / "adapter_config.json") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", model_path)
        
        logger.info(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        # Load merged/full model directly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    model.eval()
    logger.info("Model loaded successfully!")
    return model, tokenizer


def format_prompt(prompt: str, system: Optional[str] = None, chat_template: str = "alpaca") -> str:
    """Format prompt with optional system message."""
    if chat_template == "alpaca":
        if system:
            return f"### System:\n{system}\n\n### Instruction:\n{prompt}\n\n### Response:\n"
        return f"### Instruction:\n{prompt}\n\n### Response:\n"
    elif chat_template == "chatml":
        if system:
            formatted = f"<|im_start|>system\n{system}<|im_end|>\n"
        else:
            formatted = ""
        formatted += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return formatted
    else:
        return prompt


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    stream: bool = False,
):
    """Generate text from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    if stream:
        # Streaming generation
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_config["streamer"] = streamer
        
        thread = Thread(target=model.generate, kwargs={**inputs, **generation_config})
        thread.start()
        
        generated_text = ""
        for text in streamer:
            generated_text += text
            yield text
        thread.join()
    else:
        # Non-streaming generation
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
        
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text


def interactive_mode(model, tokenizer, system: Optional[str] = None):
    """Run interactive chat session."""
    print("\n" + "=" * 60)
    print("  CodeForge-LLM Interactive Chat")
    print("  Type 'exit', 'quit', or press Ctrl+C to exit")
    print("=" * 60 + "\n")
    
    conversation_history = ""
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Format prompt with history
            prompt = conversation_history + format_prompt(user_input, system=system)
            
            print("\nCodeForge: ", end="", flush=True)
            response = ""
            for chunk in generate(model, tokenizer, prompt, stream=True):
                print(chunk, end="", flush=True)
                response += chunk
            print("\n")
            
            # Update conversation history
            conversation_history = prompt + response + "\n\n"
            
            # Trim history if too long
            max_history_tokens = 3000
            tokens = tokenizer.encode(conversation_history)
            if len(tokens) > max_history_tokens:
                tokens = tokens[-max_history_tokens:]
                conversation_history = tokenizer.decode(tokens)
    
    except KeyboardInterrupt:
        print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="CodeForge-LLM Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model or LoRA adapter")
    parser.add_argument("--prompt", type=str, help="Prompt for generation")
    parser.add_argument("--system", type=str, help="System prompt")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--no_quantization", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--chat_template", type=str, default="alpaca", choices=["alpaca", "chatml"])
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(
        args.model,
        load_4bit=not args.no_quantization,
    )
    
    if args.interactive:
        interactive_mode(model, tokenizer, system=args.system)
    elif args.prompt:
        prompt = format_prompt(args.prompt, system=args.system, chat_template=args.chat_template)
        print(f"\nPrompt: {args.prompt}\n")
        print("CodeForge: ", end="", flush=True)
        for chunk in generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=True,
        ):
            print(chunk, end="", flush=True)
        print("\n")
    else:
        print("Please provide a prompt with --prompt or use --interactive mode")


if __name__ == "__main__":
    main()
