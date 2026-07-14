#!/usr/bin/env python3
"""
CodeForge-LLM Web UI
A beautiful Gradio interface for chatting with your fine-tuned model.

Usage:
    python web_ui.py --model ./checkpoints/codeforge-llm/final
    python web_ui.py --model ./checkpoints/codeforge-llm/merged --share
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel
from threading import Thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSS for dark theme styling
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --bg-hover: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #334155;
    --success: #22c55e;
    --code-bg: #0d1117;
}

* {
    font-family: 'Inter', sans-serif !important;
}

body {
    background: var(--bg-dark) !important;
}

.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    text-align: center;
    padding: 20px 0 30px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}

.header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.header p {
    color: var(--text-secondary);
    font-size: 0.95rem;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

.status-online {
    background: rgba(34, 197, 94, 0.15);
    color: var(--success);
}

.status-online::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.chat-container {
    background: var(--bg-card);
    border-radius: 12px;
    border: 1px solid var(--border);
    overflow: hidden;
}

.message.user {
    background: var(--primary) !important;
    color: white !important;
    border-radius: 12px 12px 2px 12px !important;
}

.message.bot {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    border-radius: 12px 12px 12px 2px !important;
}

.message.bot pre {
    background: var(--code-bg) !important;
    border-radius: 8px;
    padding: 12px;
    overflow-x: auto;
}

.message.bot code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85em;
}

.input-area {
    padding: 16px;
    background: var(--bg-card);
    border-top: 1px solid var(--border);
}

.btn-primary {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

.btn-primary:hover {
    background: var(--primary-dark) !important;
    transform: translateY(-1px);
}

.btn-secondary {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
}

.settings-panel {
    background: var(--bg-card);
    border-radius: 12px;
    border: 1px solid var(--border);
    padding: 20px;
}

.settings-panel h3 {
    color: var(--text-primary);
    margin-bottom: 16px;
    font-size: 1rem;
}

.slider-label {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-bottom: 4px;
}

.footer {
    text-align: center;
    padding: 20px;
    color: var(--text-secondary);
    font-size: 0.8rem;
}
"""

SYSTEM_PROMPTS = {
    "Code Expert": "You are CodeForge, an expert programming assistant. You write clean, efficient, and well-documented code. You explain complex concepts clearly and provide best practices.",
    "Debug Helper": "You are a debugging expert. Analyze code, identify bugs, explain the root cause, and provide fixed solutions with explanations.",
    "Teacher": "You are a patient programming teacher. Explain concepts step by step with examples. Adapt your explanations to the user's level.",
    "Code Reviewer": "You are a senior code reviewer. Analyze code for bugs, security issues, performance problems, and style violations. Provide actionable feedback.",
}


class CodeForgeUI:
    def __init__(self, model_path: str, load_4bit: bool = True):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.load_4bit = load_4bit
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from: {self.model_path}")
        
        is_lora = (Path(self.model_path) / "adapter_config.json").exists()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        bnb_config = None
        if self.load_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        if is_lora:
            with open(Path(self.model_path) / "adapter_config.json") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", self.model_path)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            self.model = self.model.merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        
        self.model.eval()
        logger.info("Model loaded successfully!")
    
    def format_prompt(self, message: str, history: List[Tuple[str, str]], system: str) -> str:
        """Format conversation history into prompt."""
        prompt = f"### System:\n{system}\n\n"
        for human, assistant in history:
            prompt += f"### Instruction:\n{human}\n\n### Response:\n{assistant}\n\n"
        prompt += f"### Instruction:\n{message}\n\n### Response:\n"
        return prompt
    
    def generate(
        self,
        message: str,
        history: List[Tuple[str, str]],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """Generate response with streaming."""
        prompt = self.format_prompt(message, history, system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        partial_message = ""
        for new_text in streamer:
            partial_message += new_text
            yield partial_message
        
        thread.join()


def create_ui(model_path: str, share: bool = False):
    """Create and launch the Gradio UI."""
    codeforge = CodeForgeUI(model_path)
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div class="header">
            <h1>⚡ CodeForge-LLM</h1>
            <p>Your personal AI coding assistant — fine-tuned for professional software engineering</p>
            <div style="margin-top: 12px;">
                <span class="status-badge status-online">Model Online</span>
            </div>
        </div>
        """)
        
        with gr.Row():
            # Chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    bubble_full_width=False,
                    show_copy_button=True,
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=codeforge"),
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask me to write code, explain concepts, debug, or review...",
                        show_label=False,
                        container=False,
                        scale=8,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
            
            # Settings panel
            with gr.Column(scale=1):
                gr.HTML('<div class="settings-panel">')
                gr.Markdown("### ⚙️ Settings")
                
                system_select = gr.Dropdown(
                    choices=list(SYSTEM_PROMPTS.keys()),
                    value="Code Expert",
                    label="Assistant Mode",
                )
                system_text = gr.Textbox(
                    value=SYSTEM_PROMPTS["Code Expert"],
                    label="System Prompt",
                    lines=4,
                )
                
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="Max Tokens",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-p",
                )
                gr.HTML('</div>')
        
        gr.HTML("""
        <div class="footer">
            <p>Powered by CodeForge-LLM | Built with Gradio & HuggingFace</p>
        </div>
        """)
        
        # Update system prompt when dropdown changes
        def update_system(mode):
            return SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["Code Expert"])
        
        system_select.change(
            update_system,
            inputs=system_select,
            outputs=system_text,
        )
        
        # Chat functionality
        def user_message(user_input, history):
            return "", history + [[user_input, None]]
        
        def bot_response(history, system, max_tok, temp, top_p_val):
            user_message_text = history[-1][0]
            history[-1][1] = ""
            for partial_resp in codeforge.generate(
                user_message_text,
                history[:-1],
                system,
                max_tok,
                temp,
                top_p_val,
            ):
                history[-1][1] = partial_resp
                yield history
        
        msg.submit(
            user_message,
            [msg, chatbot],
            [msg, chatbot],
            queue=False,
        ).then(
            bot_response,
            [chatbot, system_text, max_tokens, temperature, top_p],
            chatbot,
        )
        
        submit_btn.click(
            user_message,
            [msg, chatbot],
            [msg, chatbot],
            queue=False,
        ).then(
            bot_response,
            [chatbot, system_text, max_tokens, temperature, top_p],
            chatbot,
        )
        
        clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    demo.queue()
    demo.launch(
        share=share,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )


def main():
    parser = argparse.ArgumentParser(description="CodeForge-LLM Web UI")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--no_quantization", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()
    
    create_ui(args.model, share=args.share)


if __name__ == "__main__":
    main()
