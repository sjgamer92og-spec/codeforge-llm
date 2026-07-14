# CodeForge-LLM

> Your personal AI coding assistant — fine-tuned for professional software engineering, code generation, debugging, and technical reasoning.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-red?style=flat&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.38+-yellow?style=flat&logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/QLoRA-4bit-green?style=flat" alt="QLoRA">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat" alt="License">
</p>

## Features

- **QLoRA Training** — Fine-tune 7B-30B models on single GPU (8-24GB VRAM)
- **Code-Optimized** — Trained on 100k+ code instruction examples
- **Multiple Base Models** — DeepSeek Coder, CodeLlama, Qwen2.5-Coder
- **Interactive Web UI** — Beautiful Gradio chat interface with streaming
- **Easy Inference** — CLI tool with streaming and interactive modes
- **Model Merging** — Merge LoRA weights for deployment
- **Docker Support** — Containerized training and inference

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/sjgamer92og-spec/codeforge-llm.git
cd codeforge-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Download and prepare training data
python data/prepare.py --output ./data/processed

# Preview dataset
python data/prepare.py --preview 5

# Add your own data
python data/prepare.py --custom_data ./my_code_examples.jsonl --output ./data/processed
```

### 3. Train Your Model

```bash
# Quick start with default config
python train.py --config config.yaml

# Using launch script (auto-detects GPU)
bash scripts/train.sh 7b

# Custom base model
python train.py --config config.yaml --base_model codellama/CodeLlama-7b-Python
```

Training outputs:
- LoRA adapter: `./checkpoints/codeforge-llm/final/`
- Merged model: `./checkpoints/codeforge-llm/merged/`

### 4. Chat with Your Model

**Web UI (Recommended):**
```bash
python web_ui.py --model ./checkpoints/codeforge-llm/final
# Or with public link
python web_ui.py --model ./checkpoints/codeforge-llm/final --share
```

**CLI:**
```bash
# Single prompt
python inference.py --model ./checkpoints/codeforge-llm/final --prompt "Write a Python function to sort a list"

# Interactive chat
python inference.py --model ./checkpoints/codeforge-llm/final --interactive

# With system prompt
python inference.py --model ./checkpoints/codeforge-llm/final --interactive --system "You are a security expert"
```

### 5. Merge for Deployment

```bash
# Merge LoRA with base model
python merge.py --adapter ./checkpoints/codeforge-llm/final --output ./models/codeforge-merged

# Push to HuggingFace Hub
python merge.py --adapter ./checkpoints/codeforge-llm/final --output ./models/codeforge-merged --push your-username/CodeForge-7B
```

## Hardware Requirements

| GPU VRAM | Model Size | Max Seq Length | Batch Size | Notes |
|----------|-----------|----------------|------------|-------|
| 8 GB | 7B | 512 | 1 | Use `--gradient_accumulation_steps 8` |
| 16 GB | 7B-13B | 1024 | 2 | Optimal for most users |
| 24 GB | 13B | 2048 | 4 | Can train larger models |
| 40 GB+ | 30B+ | 2048+ | 4+ | Multi-GPU recommended |

## Configuration

Edit `config.yaml` to customize training:

```yaml
model:
  base_model: "deepseek-ai/deepseek-coder-6.7b-base"  # Change base model

lora:
  r: 64              # LoRA rank (higher = more capacity)
  lora_alpha: 128    # Scaling factor
  target_modules:    # Which layers to train
    - "q_proj"
    - "v_proj"
    - "o_proj"

training:
  num_train_epochs: 3
  learning_rate: 2.0e-4
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  output_dir: "./checkpoints/codeforge-llm"
```

### Supported Base Models

| Model | Size | Best For |
|-------|------|----------|
| `deepseek-ai/deepseek-coder-6.7b-base` | 6.7B | General coding |
| `deepseek-ai/deepseek-coder-33b-base` | 33B | Complex coding |
| `codellama/CodeLlama-7b-Python` | 7B | Python focus |
| `codellama/CodeLlama-13b-Python` | 13B | Python focus |
| `Qwen/Qwen2.5-Coder-7B` | 7B | Multilingual code |
| `bigcode/starcoder2-7b` | 7B | Fill-in-the-middle |

## Docker

```bash
# Build image
docker build -t codeforge-llm .

# Run training
docker run --gpus all -v $(pwd)/checkpoints:/workspace/checkpoints codeforge-llm \
  python train.py --config config.yaml

# Run web UI
docker run --gpus all -p 7860:7860 -v $(pwd)/checkpoints:/workspace/checkpoints codeforge-llm \
  python web_ui.py --model /workspace/checkpoints/codeforge-llm/final
```

## Project Structure

```
codeforge-llm/
├── config.yaml              # Training configuration
├── train.py                 # Main training script
├── inference.py             # CLI inference tool
├── web_ui.py               # Gradio web interface
├── merge.py                # Model merging utility
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container config
├── data/
│   └── prepare.py          # Dataset preparation
├── scripts/
│   └── train.sh            # Training launch script
└── checkpoints/            # Training outputs
```

## Tips for Best Results

1. **Quality Data > Quantity** — 10k high-quality examples beat 100k mediocre ones
2. **Start Small** — Fine-tune 7B first, then scale up
3. **Monitor Training** — Use Weights & Biases or TensorBoard
4. **Evaluate** — Test on your actual use cases, not just benchmarks
5. **Iterate** — Adjust LoRA rank, learning rate, and data mix based on results

## Monitoring Training

```bash
# Weights & Biases (default)
# Set your API key: export WANDB_API_KEY=your_key

# TensorBoard
pip install tensorboard
tensorboard --logdir ./checkpoints/codeforge-llm

# Or disable logging
training:
  report_to: "none"
```

## License

MIT License — feel free to use for personal and commercial projects.

## Acknowledgments

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [QLoRA](https://github.com/artidoro/qlora) by Dettmers et al.
- [DeepSeek Coder](https://github.com/deepseek-ai/DeepSeek-Coder)

---

<p align="center">
  Built with passion for coding and AI
</p>
