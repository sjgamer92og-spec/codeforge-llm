#!/bin/bash
# CodeForge-LLM Training Scripts
# Usage: bash scripts/train.sh [7b|13b|30b]

set -e

MODEL_SIZE=${1:-"7b"}
CONFIG_FILE=${2:-"config.yaml"}

echo "=================================================="
echo "  CodeForge-LLM Training Launcher"
echo "  Model size: ${MODEL_SIZE}"
echo "=================================================="

# Detect GPU and set appropriate settings
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 || echo "0")
echo "Detected GPU VRAM: ${VRAM} MB"

# Model-specific configurations
case $MODEL_SIZE in
    7b)
        BASE_MODEL="deepseek-ai/deepseek-coder-6.7b-base"
        BATCH_SIZE=2
        GRAD_ACC=4
        MAX_SEQ=2048
        LORA_R=64
        ;;
    13b)
        BASE_MODEL="deepseek-ai/deepseek-coder-33b-base"
        BATCH_SIZE=1
        GRAD_ACC=8
        MAX_SEQ=1024
        LORA_R=32
        ;;
    30b)
        BASE_MODEL="deepseek-ai/deepseek-coder-33b-base"
        BATCH_SIZE=1
        GRAD_ACC=16
        MAX_SEQ=512
        LORA_R=16
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        echo "Usage: bash scripts/train.sh [7b|13b|30b]"
        exit 1
        ;;
esac

# Adjust for low VRAM
if [ "$VRAM" -lt 12000 ]; then
    echo "Low VRAM detected, reducing settings..."
    BATCH_SIZE=1
    GRAD_ACC=8
    MAX_SEQ=1024
fi

echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACC"
echo "  Max sequence length: $MAX_SEQ"
echo "  LoRA rank: $LORA_R"
echo ""

# Create output directory
OUTPUT_DIR="./checkpoints/codeforge-${MODEL_SIZE}"
mkdir -p "$OUTPUT_DIR"

# Run training
echo "Starting training..."
python train.py \
    --config "$CONFIG_FILE" \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=================================================="
echo "  Training complete!"
echo "  Model saved to: $OUTPUT_DIR"
echo "=================================================="
