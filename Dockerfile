# CodeForge-LLM Docker Image
# Supports both training and inference

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories
RUN mkdir -p /workspace/checkpoints /workspace/data /workspace/models

# Expose port for Gradio UI
EXPOSE 7860

# Default command
CMD ["python3", "web_ui.py", "--model", "./checkpoints/codeforge-llm/final"]
