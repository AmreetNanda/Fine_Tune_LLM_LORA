# =========================
# Base Image (CUDA-enabled)
# =========================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# -------------------------
# Environment variables
# -------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/huggingface
ENV TRANSFORMERS_CACHE=/models/huggingface
ENV TORCH_HOME=/models/torch
ENV BNB_CUDA_VERSION=121

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Python setup
# -------------------------
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip

# -------------------------
# Install PyTorch (CUDA)
# -------------------------
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# -------------------------
# Install core libraries
# -------------------------
RUN pip install \
    transformers \
    datasets \
    accelerate \
    bitsandbytes \
    peft \
    flask \
    evaluate \
    sentencepiece \
    scikit-learn \
    rouge-score \
    nltk \
    tqdm

# -------------------------
# (Optional) Ollama CLI
# -------------------------
RUN curl -fsSL https://ollama.com/install.sh | sh || true

# -------------------------
# Set working directory
# -------------------------
WORKDIR /app

# -------------------------
# Copy project files
# -------------------------
COPY . /app

# -------------------------
# Create runtime directories
# -------------------------
RUN mkdir -p \
    /app/offload_dir \
    /app/models \
    /app/logs \
    /models/huggingface

# -------------------------
# Expose Flask port
# -------------------------
EXPOSE 5000

# -------------------------
# Default command
# -------------------------
CMD ["python", "deployment/serve_model.py"]
