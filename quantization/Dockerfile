# Quantization workbench for Qwen3.5 NVFP4 experiments
# Designed for DGX Spark (SM121/GB10)

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install Python and basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git wget curl vim \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CUDA FIRST (required by auto-round build)
RUN pip3 install torch==2.12.0.dev20260405+cu130 \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# Install quantization tools (torch already installed above)
# Use transformers from git for Qwen3.5 support
RUN pip3 install \
    git+https://github.com/huggingface/transformers.git \
    accelerate \
    datasets \
    sentencepiece \
    protobuf \
    safetensors \
    huggingface_hub

RUN pip3 install \
    auto-round \
    git+https://github.com/vllm-project/llm-compressor.git \
    lm-eval

# Reinstall PyTorch CUDA and transformers from git (lm-eval may override)
RUN pip3 install --force-reinstall torch==2.12.0.dev20260405+cu130 \
    --index-url https://download.pytorch.org/whl/nightly/cu130 && \
    pip3 install --force-reinstall git+https://github.com/huggingface/transformers.git

# FlashInfer not needed for quantization - only for inference
# The vllm-node container handles inference with FlashInfer

WORKDIR /workspace

# Mount points:
# /models - for model storage
# /workspace - for scripts and outputs
# /home/rob/.cache/huggingface - for HF cache

CMD ["/bin/bash"]
