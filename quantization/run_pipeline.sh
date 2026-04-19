#!/bin/bash
# Full quantization pipeline for Qwen3.5-27B
# Run this from the host: ./run_pipeline.sh
set -e

CONTAINER_IMAGE="quant-workbench"
MODELS_DIR="/models"
HF_CACHE="/home/rob/.cache/huggingface"
WORKSPACE="/home/rob/spark-vllm-docker/quantization"
OUTPUT_BASE="/models/qwen35-27b-quant-experiments"

# Model to quantize
MODEL="Qwen/Qwen3.5-27B"
MODEL_SHORT="qwen35-27b"

# Create output directories
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "Qwen3.5-27B NVFP4 Quantization Pipeline"
echo "=============================================="
echo "Model: $MODEL"
echo "Output: $OUTPUT_BASE"
echo ""

# Helper to run commands in container
run_in_container() {
    docker run --rm --gpus all \
        -v "$MODELS_DIR:/models" \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$WORKSPACE:/workspace" \
        -e HF_HOME=/root/.cache/huggingface \
        -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
        "$CONTAINER_IMAGE" \
        "$@"
}

# Step 1: Download model if not cached
echo "[Step 1] Ensuring model is downloaded..."
run_in_container python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Checking model cache...')
tokenizer = AutoTokenizer.from_pretrained('$MODEL', trust_remote_code=True)
print('Tokenizer ready')
# Just check config, don't load full model yet
from transformers import AutoConfig
config = AutoConfig.from_pretrained('$MODEL', trust_remote_code=True)
print(f'Model config loaded: {config.num_hidden_layers} layers, {config.hidden_size} hidden')
"
echo "[Step 1] Done"
echo ""

# Step 2: Run baseline quality eval (optional - takes time)
if [ "$1" == "--with-baseline" ]; then
    echo "[Step 2] Running baseline quality evaluation..."
    run_in_container python3 /workspace/eval_quality.py \
        --model "$MODEL" \
        --output "/models/$MODEL_SHORT-baseline-eval" \
        --limit 50 \
        --batch-size 2
    echo "[Step 2] Done"
    echo ""
fi

# Step 3: Run sensitivity analysis
echo "[Step 3] Running sensitivity analysis..."
run_in_container python3 /workspace/sensitivity_analysis.py \
    --model "$MODEL" \
    --output "/models/$MODEL_SHORT-sensitivity" \
    --nsamples 64 \
    --seqlen 2048 \
    --batch-size 2
echo "[Step 3] Done"
echo ""

# Step 4: Create quantization variants
echo "[Step 4] Creating quantization variants..."

echo "  [4a] all-fp4 variant..."
run_in_container python3 /workspace/quantize_nvfp4.py \
    --model "$MODEL" \
    --variant all-fp4 \
    --output "$OUTPUT_BASE/$MODEL_SHORT-all-fp4" \
    --nsamples 128 \
    --seqlen 4096

echo "  [4b] fp4-bf16-critical variant (like Sehyo's approach)..."
run_in_container python3 /workspace/quantize_nvfp4.py \
    --model "$MODEL" \
    --variant fp4-bf16-critical \
    --output "$OUTPUT_BASE/$MODEL_SHORT-fp4-bf16-critical" \
    --nsamples 128 \
    --seqlen 4096

echo "  [4c] fp4-fp8-sensitive variant..."
run_in_container python3 /workspace/quantize_nvfp4.py \
    --model "$MODEL" \
    --variant fp4-fp8-sensitive \
    --output "$OUTPUT_BASE/$MODEL_SHORT-fp4-fp8-sensitive" \
    --sensitivity-dir "/models/$MODEL_SHORT-sensitivity" \
    --nsamples 128 \
    --seqlen 4096

echo "[Step 4] Done"
echo ""

# Step 5: Evaluate quality of each variant
echo "[Step 5] Evaluating quality of variants..."

for variant in all-fp4 fp4-bf16-critical fp4-fp8-sensitive; do
    echo "  Evaluating $variant..."
    run_in_container python3 /workspace/eval_quality.py \
        --model "$OUTPUT_BASE/$MODEL_SHORT-$variant" \
        --output "$OUTPUT_BASE/$MODEL_SHORT-$variant-eval" \
        --limit 50 \
        --batch-size 2
done

echo "[Step 5] Done"
echo ""

# Step 6: Summary
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Sensitivity analysis: /models/$MODEL_SHORT-sensitivity/"
echo "  Variants:"
for variant in all-fp4 fp4-bf16-critical fp4-fp8-sensitive; do
    echo "    - $OUTPUT_BASE/$MODEL_SHORT-$variant/"
done
echo ""
echo "Quality results:"
for variant in all-fp4 fp4-bf16-critical fp4-fp8-sensitive; do
    echo "    - $OUTPUT_BASE/$MODEL_SHORT-$variant-eval/"
done
echo ""
echo "Next: Run speed benchmarks with vllm-node container"
