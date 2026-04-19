#!/bin/bash
# Quick A/B performance test: K=64 CUTLASS tiles vs baseline (K=256)
# Uses Qwen3.5-35B-A3B-NVFP4 (smallest available NVFP4 MoE model)
#
# Test methodology:
#   1. Run with patched CUTLASS headers + FlashInfer K=64 dispatch
#   2. Run baseline (stock FlashInfer K=256)
#   3. Compare single-user decode and prefill throughput

set -euo pipefail

IMAGE="vllm-nvfp4-k64-tf5:latest"
MODEL="/models/Qwen3.5-35B-A3B-NVFP4"
PORT=8199
RESULTS_DIR="/home/rob/spark-vllm-docker/bakeoff-results"
mkdir -p "$RESULTS_DIR"

# CUTLASS header paths (inside container)
FI_CUTLASS="/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include"
FI_MOE_DSL="/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell"

# Patched CUTLASS headers on host
CUTLASS_SRC="/home/rob/cutlass/include"

# FlashInfer files that need mma_inst_tile_k patched
FI_FILES=(
  "blockscaled_contiguous_grouped_gemm.py"
  "blockscaled_contiguous_grouped_gemm_finalize_fusion.py"
  "blockscaled_contiguous_grouped_gemm_swiglu_fusion.py"
  "blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py"
)

wait_for_server() {
    local port=$1
    local timeout=600
    local start=$(date +%s)
    echo "Waiting for vLLM server on port $port..."
    while ! curl -s "http://localhost:$port/health" > /dev/null 2>&1; do
        sleep 2
        local now=$(date +%s)
        if (( now - start > timeout )); then
            echo "ERROR: Server failed to start within ${timeout}s"
            return 1
        fi
    done
    echo "Server ready ($(( $(date +%s) - start ))s)"
}

run_benchmark() {
    local label=$1
    echo ""
    echo "=== Benchmark: $label ==="
    echo ""

    # Single-user decode: 128 input, 256 output
    echo "--- Decode (128 in / 256 out) ---"
    curl -s "http://localhost:$PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "prompt": "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational time dilation, and the equivalence principle. Include mathematical formulations where appropriate.",
            "max_tokens": 256,
            "temperature": 0,
            "stream": false
        }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
u = r.get('usage', {})
toks = u.get('completion_tokens', 0)
# Extract timing from response if available
print(f'  Completion tokens: {toks}')
print(f'  Response: OK')
" 2>/dev/null || echo "  Request failed"

    # Benchmark with timing: 3 runs
    for run in 1 2 3; do
        echo "--- Run $run: Timed decode ---"
        start=$(date +%s%N)
        output=$(curl -s "http://localhost:$PORT/v1/completions" \
            -H "Content-Type: application/json" \
            -d '{
                "model": "'"$MODEL"'",
                "prompt": "Write a comprehensive guide to building distributed systems.",
                "max_tokens": 512,
                "temperature": 0,
                "stream": false
            }')
        end=$(date +%s%N)
        elapsed_ms=$(( (end - start) / 1000000 ))
        toks=$(echo "$output" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")
        if [ "$toks" -gt 0 ] 2>/dev/null; then
            tps=$(python3 -c "print(f'{$toks / ($elapsed_ms / 1000):.1f}')")
            echo "  ${toks} tokens in ${elapsed_ms}ms = ${tps} tok/s"
        else
            echo "  Failed (${elapsed_ms}ms)"
        fi
    done
}

stop_server() {
    echo "Stopping server..."
    docker stop vllm-k64-test 2>/dev/null || true
    docker rm vllm-k64-test 2>/dev/null || true
    sleep 2
}

VLLM_CMD="vllm serve $MODEL \
    --moe-backend flashinfer_cutedsl \
    --max-model-len 4096 \
    --port $PORT --host 0.0.0.0 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.7 \
"

# ============================
# TEST 1: K=64 (patched)
# ============================
echo "============================================"
echo "  TEST 1: K=64 patched CUTLASS + FlashInfer"
echo "============================================"

stop_server

# Create patch script for FlashInfer's mma_inst_tile_k
PATCH_SCRIPT='
import re, glob
for f in glob.glob("/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell/blockscaled_*.py"):
    with open(f) as fh: src = fh.read()
    if "mma_inst_tile_k = 4" in src:
        src = src.replace("mma_inst_tile_k = 4", "mma_inst_tile_k = 1")
        with open(f, "w") as fh: fh.write(src)
        print(f"Patched: {f}")
'

# Clear FlashInfer JIT cache and patch
docker run -d --name vllm-k64-test \
    --gpus all \
    --shm-size=64g \
    -v /models:/models:ro \
    -v "$CUTLASS_SRC/cute/atom/copy_traits_sm90_tma.hpp:$FI_CUTLASS/cute/atom/copy_traits_sm90_tma.hpp:ro" \
    -v "$CUTLASS_SRC/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl:$FI_CUTLASS/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl:ro" \
    -v "$CUTLASS_SRC/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp:$FI_CUTLASS/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp:ro" \
    -v "$CUTLASS_SRC/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp:$FI_CUTLASS/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp:ro" \
    -p $PORT:$PORT \
    --entrypoint bash \
    "$IMAGE" -c "
        # Clear FlashInfer JIT cache
        rm -rf /root/.cache/flashinfer_jit/ 2>/dev/null
        # Patch FlashInfer tile_k
        python3 -c '$PATCH_SCRIPT'
        # Start vLLM
        $VLLM_CMD
    "

wait_for_server $PORT
run_benchmark "K=64 (patched)" 2>&1 | tee "$RESULTS_DIR/k64-patched.txt"
stop_server

# ============================
# TEST 2: Baseline (stock K=256)
# ============================
echo ""
echo "============================================"
echo "  TEST 2: Baseline (stock K=256)"
echo "============================================"

docker run -d --name vllm-k64-test \
    --gpus all \
    --shm-size=64g \
    -v /models:/models:ro \
    -p $PORT:$PORT \
    --entrypoint bash \
    "$IMAGE" -c "
        rm -rf /root/.cache/flashinfer_jit/ 2>/dev/null
        $VLLM_CMD
    "

wait_for_server $PORT
run_benchmark "Baseline (K=256)" 2>&1 | tee "$RESULTS_DIR/k256-baseline.txt"
stop_server

echo ""
echo "============================================"
echo "  RESULTS COMPARISON"
echo "============================================"
echo ""
echo "K=64 patched:"
grep "tok/s" "$RESULTS_DIR/k64-patched.txt" 2>/dev/null || echo "  (no results)"
echo ""
echo "Baseline (K=256):"
grep "tok/s" "$RESULTS_DIR/k256-baseline.txt" 2>/dev/null || echo "  (no results)"
