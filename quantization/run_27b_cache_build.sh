#!/bin/bash
# Build AutoRound cache for Qwen3.5-27B.
# Reduced hyperparameters for tractability:
#   iters 100 (vs 200), nsamples 64 (vs 128), seqlen 1024 (vs 2048), batch 2 (vs 4)
# Skip DPQ stage entirely via --cache-only (memory-constrained at 27B).
set -e
cd /home/rob/spark-vllm-docker/quantization

export PYTHONPATH=/tmp/auto-round:$PYTHONPATH
MODEL=/models/Qwen3.5-27B-bf16
CACHE=/tmp/dpq_cache/qwen35-27b-nvfp4-noR-100iter-64s-1024
LOGDIR=/tmp/dpq_27b_cache
mkdir -p "$LOGDIR" "$CACHE"

echo "[$(date '+%H:%M:%S')] starting 27B AutoRound cache build"
run_stage() {
    local stage=$1
    local logfile="$LOGDIR/cache_build_${stage}.log"
    echo "[$(date '+%H:%M:%S')] Stage: $stage"
    python3 dpq_autoround_first.py \
        --model "$MODEL" \
        --output /tmp/qwen35-27b-cache-unused \
        --cache-dir "$CACHE" \
        --cache-only \
        --stages "$stage" \
        --no-hadamard \
        --autoround-iters 100 \
        --autoround-nsamples 32 \
        --autoround-seqlen 1024 \
        --autoround-batch-size 1 \
        --autoround-dataset NeelNanda/pile-10k \
        > "$logfile" 2>&1
    local rc=$?
    echo "[$(date '+%H:%M:%S')] Stage $stage exit=$rc"
    if [ $rc -ne 0 ]; then
        echo "FAILED at stage $stage — see $logfile"
        tail -20 "$logfile"
        exit $rc
    fi
    ls -la "$CACHE/"
}

# Run FP4 and FP8 in separate Python processes — no memory leaks between stages
run_stage fp4
run_stage fp8
echo "[$(date '+%H:%M:%S')] 27B cache build complete (exit=$?)"
ls -la "$CACHE/"
