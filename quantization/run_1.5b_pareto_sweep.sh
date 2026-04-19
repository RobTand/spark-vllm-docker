#!/bin/bash
# Pareto sweep: DPQ at 3 efficiency thresholds on Qwen2.5-1.5B
# AutoRound-first pipeline, no rotation (rotation hurt quality on 0.5B)
set -e
cd /home/rob/spark-vllm-docker/quantization

export PYTHONPATH=/tmp/auto-round:$PYTHONPATH
MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323
LOGDIR=/tmp/dpq_1.5b_sweep
mkdir -p "$LOGDIR"

# Common AutoRound hyperparameters — 200 iters is enough for convergence
# (from memory: AutoRound converges at ~148-198 iters, 200 ceiling sufficient)
AR_ARGS="--autoround-iters 200 --autoround-nsamples 128 --autoround-seqlen 2048 \
         --autoround-batch-size 4 --autoround-dataset NeelNanda/pile-10k"
DPQ_ARGS="--dpq-steps 150 --dpq-calib-samples 16 --dpq-calib-seqlen 512"

# 3 efficiency thresholds — permissive → strict
# Lower efficiency = more aggressive escalation (better quality, higher cost)
# Higher efficiency = cheaper (more FP4, fewer FP8/BF16)
run() {
    local label=$1
    local eff=$2
    local outdir="/tmp/qwen25-1.5b-af-${label}"
    echo ""
    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] Run: $label  (min_efficiency=$eff)"
    echo "  output: $outdir"
    echo "================================================================"
    python3 dpq_autoround_first.py \
        --model "$MODEL" \
        --output "$outdir" \
        --min-efficiency "$eff" \
        --no-hadamard \
        $AR_ARGS \
        $DPQ_ARGS \
        > "$LOGDIR/${label}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] $label done, manifest:"
    python3 -c "
import json
with open('$outdir/dpq_autoround_first_manifest.json') as f:
    m = json.load(f)
print('  counts:', m['counts'])
print('  avg_cost_vs_fp4:', round(m['avg_cost_vs_fp4'], 3))
print('  AutoRound FP4 KL:', round(m['autoround_fp4_kl_vs_bf16'], 6))
print('  AutoRound FP8 KL:', round(m['autoround_fp8_kl_vs_bf16'], 6))
print('  final KL:', round(m['final_kl'], 6))
"
}

run permissive 0.25
run balanced 0.5
run strict 1.0

echo ""
echo "================================================================"
echo "[$(date '+%H:%M:%S')] Sweep complete. Logs in $LOGDIR/"
echo "================================================================"
ls -la /tmp/qwen25-1.5b-af-*/dpq_autoround_first_manifest.json
