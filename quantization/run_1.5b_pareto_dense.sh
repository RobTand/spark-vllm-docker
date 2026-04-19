#!/bin/bash
# Dense Pareto sweep on Qwen2.5-1.5B using cached AutoRound weights.
# The first invocation runs AutoRound (~28 min), populates cache, runs DPQ.
# Subsequent invocations reuse the cache (~1 min each).
set -e
cd /home/rob/spark-vllm-docker/quantization

export PYTHONPATH=/tmp/auto-round:$PYTHONPATH
MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323
CACHE=/tmp/dpq_cache/qwen25-1.5b-nvfp4-noR-200iter-128s-2048
LOGDIR=/tmp/dpq_1.5b_dense
mkdir -p "$LOGDIR" "$CACHE"

AR_ARGS="--autoround-iters 200 --autoround-nsamples 128 --autoround-seqlen 2048 \
         --autoround-batch-size 4 --autoround-dataset NeelNanda/pile-10k"
DPQ_ARGS="--dpq-steps 150 --dpq-calib-samples 16 --dpq-calib-seqlen 512"

# Dense Pareto sweep — 12 efficiency thresholds
# Lower eff = more aggressive escalation (better quality, higher cost)
EFFS=(0.10 0.15 0.20 0.25 0.35 0.50 0.65 0.80 1.00 1.25 1.50 2.00)

run() {
    local eff=$1
    local label=$(printf "eff%05.2f" "$eff")
    local outdir="/tmp/qwen25-1.5b-dense/${label}"
    mkdir -p "$(dirname "$outdir")"
    echo ""
    echo "=== [$(date '+%H:%M:%S')] $label (eff=$eff) ==="
    python3 dpq_autoround_first.py \
        --model "$MODEL" \
        --output "$outdir" \
        --cache-dir "$CACHE" \
        --min-efficiency "$eff" \
        --no-hadamard \
        $AR_ARGS \
        $DPQ_ARGS \
        > "$LOGDIR/${label}.log" 2>&1
    python3 -c "
import json
with open('$outdir/dpq_autoround_first_manifest.json') as f:
    m = json.load(f)
print(f\"  counts={m['counts']} cost={m['avg_cost_vs_fp4']:.3f} finalKL={m['final_kl']:.5f} gap={m['gap_closure']:.3f}\")
"
}

for eff in "${EFFS[@]}"; do
    run "$eff"
done

echo ""
echo "=== Dense sweep complete ==="
python3 - <<'PYEOF'
import json, glob
rows = []
for mpath in sorted(glob.glob('/tmp/qwen25-1.5b-dense/eff*/dpq_autoround_first_manifest.json')):
    with open(mpath) as f: m = json.load(f)
    rows.append(m)
rows.sort(key=lambda r: r['avg_cost_vs_fp4'])
print(f"{'eff':>6} {'cost':>7} {'final_KL':>10} {'gap':>6} {'fp4/fp8/bf16':>16}")
print("-"*55)
for r in rows:
    c = r['counts']
    print(f"{r['min_efficiency']:>6.2f} {r['avg_cost_vs_fp4']:>7.3f} {r['final_kl']:>10.5f} "
          f"{r['gap_closure']:>6.3f} {c['fp4']}/{c['fp8']}/{c['bf16']:>6}")
PYEOF
