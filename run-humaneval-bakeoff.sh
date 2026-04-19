#!/bin/bash
# HumanEval+ bakeoff — runs models sequentially through EvalPlus HumanEval
# Uses 40-worker parallel sharding for speed
# Survives disconnection when run in tmux/nohup

set -e
cd /home/rob/spark-vllm-docker
LOG="bakeoff-results/humaneval-bakeoff.log"
mkdir -p bakeoff-results
exec > >(tee -a "$LOG") 2>&1

echo "========================================="
echo "HumanEval+ Bakeoff — $(date)"
echo "========================================="

wait_for_server() {
    echo "Waiting for server (up to 20 min)..."
    for i in $(seq 1 240); do
        if curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null; then
            echo "Server ready!"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Server didn't come up"
    return 1
}

run_humaneval() {
    local label="$1"
    local recipe="$2"
    local result_dir="evalplus_results/${label}"

    echo ""
    echo "========================================="
    echo "BENCHMARK: $label"
    echo "Recipe: $recipe"
    echo "Time: $(date)"
    echo "========================================="

    # Launch model
    docker rm -f vllm_node 2>/dev/null || true
    sleep 2
    python3 run-recipe.py "$recipe" --solo -d

    if ! wait_for_server; then
        echo "FAILED: Server didn't start for $label"
        mkdir -p "bakeoff-results/$label"
        docker logs vllm_node 2>&1 | tail -100 > "bakeoff-results/$label/crash.log"
        return 1
    fi

    # Extra settle time for model warmup
    sleep 10

    # Clear previous results
    rm -rf "$result_dir"

    # Run 40 parallel shards
    echo "Running HumanEval (40 parallel workers, 4096 max tokens)..."
    python3 << PYEOF
import subprocess, os, json, glob
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_shard(i):
    shard_dir = f"${result_dir}/shard_{i:03d}"
    os.makedirs(f"{shard_dir}/humaneval", exist_ok=True)
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "dummy"
    try:
        r = subprocess.run(
            ["python3", "-m", "evalplus.codegen",
             "\$(curl -s http://localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)[\"data\"][0][\"id\"])')",
             "humaneval",
             "--backend", "openai",
             "--base-url", "http://localhost:8000/v1",
             "--greedy",
             "--id-range", f"[{i},{i+1}]",
             "--root", shard_dir],
            env=env, capture_output=True, timeout=600
        )
        return i, r.returncode
    except subprocess.TimeoutExpired:
        return i, -1

done = 0
failed = []
with ProcessPoolExecutor(max_workers=40) as pool:
    futures = {pool.submit(run_shard, i): i for i in range(164)}
    for f in as_completed(futures):
        try:
            i, rc = f.result()
        except Exception:
            i = futures[f]
            rc = -1
        done += 1
        if rc != 0:
            failed.append(i)
        if done % 20 == 0 or done == 164:
            print(f"  {done}/164 done ({len(failed)} failed)", flush=True)

# Retry failed ones
if failed:
    print(f"Retrying {len(failed)} failed problems...")
    for i in failed[:]:
        shard_dir = f"${result_dir}/shard_{i:03d}_retry"
        os.makedirs(f"{shard_dir}/humaneval", exist_ok=True)
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "dummy"
        try:
            r = subprocess.run(
                ["python3", "-m", "evalplus.codegen",
                 "\$(curl -s http://localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)[\"data\"][0][\"id\"])')",
                 "humaneval",
                 "--backend", "openai",
                 "--base-url", "http://localhost:8000/v1",
                 "--greedy",
                 "--id-range", f"[{i},{i+1}]",
                 "--root", shard_dir],
                env=env, capture_output=True, timeout=600
            )
            if r.returncode == 0:
                failed.remove(i)
        except:
            pass

# Merge
results = {}
for f in glob.glob('${result_dir}/shard*/humaneval/*_temp_0.0.jsonl'):
    with open(f) as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                results[d['task_id']] = line.strip()

out_dir = '${result_dir}/humaneval'
os.makedirs(out_dir, exist_ok=True)
# Use a generic name for the merged file
out = f'{out_dir}/results_temp_0.0.jsonl'
with open(out, 'w') as f:
    for line in results.values():
        f.write(line + '\n')

print(f"Total: {len(results)}/164")
if failed:
    print(f"Still failed: {failed}")

print("\\nEvaluating...")
r = subprocess.run(["evalplus.evaluate", "--dataset", "humaneval", "--samples", out, "--i-just-wanna-run"], capture_output=True, text=True)
print(r.stdout)
print(r.stderr)

# Save summary
with open("bakeoff-results/${label}/humaneval_summary.txt", "w") as f:
    f.write(f"Label: ${label}\\n")
    f.write(f"Total: {len(results)}/164\\n")
    f.write(f"Failed: {failed}\\n")
    f.write(r.stdout)
PYEOF

    echo "$label HumanEval complete!"
}

# =========================================
# Run all 3 models
# =========================================
run_humaneval "nemotron3-super-nvfp4" "recipes/nemotron-3-super-nvfp4.yaml"
run_humaneval "mistral-small-4-nvfp4" "recipes/mistral-small-4-119b-nvfp4.yaml"
run_humaneval "qwen122b-nvfp4" "recipes/qwen3.5-122b-a10b-nvfp4.yaml"

echo ""
echo "========================================="
echo "ALL BENCHMARKS COMPLETE — $(date)"
echo "========================================="
echo "Results:"
for d in bakeoff-results/*/humaneval_summary.txt; do
    echo ""
    echo "--- $(dirname $d | xargs basename) ---"
    cat "$d"
done
