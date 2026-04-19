#!/usr/bin/env python3
"""HumanEval+ bakeoff — benchmark multiple models via EvalPlus.
Uses 40-worker parallel sharding for speed.
Run in tmux/nohup for persistence."""

import subprocess, os, json, glob, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MODELS = [
    ("nemotron3-super-nvfp4", "recipes/nemotron-3-super-nvfp4.yaml"),
    ("mistral-small-4-nvfp4", "recipes/mistral-small-4-119b-nvfp4.yaml"),
    ("qwen122b-nvfp4",        "recipes/qwen3.5-122b-a10b-nvfp4.yaml"),
]

MAX_WORKERS = 40
TIMEOUT_PER_PROBLEM = 600  # 10 minutes
TOTAL_PROBLEMS = 164

os.chdir("/home/rob/spark-vllm-docker")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_model_name():
    """Auto-detect model name from vLLM server."""
    import requests
    r = requests.get("http://localhost:8000/v1/models", timeout=5)
    return r.json()["data"][0]["id"]

def wait_for_server(timeout_min=20):
    log(f"Waiting for server (up to {timeout_min} min)...")
    import requests
    for _ in range(timeout_min * 12):
        try:
            r = requests.get("http://localhost:8000/v1/models", timeout=5)
            name = r.json()["data"][0]["id"]
            log(f"Server ready: {name}")
            return name
        except:
            time.sleep(5)
    log("ERROR: Server didn't come up")
    return None

def run_shard(args):
    i, model_name, result_dir = args
    shard_dir = f"{result_dir}/shard_{i:03d}"
    os.makedirs(f"{shard_dir}/humaneval", exist_ok=True)
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "dummy"
    try:
        r = subprocess.run(
            ["python3", "-m", "evalplus.codegen",
             model_name, "humaneval",
             "--backend", "openai",
             "--base-url", "http://localhost:8000/v1",
             "--greedy",
             "--id-range", f"[{i},{i+1}]",
             "--root", shard_dir],
            env=env, capture_output=True, timeout=TIMEOUT_PER_PROBLEM
        )
        return i, r.returncode
    except subprocess.TimeoutExpired:
        return i, -1
    except Exception as e:
        return i, -2

def run_humaneval(label, recipe):
    log(f"{'='*50}")
    log(f"BENCHMARK: {label}")
    log(f"Recipe: {recipe}")
    log(f"{'='*50}")

    result_dir = f"evalplus_results/{label}"
    summary_dir = f"bakeoff-results/{label}"
    os.makedirs(summary_dir, exist_ok=True)

    # Launch model
    subprocess.run(["docker", "rm", "-f", "vllm_node"], capture_output=True)
    time.sleep(2)
    subprocess.run(["python3", "run-recipe.py", recipe, "--solo", "-d"])

    model_name = wait_for_server()
    if not model_name:
        with open(f"{summary_dir}/humaneval_summary.txt", "w") as f:
            f.write(f"Label: {label}\nFAILED: Server didn't start\n")
        return

    time.sleep(10)  # warmup

    # Clear previous
    subprocess.run(["rm", "-rf", result_dir], capture_output=True)

    # Run 40 parallel shards
    log(f"Running HumanEval ({MAX_WORKERS} parallel workers)...")
    done = 0
    failed = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_shard, (i, model_name, result_dir)): i
                   for i in range(TOTAL_PROBLEMS)}
        for f in as_completed(futures):
            try:
                i, rc = f.result()
            except Exception:
                i = futures[f]
                rc = -1
            done += 1
            if rc != 0:
                failed.append(i)
            if done % 20 == 0 or done == TOTAL_PROBLEMS:
                log(f"  {done}/{TOTAL_PROBLEMS} done ({len(failed)} failed)")

    # Retry failed
    if failed:
        log(f"Retrying {len(failed)} failed problems...")
        retry_failed = []
        for i in failed:
            _, rc = run_shard((i, model_name, f"{result_dir}_retry"))
            if rc != 0:
                retry_failed.append(i)
        failed = retry_failed
        if failed:
            log(f"Still failed after retry: {failed}")

    # Merge results
    results = {}
    for f in glob.glob(f'{result_dir}*/shard*/humaneval/*_temp_0.0.jsonl'):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    results[d['task_id']] = line.strip()

    out_dir = f'{result_dir}/humaneval'
    os.makedirs(out_dir, exist_ok=True)
    out = f'{out_dir}/results_temp_0.0.jsonl'
    with open(out, 'w') as fh:
        for line in results.values():
            fh.write(line + '\n')

    log(f"Total: {len(results)}/{TOTAL_PROBLEMS}")

    # Evaluate
    log("Evaluating...")
    r = subprocess.run(
        ["evalplus.evaluate", "--dataset", "humaneval",
         "--samples", out, "--i-just-wanna-run"],
        capture_output=True, text=True
    )
    eval_output = r.stdout + r.stderr
    print(eval_output)

    # Save summary
    with open(f"{summary_dir}/humaneval_summary.txt", "w") as f:
        f.write(f"Label: {label}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total: {len(results)}/{TOTAL_PROBLEMS}\n")
        f.write(f"Failed problems: {failed}\n")
        f.write(eval_output)

    log(f"{label} complete!")

def main():
    log("HumanEval+ Bakeoff starting")

    for label, recipe in MODELS:
        try:
            run_humaneval(label, recipe)
        except Exception as e:
            log(f"ERROR in {label}: {e}")
            import traceback
            traceback.print_exc()

    log("")
    log("=" * 50)
    log("ALL BENCHMARKS COMPLETE")
    log("=" * 50)
    for f in sorted(glob.glob("bakeoff-results/*/humaneval_summary.txt")):
        label = Path(f).parent.name
        print(f"\n--- {label} ---")
        print(open(f).read())

if __name__ == "__main__":
    main()
