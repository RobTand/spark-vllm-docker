#!/usr/bin/env python3
"""
Quality evaluation for quantized models using lm-eval.
Runs gsm8k_cot and ifeval benchmarks.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_lm_eval(
    model_path: str,
    output_dir: str,
    tasks: list[str] = ["gsm8k_cot_llama", "ifeval"],
    limit: int = 100,
    batch_size: int = 4,
    device: str = "cuda",
):
    """
    Run lm-eval on a model and save results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Running {task} on {model_path}")
        print(f"{'='*60}\n")

        output_file = output_path / f"{task}_results.json"

        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
            "--tasks", task,
            "--limit", str(limit),
            "--batch_size", str(batch_size),
            "--device", device,
            "--output_path", str(output_path),
            "--log_samples",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            print(result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr}", file=sys.stderr)

            # Parse results from output
            if result.returncode == 0:
                # lm_eval outputs results to a JSON file
                result_files = list(output_path.glob(f"*{task}*.json"))
                if result_files:
                    with open(result_files[-1]) as f:
                        task_results = json.load(f)
                    results[task] = task_results
                else:
                    results[task] = {"status": "completed", "output": result.stdout}
            else:
                results[task] = {"status": "failed", "error": result.stderr}

        except subprocess.TimeoutExpired:
            results[task] = {"status": "timeout"}
        except Exception as e:
            results[task] = {"status": "error", "error": str(e)}

    # Save combined results
    combined_file = output_path / "combined_results.json"
    with open(combined_file, "w") as f:
        json.dump({
            "model": model_path,
            "tasks": tasks,
            "limit": limit,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {combined_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model quality with lm-eval")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or HF name")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["gsm8k_cot_llama", "ifeval"],
                        help="Evaluation tasks")
    parser.add_argument("--limit", type=int, default=100,
                        help="Number of samples per task")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")

    args = parser.parse_args()

    run_lm_eval(
        model_path=args.model,
        output_dir=args.output,
        tasks=args.tasks,
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
