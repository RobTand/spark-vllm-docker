#!/usr/bin/env python3
"""tiny_bakeoff.py — one-command tiny-model DynaQuant regression bakeoff.

This orchestrates the canonical small-model validation loop:

  1. local_reconstruct
  2. measure_interactions
  3. calibrate_allocator
  4. quadratic_refine_allocator
  5. oracle_search
  6. bakeoff

It is designed to answer one question quickly and consistently:
    "Did the latest change earn its complexity?"

The script assumes a tiny model and precomputed probe/cost/cache by default,
but all paths are overridable.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)
DEFAULT_PROBE = "/tmp/tiny_probe.pkl"
DEFAULT_COSTS = "/tmp/tiny_cost.pkl"
DEFAULT_ACT_CACHE = "/tmp/tiny_act"
DEFAULT_OUTPUT_DIR = "/tmp/dynaquant_tiny_bakeoff"


def _run(cmd: list[str], cwd: str, dry_run: bool):
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"[bakeoff-run] {rendered}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def _paths(out_dir: Path):
    return {
        "costs_refined": out_dir / "costs_refined.pkl",
        "interactions": out_dir / "interactions.json",
        "refined": out_dir / "refined.json",
        "calibration": out_dir / "calibration.json",
        "oracle": out_dir / "oracle.json",
        "decision": out_dir / "decision.json",
    }


def build_bakeoff_commands(args) -> tuple[dict[str, Path], list[list[str]]]:
    out_dir = Path(args.output_dir)
    paths = _paths(out_dir)
    commands = []
    commands.append([
        sys.executable,
        "-m",
        "quantization.dynaquant.local_reconstruct",
        "--model", args.model,
        "--probe", args.probe,
        "--costs", args.costs,
        "--activation-cache-dir", args.activation_cache_dir,
        "--formats", args.formats,
        "--target-bits", str(args.target_bits),
        "--top-units", str(args.top_units),
        "--device", args.device,
        "--dtype", "bf16",
        "--output", str(paths["costs_refined"]),
    ])
    commands.append([
        sys.executable,
        "-m",
        "quantization.dynaquant.measure_interactions",
        "--model", args.model,
        "--probe", args.probe,
        "--costs", str(paths["costs_refined"]),
        "--formats", args.formats,
        "--target-bits", str(args.target_bits),
        "--top-units", str(args.top_units),
        "--neighbor-radius", str(args.neighbor_radius),
        "--n-calib-samples", str(args.n_calib_samples),
        "--calib-seqlen", str(args.calib_seqlen),
        "--device", args.device,
        "--output", str(paths["interactions"]),
    ])
    commands.append([
        sys.executable,
        "-m",
        "quantization.dynaquant.calibrate_allocator",
        "--model", args.model,
        "--probe", args.probe,
        "--costs", str(paths["costs_refined"]),
        "--formats", args.formats,
        "--pareto-targets", f"4.5,{args.target_bits},16.0",
        "--selection", "baseline,knee,high",
        "--n-calib-samples", str(args.n_calib_samples),
        "--calib-seqlen", str(args.calib_seqlen),
        "--device", args.device,
        "--output", str(paths["calibration"]),
    ])
    commands.append([
        sys.executable,
        "-m",
        "quantization.dynaquant.quadratic_refine_allocator",
        "--interactions", str(paths["interactions"]),
        "--calibration", str(paths["calibration"]),
        "--output", str(paths["refined"]),
    ])
    if not args.skip_oracle:
        commands.append([
            sys.executable,
            "-m",
            "quantization.dynaquant.oracle_search",
            "--interactions", str(paths["interactions"]),
            "--model", args.model,
            "--n-calib-samples", str(args.n_calib_samples),
            "--calib-seqlen", str(args.calib_seqlen),
            "--device", args.device,
            "--max-combos", str(args.oracle_max_combos),
            "--output", str(paths["oracle"]),
        ])
    bakeoff_cmd = [
        sys.executable,
        "-m",
        "quantization.dynaquant.bakeoff",
        "--calibration", str(paths["calibration"]),
        "--candidate", "refined",
        "--refined", str(paths["refined"]),
        "--output", str(paths["decision"]),
    ]
    if not args.skip_oracle:
        bakeoff_cmd.extend(["--oracle", str(paths["oracle"])])
    commands.append(bakeoff_cmd)
    return paths, commands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--probe", default=DEFAULT_PROBE)
    ap.add_argument("--costs", default=DEFAULT_COSTS)
    ap.add_argument("--activation-cache-dir", default=DEFAULT_ACT_CACHE)
    ap.add_argument("--formats", default="NVFP4,MXFP8,BF16")
    ap.add_argument("--target-bits", type=float, default=4.8)
    ap.add_argument("--top-units", type=int, default=6)
    ap.add_argument("--neighbor-radius", type=int, default=1)
    ap.add_argument("--n-calib-samples", type=int, default=2)
    ap.add_argument("--calib-seqlen", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--oracle-max-combos", type=int, default=1024)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, commands = build_bakeoff_commands(args)
    cwd = os.getcwd()
    for cmd in commands:
        _run(cmd, cwd, args.dry_run)

    if args.dry_run:
        summary = {
            "output_dir": str(out_dir),
            "paths": {k: str(v) for k, v in paths.items()},
            "oracle_enabled": not args.skip_oracle,
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
