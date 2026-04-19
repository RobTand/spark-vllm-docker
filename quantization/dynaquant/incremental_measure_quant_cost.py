#!/usr/bin/env python3
"""incremental_measure_quant_cost.py — shard cost measurement and merge outputs.

Runs `measure_quant_cost.py` against subsets of the probe/cost namespace so the
measurement stage becomes resumable and bounded, mirroring incremental_probe.
"""
from __future__ import annotations

import argparse
import pickle
import re
import subprocess
import sys
from pathlib import Path

from .incremental_probe import build_layer_shard_regexes, load_num_hidden_layers


def merge_cost_pickles(paths: list[Path], output_path: Path):
    merged_costs = {}
    merged_formats = None
    shard_metas = []
    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        costs = data["costs"]
        overlap = set(merged_costs) & set(costs)
        if overlap:
            raise ValueError(f"cost shards overlap on {len(overlap)} entries")
        merged_costs.update(costs)
        if merged_formats is None:
            merged_formats = data.get("formats", [])
        shard_metas.append(data.get("meta", {}))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "costs": merged_costs,
            "formats": merged_formats or [],
            "meta": {
                "incremental": True,
                "n_shards": len(paths),
                "shards": shard_metas,
            },
        }, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--activation-cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--formats", default="")
    ap.add_argument("--skip-missing-activations", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mode", choices=["auto", "batched", "unbatched"], default="auto")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--no-unfuse-moe", action="store_false", dest="unfuse_moe", default=True)
    ap.add_argument("--swap-grow-limit-mb", type=int, default=256)
    ap.add_argument("--min-mem-available-mb", type=int, default=2048)
    ap.add_argument("--no-watchdog", action="store_true")
    ap.add_argument("--layers-per-shard", type=int, default=1)
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=None)
    args = ap.parse_args()

    n_layers = load_num_hidden_layers(args.model)
    start = max(0, args.start_layer)
    end = n_layers if args.end_layer is None else min(args.end_layer, n_layers)
    if start >= end:
        raise SystemExit(f"empty layer range: start={start} end={end}")

    all_regexes = build_layer_shard_regexes(n_layers,
                                            args.layers_per_shard,
                                            layer_prefix="model.layers")
    first_shard = start // args.layers_per_shard
    last_shard = (end + args.layers_per_shard - 1) // args.layers_per_shard
    shard_regexes = all_regexes[first_shard:last_shard]

    work_dir = Path(args.work_dir)
    shard_dir = work_dir / "shards"
    log_dir = work_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[Path] = []
    for shard_idx, linear_include in enumerate(shard_regexes):
        shard_path = shard_dir / f"cost_shard_{shard_idx:03d}.pkl"
        shard_log = log_dir / f"cost_shard_{shard_idx:03d}.log"
        shard_paths.append(shard_path)
        if shard_path.exists():
            print(f"[incremental-cost] reuse shard {shard_idx}: {shard_path}", flush=True)
            continue

        cmd = [
            sys.executable, "-m", "quantization.dynaquant.measure_quant_cost",
            "--model", args.model,
            "--probe", args.probe,
            "--activation-cache-dir", args.activation_cache_dir,
            "--output", str(shard_path),
            "--device", args.device,
            "--dtype", args.dtype,
            "--mode", args.mode,
            "--chunk-size", str(args.chunk_size),
        ]
        if args.device_map is not None:
            cmd.extend(["--device-map", args.device_map])
        if args.formats:
            cmd.extend(["--formats", args.formats])
        if args.skip_missing_activations:
            cmd.append("--skip-missing-activations")
        if args.threads:
            cmd.extend(["--threads", str(args.threads)])
        if not args.unfuse_moe:
            cmd.append("--no-unfuse-moe")
        if args.no_watchdog:
            cmd.append("--no-watchdog")
        else:
            cmd.extend([
                "--swap-grow-limit-mb", str(args.swap_grow_limit_mb),
                "--min-mem-available-mb", str(args.min_mem_available_mb),
            ])
        # Reuse the probe's stat file but filter targets by a shard-local copy.
        # We keep this simple by generating a reduced probe pickle per shard.
        with open(args.probe, "rb") as f:
            probe = pickle.load(f)
        probe["stats"] = {k: v for k, v in probe["stats"].items() if re.search(linear_include, k)}
        shard_probe = shard_dir / f"probe_subset_{shard_idx:03d}.pkl"
        with open(shard_probe, "wb") as f:
            pickle.dump(probe, f)
        cmd[cmd.index("--probe") + 1] = str(shard_probe)

        print(f"[incremental-cost] shard {shard_idx}: include={linear_include}", flush=True)
        with open(shard_log, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise SystemExit(f"cost shard {shard_idx} failed, see {shard_log}")

    merge_cost_pickles(shard_paths, Path(args.output))
    print(f"[incremental-cost] wrote merged cost to {args.output}", flush=True)


if __name__ == "__main__":
    main()
