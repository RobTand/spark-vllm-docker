#!/usr/bin/env python3
"""incremental_probe.py — run the DynaQuant sensitivity probe in shards.

This keeps the live hook state bounded by probing only a subset of transformer
blocks at a time, then merges the per-shard probe artifacts into one final
`probe.pkl`. Activation snapshots are written into one shared cache dir so
later stages can consume them normally.

This does not magically eliminate model residency cost; it does eliminate the
"all layers hooked at once" part of the memory profile and makes long runs
resumable.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import torch

from .sensitivity_probe import (
    load_calibration,
    load_probe_model_and_tokenizer,
    run_probe_pass,
    stage_text_only,
)


def build_layer_shard_regexes(num_hidden_layers: int,
                              layers_per_shard: int,
                              layer_prefix: str = "model.layers") -> list[str]:
    regexes: list[str] = []
    for start in range(0, num_hidden_layers, layers_per_shard):
        end = min(start + layers_per_shard, num_hidden_layers)
        if end - start == 1:
            body = rf"{re.escape(layer_prefix)}\.{start}\."
        else:
            idxs = "|".join(str(i) for i in range(start, end))
            body = rf"{re.escape(layer_prefix)}\.(?:{idxs})\."
        regexes.append(body)
    return regexes


def _merge_nested_counts(dst: dict, src: dict):
    for key, sub in src.items():
        tgt = dst.setdefault(key, {})
        for sk, sv in sub.items():
            tgt[sk] = tgt.get(sk, 0.0) + float(sv)


def merge_probe_pickles(paths: list[Path], output_path: Path):
    merged = None
    merged_stats = {}
    merged_router_counts = {}
    merged_router_totals = defaultdict(int)
    merged_expert_info = {}
    shard_metas = []

    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if merged is None:
            merged = data
        overlap = set(merged_stats) & set(data["stats"])
        if overlap:
            raise ValueError(f"probe shards overlap on {len(overlap)} stats entries")
        merged_stats.update(data["stats"])
        _merge_nested_counts(merged_router_counts, data.get("router_counts", {}))
        for rk, rv in data.get("router_totals", {}).items():
            merged_router_totals[rk] += int(rv)
        merged_expert_info.update(data.get("expert_info", {}))
        shard_metas.append(data.get("meta", {}))

    if merged is None:
        raise ValueError("no probe shards to merge")

    merged["stats"] = merged_stats
    merged["router_counts"] = dict(merged_router_counts)
    merged["router_totals"] = dict(merged_router_totals)
    merged["expert_info"] = merged_expert_info
    merged["meta"] = {
        **merged.get("meta", {}),
        "incremental": True,
        "n_shards": len(paths),
        "shards": shard_metas,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(merged, f)


def load_num_hidden_layers(model_path: str) -> int:
    staged = stage_text_only(model_path)
    cfg_path = Path(staged) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    n = cfg.get("num_hidden_layers")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Could not infer num_hidden_layers from {cfg_path}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ultrachat_200k")
    ap.add_argument("--nsamples", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output", required=True)
    ap.add_argument("--activation-cache-dir", required=True)
    ap.add_argument("--work-dir", required=True,
                    help="Stores shard logs/pickles; safe to resume.")
    ap.add_argument("--layers-per-shard", type=int, default=1)
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=None)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", action="store_false",
                    dest="gradient_checkpointing")
    ap.add_argument("--importance-weighting", action="store_true", default=True)
    ap.add_argument("--no-importance-weighting", action="store_false",
                    dest="importance_weighting")
    ap.add_argument("--unfuse-moe", action="store_true", default=True)
    ap.add_argument("--no-unfuse-moe", action="store_false", dest="unfuse_moe")
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
    Path(args.activation_cache_dir).mkdir(parents=True, exist_ok=True)

    # Persistent runner: load once, calibrate once, sweep shard regexes.
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[incremental] loading model once for {len(shard_regexes)} shards", flush=True)
    _, tokenizer, model, exec_device, load_device_map = load_probe_model_and_tokenizer(
        args.model,
        requested_device=args.device,
        dtype=dtype,
        device_map=args.device_map,
        unfuse_moe=args.unfuse_moe,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    calib = load_calibration(tokenizer, args.dataset, args.nsamples, args.seqlen)
    print(f"[incremental] calibration ready: {tuple(calib.shape)}", flush=True)

    shard_paths = []
    for shard_idx, linear_include in enumerate(shard_regexes):
        shard_path = shard_dir / f"probe_shard_{shard_idx:03d}.pkl"
        shard_paths.append(shard_path)
        if shard_path.exists():
            print(f"[incremental] reuse shard {shard_idx}: {shard_path}", flush=True)
            continue
        print(f"[incremental] shard {shard_idx}: include={linear_include}", flush=True)
        run_probe_pass(
            model=model,
            tokenizer=tokenizer,
            calib=calib,
            model_name=args.model,
            dataset_name=args.dataset,
            seqlen=args.seqlen,
            dtype_name=args.dtype,
            requested_device=args.device,
            load_device_map=load_device_map,
            exec_device=exec_device,
            linear_include=linear_include,
            linear_exclude=r"(?:^lm_head$|\.lm_head$|mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|block_sparse_moe\.gate$)",
            importance_weighting=args.importance_weighting,
            activation_cache_dir=args.activation_cache_dir,
            output_path=str(shard_path),
        )
        if exec_device.type == "cuda":
            torch.cuda.empty_cache()

    merge_probe_pickles(shard_paths, Path(args.output))
    print(f"[incremental] wrote merged probe to {args.output}", flush=True)


if __name__ == "__main__":
    main()
