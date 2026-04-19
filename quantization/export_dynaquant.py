#!/usr/bin/env python3
"""
export_dynaquant.py — Export a DynaQuant model at arbitrary bit widths.

Operates directly on safetensors files (no model loading required).
Streams through source shards, quantizes weight-by-weight, and writes
packed output incrementally. Peak memory = 1 tensor at a time.

Output format:
    output_dir/
        config.json, tokenizer.json, etc. (from source)
        dynaquant_config.json  — recipe + metadata
        packed_weights/        — per-layer .packed (uint8) + .scales (fp32)
        passthrough-NNN.safetensors  — non-quantized weights in shards

Usage:
    python3 export_dynaquant.py \\
        --model /models/Qwen3.5-27B-bf16 \\
        --pareto /tmp/pareto/qwen35-27b-full-linear.json \\
        --step knee \\
        --output /tmp/dynaquant-27b-full
"""
import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "kernels"))
from pack_utils import pack_Nbit_tensor


def resolve_name_mapping(source_names: list, recipe: dict) -> dict:
    """Map source safetensors names to recipe names (handles model.language_model. prefix)."""
    recipe_set = set(recipe.keys())
    mapping = {}
    for sname in source_names:
        if sname in recipe_set:
            mapping[sname] = sname
        elif sname.startswith("model.language_model."):
            rname = "model." + sname[len("model.language_model."):]
            if rname in recipe_set:
                mapping[sname] = rname
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pareto", required=True)
    parser.add_argument("--step", default="knee")
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-size", type=int, default=16)
    args = parser.parse_args()

    t0 = time.time()

    # Load recipe
    with open(args.pareto) as f:
        pareto_data = json.load(f)
    pareto = pareto_data["pareto"]
    if args.step == "knee":
        entry = min(pareto, key=lambda p: abs(p["step"] - pareto_data["knee_step"]))
    else:
        entry = min(pareto, key=lambda p: abs(p["step"] - int(args.step)))
    recipe = entry["recipe"]

    hist = Counter(recipe.values())
    print(f"[dynaquant] Recipe: {dict(sorted(hist.items()))}")
    print(f"[dynaquant] Predicted cost: {entry['cost_bytes']/1e9:.2f} GB")

    source_dir = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    packed_dir = output_dir / "packed_weights"
    packed_dir.mkdir(exist_ok=True)

    # Copy config files
    for f in source_dir.iterdir():
        if f.suffix in (".json", ".jinja", ".txt") and "safetensors" not in f.name:
            shutil.copy2(f, output_dir / f.name)

    # Load source index
    from safetensors import safe_open
    from safetensors.torch import save_file

    with open(source_dir / "model.safetensors.index.json") as f:
        source_index = json.load(f)
    weight_map = source_index["weight_map"]

    # Build name mapping
    name_mapping = resolve_name_mapping(list(weight_map.keys()), recipe)
    print(f"[dynaquant] {len(name_mapping)} tensors mapped to recipe")

    # Process shards
    layer_configs = {}
    total_packed_bytes = 0
    total_passthrough_bytes = 0
    n_quantized = 0

    # Passthrough weights in shards
    MAX_SHARD = 4 * 1024 * 1024 * 1024
    pt_shard = {}
    pt_shard_size = 0
    pt_shard_idx = 0
    pt_shard_files = []

    def flush_shard():
        nonlocal pt_shard, pt_shard_size, pt_shard_idx
        if pt_shard:
            fname = f"passthrough-{pt_shard_idx:03d}.safetensors"
            save_file(pt_shard, str(output_dir / fname))
            pt_shard_files.append(fname)
            pt_shard_idx += 1
            pt_shard = {}
            pt_shard_size = 0

    group_size = args.group_size
    files_to_load = sorted(set(weight_map.values()))

    for shard_file in files_to_load:
        shard_path = source_dir / shard_file
        print(f"[dynaquant] processing {shard_file}", flush=True)

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for tname in f.keys():
                tensor = f.get_tensor(tname)

                # Check if this tensor should be quantized
                rname = name_mapping.get(tname)
                bits = recipe.get(rname, 16) if rname else 16

                if bits < 16 and tname.endswith(".weight") and tensor.dim() == 2:
                    # Quantize and pack
                    w = tensor.float()
                    out_f, in_f = w.shape

                    # Pad to group_size
                    orig_in_f = in_f
                    if in_f % group_size != 0:
                        pad = group_size - (in_f % group_size)
                        w = torch.nn.functional.pad(w, (0, pad))
                        in_f += pad

                    packed, scales = pack_Nbit_tensor(w, bits, group_size)

                    mod_name = tname.replace(".weight", "")
                    torch.save(packed, packed_dir / f"{mod_name}.packed")
                    torch.save(scales, packed_dir / f"{mod_name}.scales")

                    layer_configs[mod_name] = {
                        "n_bits": bits,
                        "out_features": out_f,
                        "in_features": in_f,
                        "original_in_features": orig_in_f,
                        "group_size": group_size,
                    }
                    total_packed_bytes += packed.numel()
                    n_quantized += 1
                    del packed, scales, w
                else:
                    # Passthrough
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    if pt_shard_size + tensor_bytes > MAX_SHARD and pt_shard:
                        flush_shard()
                    pt_shard[tname] = tensor
                    pt_shard_size += tensor_bytes
                    total_passthrough_bytes += tensor_bytes

                del tensor

    flush_shard()

    # Build weight map for passthrough shards
    pt_weight_map = {}
    for fname in pt_shard_files:
        fpath = output_dir / fname
        with safe_open(str(fpath), framework="pt", device="cpu") as f:
            for k in f.keys():
                pt_weight_map[k] = fname

    # Save index for passthrough weights
    pt_index = {
        "metadata": {"total_size": total_passthrough_bytes},
        "weight_map": pt_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(pt_index, f, indent=2)

    # Save DynaQuant config
    dq_config = {
        "format": "dynaquant",
        "version": 1,
        "group_size": group_size,
        "recipe": recipe,
        "layer_configs": layer_configs,
        "passthrough_shards": pt_shard_files,
        "stats": {
            "n_quantized": n_quantized,
            "n_passthrough_tensors": len(pt_weight_map),
            "packed_bytes": total_packed_bytes,
            "passthrough_bytes": total_passthrough_bytes,
            "total_bytes": total_packed_bytes + total_passthrough_bytes,
            "bits_histogram": {str(k): v for k, v in sorted(hist.items())},
        },
    }
    with open(output_dir / "dynaquant_config.json", "w") as f:
        json.dump(dq_config, f, indent=2)

    elapsed = time.time() - t0
    print(f"[dynaquant] done in {elapsed:.0f}s")
    print(f"[dynaquant] {n_quantized} quantized, {len(pt_weight_map)} passthrough")
    print(f"[dynaquant] packed: {total_packed_bytes/1e9:.2f} GB, "
          f"passthrough: {total_passthrough_bytes/1e9:.2f} GB")
    print(f"[dynaquant] total: {(total_packed_bytes + total_passthrough_bytes)/1e9:.2f} GB "
          f"(vs {total_passthrough_bytes/1e9 + n_quantized * 0.01:.0f} GB bf16)")
    print(f"[dynaquant] saved to {output_dir}")


if __name__ == "__main__":
    main()
