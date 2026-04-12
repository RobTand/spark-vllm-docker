#!/usr/bin/env python3
"""
export_dynaquant_ct.py — Export a DynaQuant model with pre-packed weights.

Reads source model safetensors, packs quantized layers to N-bit, and
writes a new set of safetensors where:
  - Quantized layers: {prefix}.weight_packed (uint8 1D)
                      {prefix}.weight_scale  (fp32 2D)
                      {prefix}.weight_bits   (int8 scalar)
  - Non-quantized layers: kept as-is (bf16)
  - config.json: quantization_config with targets=["Linear"], format=dynaquant-pack-quantized

Each projection is self-describing (bit width encoded in weight_bits tensor).
Fused layers in vLLM (gate_up, qkv) load sub-projections at independent bit
widths — no snapping to a common bit width.

The output model loads directly in vLLM with the DynaQuant scheme —
no runtime packing needed.

Usage:
    python3 export_dynaquant_ct.py \\
        --model /models/Qwen3.5-27B-bf16 \\
        --pareto /tmp/pareto/qwen35-27b-full-linear.json \\
        --step knee \\
        --output /tmp/dynaquant-27b-native
"""
import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "kernels"))
from pack_utils import pack_Nbit_tensor


def resolve_name_mapping(source_names, recipe):
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

    source_dir = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_size = args.group_size

    # Copy non-weight files
    for f in source_dir.iterdir():
        if f.is_dir():
            continue
        if f.suffix in (".json", ".jinja", ".txt") and "safetensors" not in f.name and f.name != "config.json":
            shutil.copy2(f, output_dir / f.name)

    # Load source index
    from safetensors import safe_open
    from safetensors.torch import save_file

    with open(source_dir / "model.safetensors.index.json") as f:
        source_index = json.load(f)

    name_mapping = resolve_name_mapping(list(source_index["weight_map"].keys()), recipe)

    # Process shards — stream through, pack quantized layers, pass through rest
    MAX_SHARD = 5 * 1024 * 1024 * 1024
    out_shard = {}
    out_shard_size = 0
    out_shard_idx = 0
    out_weight_map = {}
    n_quantized = 0
    total_packed = 0
    total_passthrough = 0

    # Track bit widths used (for quantization_config)
    bits_used = set()
    ignore = []

    def flush_shard():
        nonlocal out_shard, out_shard_size, out_shard_idx
        if out_shard:
            fname = f"model-{out_shard_idx+1:05d}-of-NNNNN.safetensors"
            save_file(out_shard, str(output_dir / fname))
            for name in out_shard:
                out_weight_map[name] = fname
            out_shard_idx += 1
            out_shard = {}
            out_shard_size = 0

    for shard_file in sorted(set(source_index["weight_map"].values())):
        shard_path = source_dir / shard_file
        print(f"[dynaquant] processing {shard_file}", flush=True)

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for tname in f.keys():
                tensor = f.get_tensor(tname)
                rname = name_mapping.get(tname)
                bits = recipe.get(rname, 16) if rname else 16

                # Skip special layers that vLLM handles differently
                is_special = any(s in tname for s in [
                    "lm_head", "embed_tokens", "mtp.",
                ])
                if bits < 16 and tname.endswith(".weight") and tensor.dim() == 2 and not is_special:
                    # Pack to N-bit
                    w = tensor.float()
                    out_f, in_f = w.shape
                    if in_f % group_size != 0:
                        pad = group_size - (in_f % group_size)
                        w = torch.nn.functional.pad(w, (0, pad))
                        in_f += pad

                    packed, scales = pack_Nbit_tensor(w, bits, group_size)

                    prefix = tname.replace(".weight", "")
                    packed_name = f"{prefix}.weight_packed"
                    scale_name = f"{prefix}.weight_scale"
                    bits_name = f"{prefix}.weight_bits"

                    packed_bytes = packed.numel()
                    scale_bytes = scales.numel() * scales.element_size()
                    bits_tensor = torch.tensor(bits, dtype=torch.int8)

                    if out_shard_size + packed_bytes + scale_bytes > MAX_SHARD and out_shard:
                        flush_shard()

                    out_shard[packed_name] = packed
                    out_shard[scale_name] = scales
                    out_shard[bits_name] = bits_tensor
                    out_shard_size += packed_bytes + scale_bytes
                    total_packed += packed_bytes

                    bits_used.add(bits)
                    n_quantized += 1
                    del packed, scales, w
                else:
                    # Passthrough — track weight tensors for ignore list
                    if tname.endswith(".weight"):
                        ignore.append(tname.replace(".weight", ""))
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    if out_shard_size + tensor_bytes > MAX_SHARD and out_shard:
                        flush_shard()
                    out_shard[tname] = tensor
                    out_shard_size += tensor_bytes
                    total_passthrough += tensor_bytes

                del tensor

    flush_shard()

    # Fix shard filenames (replace NNNNN with actual count)
    n_shards = out_shard_idx
    final_weight_map = {}
    for tname, fname in out_weight_map.items():
        new_fname = fname.replace("NNNNN", f"{n_shards:05d}")
        final_weight_map[tname] = new_fname
    # Rename files
    for old_fname in set(out_weight_map.values()):
        new_fname = old_fname.replace("NNNNN", f"{n_shards:05d}")
        if old_fname != new_fname:
            (output_dir / old_fname).rename(output_dir / new_fname)

    # Write index
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump({
            "metadata": {"total_size": total_packed + total_passthrough},
            "weight_map": final_weight_map,
        }, f, indent=2)

    # Apply HF→vLLM name mapping to ignore list so compressed-tensors
    # can match against vLLM's internal module names.
    # Common mappings (Qwen3.5, Qwen3-VL family):
    vllm_prefix_map = [
        ("model.visual.", "visual."),
        ("model.language_model.", "language_model.model."),
        ("lm_head.", "language_model.lm_head."),
    ]

    def to_vllm_name(name):
        for old, new in vllm_prefix_map:
            if name.startswith(old):
                return new + name[len(old):]
        # Strip leading "model." if no specific mapping matched
        if name.startswith("model."):
            return name[len("model."):]
        return name

    vllm_ignore = [to_vllm_name(n) for n in ignore]

    # Build quantization_config — single group targeting all Linear layers.
    # Per-projection bit widths are self-describing via weight_bits tensors.
    # Use the min bit width in the config since the scheme reads actual bits
    # from weight_bits at load time; this just needs to trigger DynaQuant routing.
    min_bits = min(bits_used) if bits_used else 4
    config_groups = {
        "dynaquant": {
            "weights": {
                "num_bits": min_bits,
                "type": "int",
                "strategy": "group",
                "group_size": group_size,
                "symmetric": True,
            },
            "input_activations": None,
            "targets": ["Linear"],
        },
    }

    quant_config = {
        "config_groups": config_groups,
        "format": "dynaquant-pack-quantized",
        "ignore": vllm_ignore,
        "quant_method": "compressed-tensors",
    }

    # Write config.json
    with open(source_dir / "config.json") as f:
        config = json.load(f)
    config["quantization_config"] = quant_config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - t0
    total_gb = (total_packed + total_passthrough) / 1e9
    print(f"[dynaquant] Done in {elapsed:.0f}s")
    print(f"[dynaquant] {n_quantized} layers packed, {len(bits_used)} distinct bit widths")
    print(f"[dynaquant] packed: {total_packed/1e9:.2f} GB + passthrough: {total_passthrough/1e9:.2f} GB = {total_gb:.2f} GB total")


if __name__ == "__main__":
    main()
