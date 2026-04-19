"""
Quantize DeltaNet linear_attn BF16 layers to NVFP4 in an existing checkpoint.

Uses FlashInfer's nvfp4_quantize for round-to-nearest FP4 quantization.
Produces weight_packed (uint8), weight_scale (fp8), weight_global_scale (fp32),
and input_global_scale (fp32) matching the format of existing NVFP4 layers.

Usage (inside container with FlashInfer available):
  python3 quantize_linear_attn.py \
    --source /path/to/Sehyo-checkpoint \
    --output /path/to/output

Works on any Qwen3.5 NVFP4 checkpoint (27B or 122B).
"""

import argparse
import json
import os
import struct
import shutil
from pathlib import Path

import torch


def get_safetensors_header(path):
    """Read safetensors header without loading tensors."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return header


def find_bf16_linear_attn_weights(checkpoint_dir):
    """Find BF16 linear_attn weight tensors that should be quantized."""
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    targets = {}
    for tensor_name, shard_file in index["weight_map"].items():
        if "linear_attn" in tensor_name and tensor_name.endswith(".weight"):
            # Skip conv1d (3D tensors) and norm (1D tensors)
            if "conv1d" in tensor_name or ".norm." in tensor_name:
                continue
            targets[tensor_name] = shard_file

    return targets, index


def quantize_tensor_to_nvfp4(weight_bf16, device="cuda"):
    """Quantize a BF16 weight tensor to NVFP4 format.

    Returns: weight_packed (uint8), weight_scale (fp8), weight_global_scale (fp32),
             input_global_scale (fp32)
    """
    import flashinfer
    from flashinfer import SfLayout

    weight_bf16 = weight_bf16.to(device)

    # Compute global scale: max_abs * 2 / (FP4_MAX * FP8_MAX) = max_abs * 2 / (6.0 * 448.0)
    # This is the standard NVFP4 global scale computation
    max_abs = weight_bf16.float().abs().max()
    fp4_max = 6.0
    fp8_max = 448.0
    weight_global_scale = (max_abs / (fp4_max * fp8_max)).to(torch.float32)

    # For input_global_scale, use same value as weight (reasonable default for RTN)
    # This gets refined during actual calibration but for RTN it's a placeholder
    input_global_scale = weight_global_scale.clone()

    # Quantize using FlashInfer (linear layout, no shuffle)
    global_sf_inv = (fp4_max * fp8_max) / max_abs
    weight_fp4, weight_scale = flashinfer.nvfp4_quantize(
        weight_bf16,
        global_sf_inv.to(device),
        sfLayout=SfLayout.layout_128x4,  # Match CUTLASS expected layout
        do_shuffle=False,
        sf_vec_size=16,
    )

    return (
        weight_fp4.view(torch.uint8).cpu(),
        weight_scale.view(torch.float8_e4m3fn).cpu(),
        weight_global_scale.cpu().view(1),
        input_global_scale.cpu().view(1),
    )


def main():
    parser = argparse.ArgumentParser(description="Quantize DeltaNet BF16 layers to NVFP4")
    parser.add_argument("--source", required=True, help="Source checkpoint directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device for quantization")
    parser.add_argument("--dry-run", action="store_true", help="Only list targets")
    args = parser.parse_args()

    # Find target tensors
    targets, index = find_bf16_linear_attn_weights(args.source)
    print(f"Found {len(targets)} BF16 linear_attn weight tensors to quantize")

    if args.dry_run:
        for name, shard in sorted(targets.items()):
            print(f"  {name} -> {shard}")
        return

    # Copy checkpoint
    print(f"Copying checkpoint to {args.output}...")
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    shutil.copytree(args.source, args.output)

    # Group targets by shard file
    from collections import defaultdict
    shard_targets = defaultdict(list)
    for tensor_name, shard_file in targets.items():
        shard_targets[shard_file].append(tensor_name)

    # Process each shard
    from safetensors import safe_open
    from safetensors.torch import save_file

    weight_map = dict(index["weight_map"])
    total_quantized = 0

    for shard_file, tensor_names in shard_targets.items():
        shard_path = os.path.join(args.output, shard_file)
        print(f"\nProcessing {shard_file} ({len(tensor_names)} tensors to quantize)...")

        # Load all tensors from this shard
        tensors = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        # Quantize target tensors
        for tensor_name in tensor_names:
            weight = tensors.pop(tensor_name)
            print(f"  Quantizing {tensor_name}: {weight.shape} {weight.dtype}")

            packed, scale, global_scale, input_scale = quantize_tensor_to_nvfp4(
                weight, device=args.device
            )

            # Add new tensors with NVFP4 naming convention
            base_name = tensor_name.rsplit(".weight", 1)[0]
            tensors[f"{base_name}.weight_packed"] = packed
            tensors[f"{base_name}.weight_scale"] = scale
            tensors[f"{base_name}.weight_global_scale"] = global_scale
            tensors[f"{base_name}.input_global_scale"] = input_scale

            # Update weight map
            del weight_map[tensor_name]
            weight_map[f"{base_name}.weight_packed"] = shard_file
            weight_map[f"{base_name}.weight_scale"] = shard_file
            weight_map[f"{base_name}.weight_global_scale"] = shard_file
            weight_map[f"{base_name}.input_global_scale"] = shard_file

            total_quantized += 1

        # Save updated shard
        save_file(tensors, shard_path)
        print(f"  Saved {shard_file}")

    # Update index
    index["weight_map"] = weight_map
    index_path = os.path.join(args.output, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    # Update config.json to remove linear_attn from ignore list
    config_path = os.path.join(args.output, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    if "quantization_config" in config:
        qc = config["quantization_config"]
        if "ignore" in qc:
            qc["ignore"] = [i for i in qc["ignore"] if "linear_attn" not in i
                            or "conv1d" in i]
        # Also check nested config_groups
        if "config_groups" in qc:
            for group in qc["config_groups"].values():
                if "ignore" in group:
                    group["ignore"] = [i for i in group["ignore"]
                                       if "linear_attn" not in i or "conv1d" in i]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone! Quantized {total_quantized} tensors.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
