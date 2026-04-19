"""
Upgrade an NVFP4A16 checkpoint to full NVFP4 by adding input_global_scale tensors.

This enables CUTLASS W4A4 (native FP4 compute) instead of Marlin W4A16 (dequant).

For layers that exist in a reference NVFP4 checkpoint (e.g., Sehyo's), we copy
the calibrated input_global_scale. For newly quantized layers (DeltaNet), we use
the median of existing scales as a reasonable default.
"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch


def get_scales_from_reference(ref_path):
    """Extract input_global_scale values from a reference NVFP4 checkpoint."""
    scales = {}
    # Find safetensors files
    import glob
    files = glob.glob(os.path.join(ref_path, "*.safetensors"))
    for f in files:
        with safe_open(f, framework="numpy") as sf:
            for k in sf.keys():
                if "input_global_scale" in k:
                    scales[k] = sf.get_tensor(k)
    return scales


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="NVFP4A16 checkpoint to upgrade")
    parser.add_argument("--reference", help="Reference NVFP4 checkpoint with calibrated scales")
    parser.add_argument("--default-scale", type=float, default=72.25, help="Default input_global_scale for uncalibrated layers")
    args = parser.parse_args()

    # Load reference scales if available
    ref_scales = {}
    if args.reference:
        print(f"Loading reference scales from {args.reference}...")
        ref_scales = get_scales_from_reference(args.reference)
        print(f"  Found {len(ref_scales)} calibrated input_global_scale values")

    # Process each safetensors shard
    import glob
    shards = sorted(glob.glob(os.path.join(args.checkpoint, "model*.safetensors")))

    # Load the index
    index_path = os.path.join(args.checkpoint, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    added = 0
    for shard_path in shards:
        shard_name = os.path.basename(shard_path)
        tensors = {}
        new_tensors = {}

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        # Find weight_packed tensors that need input_global_scale
        for k in list(tensors.keys()):
            if not k.endswith(".weight_packed"):
                continue

            base = k.rsplit(".weight_packed", 1)[0]
            igs_key = f"{base}.input_global_scale"

            # Skip if already has input_global_scale
            if igs_key in tensors or igs_key in weight_map:
                continue

            # Try to get from reference
            if igs_key in ref_scales:
                val = torch.tensor(ref_scales[igs_key], dtype=torch.float32)
            else:
                val = torch.tensor([args.default_scale], dtype=torch.float32)

            new_tensors[igs_key] = val
            weight_map[igs_key] = shard_name
            added += 1

        if new_tensors:
            print(f"  {shard_name}: adding {len(new_tensors)} input_global_scale tensors")
            tensors.update(new_tensors)
            save_file(tensors, shard_path)

    # Update index
    index["weight_map"] = weight_map
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    # Update quantization_config to indicate full NVFP4 (with input activations)
    config_path = os.path.join(args.checkpoint, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    qc = config.get("quantization_config", {})
    for group_name, group in qc.get("config_groups", {}).items():
        # Add input_activations to the scheme
        if "input_activations" not in group or group["input_activations"] is None:
            group["input_activations"] = {
                "dynamic": "local",
                "group_size": 16,
                "num_bits": 4,
                "observer": "static_minmax",
                "observer_kwargs": {},
                "scale_dtype": "torch.float8_e4m3fn",
                "strategy": "tensor_group",
                "symmetric": True,
                "type": "float",
                "zp_dtype": None,
                "actorder": None,
                "block_structure": None,
            }
            print(f"  Added input_activations to {group_name}")

        # Update format to nvfp4-pack-quantized
        if group.get("format") == "pack-quantized":
            group["format"] = "nvfp4-pack-quantized"
            print(f"  Updated format to nvfp4-pack-quantized for {group_name}")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone! Added {added} input_global_scale tensors.")
    print("Checkpoint is now full NVFP4 (W4A4) compatible.")


if __name__ == "__main__":
    main()
