#!/usr/bin/env python3
"""
export_compressed_tensors.py — export a DynaQuant block-level recipe as a
compressed-tensors model in the exact format vLLM expects.

Operates directly on safetensors files from the source model:
- FP4 blocks → NVFP4 packed format (weight_packed, weight_scale, etc.)
- FP8 blocks → RTN-FP8 applied, stored as bf16
- Visual/non-text weights → passed through unchanged

The output model can be served directly:
    vllm serve /path/to/output --trust-remote-code

Usage:
    python3 export_compressed_tensors.py \\
        --model /models/Qwen3.5-27B-bf16 \\
        --pareto /tmp/pareto/qwen35-27b-hw-block.json \\
        --step knee \\
        --output /tmp/dynaquant-27b-ct
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
from build_rtn_cache import rtn_fp8_any_shape


GROUP_SIZE = 16


def quantize_to_nvfp4(weight: torch.Tensor) -> dict:
    """Quantize a 2D weight tensor to NVFP4 packed format.

    Uses the exact same convention as vLLM's ref_nvfp4_quant:
      global_scale = reciprocal of (max_abs / (fp4_max * fp8_max))
      per_group_scale = global_scale * group_max / fp4_max → cast to FP8
      output_scale = global_scale / per_group_scale
      quantized = cast_to_fp4(weight * output_scale)
      dequant = fp4_value * per_group_scale * global_scale

    Returns weight_packed (uint8), weight_scale (fp8), weight_global_scale (fp32),
    input_global_scale (fp32).
    """
    FP4_MAX = 6.0
    FP4_MAX_RECIP = 1.0 / FP4_MAX

    out_f, in_f = weight.shape
    assert in_f % GROUP_SIZE == 0, f"in_features {in_f} not divisible by {GROUP_SIZE}"

    w = weight.float()
    n_groups = in_f // GROUP_SIZE
    w_grouped = w.view(out_f, n_groups, GROUP_SIZE)

    # Global scale: reciprocal convention matching NVIDIA/vLLM
    # global_scale maps the overall weight range so per-group scales fit in FP8
    vec_max = w_grouped.abs().amax(dim=-1, keepdim=True)  # (out, n_groups, 1)
    overall_max = vec_max.max().item()
    if overall_max == 0:
        overall_max = 1.0
    # Convention: global_scale = 1 / (overall_max / (FP4_MAX * FP8_MAX))
    # But we match ref_nvfp4_quant which takes global_scale as input and computes:
    #   scale = global_scale * vec_max / FP4_MAX
    # For this to produce valid FP8 scales, we need:
    #   global_scale * overall_max / FP4_MAX <= 448  (FP8 max)
    # So: global_scale <= 448 * FP4_MAX / overall_max = 2688 / overall_max
    # NVIDIA uses: global_scale = 1.0 / (overall_max / (FP4_MAX * 448))
    #            = FP4_MAX * 448 / overall_max = 2688 / overall_max
    weight_global_scale = FP4_MAX * 448.0 / overall_max

    # Per-group scale: scale = global_scale * vec_max / FP4_MAX → cast to FP8
    scale = weight_global_scale * vec_max.squeeze(-1) * FP4_MAX_RECIP
    scale = torch.clamp(scale, max=448.0, min=-448.0)
    scale_fp8 = scale.to(torch.float8_e4m3fn)
    scale_fp32 = scale_fp8.float()

    # Output scale for quantization: output_scale = global_scale / scale
    # This maps weights to [-FP4_MAX, FP4_MAX] range
    output_scale = weight_global_scale / (scale_fp32 + 1e-10)
    output_scale = output_scale.unsqueeze(-1)  # (out, n_groups, 1)

    # Quantize: scale weights then cast to nearest FP4 E2M1
    scaled_w = w_grouped * output_scale
    scaled_w = torch.clamp(scaled_w, -FP4_MAX, FP4_MAX)

    # Cast to nearest FP4 E2M1 value
    sign = torch.sign(scaled_w)
    abs_w = scaled_w.abs()
    # FP4 E2M1 values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32,
                              device=weight.device)
    diffs = (abs_w.unsqueeze(-1) - fp4_values.view(1, 1, 1, -1)).abs()
    codes = diffs.argmin(dim=-1)  # magnitude code 0-7
    sign_bit = (sign < 0).to(torch.int32)
    codes = codes | (sign_bit << 3)  # 4-bit code: sign(1) | magnitude(3)

    # Pack 2 FP4 codes per byte: low nibble = even index, high nibble = odd index
    codes_flat = codes.view(out_f, -1)  # (out, in_f)
    even = codes_flat[:, 0::2]
    odd = codes_flat[:, 1::2]
    packed = (even | (odd << 4)).to(torch.uint8)

    # Input global scale: for W4A4 with dynamic activations, this is calibrated
    # from activation statistics. We set to 1.0 as placeholder since vLLM
    # recomputes it dynamically with observer="static_minmax" + dynamic="local"
    input_global_scale = torch.tensor([1.0], dtype=torch.float32)

    return {
        "weight_packed": packed,
        "weight_scale": scale_fp8,
        "weight_global_scale": torch.tensor([weight_global_scale], dtype=torch.float32),
        "input_global_scale": input_global_scale,
    }


def resolve_name_mapping(source_names: list, recipe: dict) -> dict:
    """Build a mapping from source safetensors names to recipe names.

    The recipe has names like 'model.layers.22.mlp.gate_proj.weight'
    but the source model may use 'model.language_model.layers.22.mlp.gate_proj.weight'.
    """
    recipe_set = set(recipe.keys())
    mapping = {}

    for sname in source_names:
        # Direct match
        if sname in recipe_set:
            mapping[sname] = sname
            continue

        # Try stripping 'model.language_model.' → 'model.'
        if sname.startswith("model.language_model."):
            rname = "model." + sname[len("model.language_model."):]
            if rname in recipe_set:
                mapping[sname] = rname
                continue

        # Try stripping 'model.' → bare name
        if sname.startswith("model."):
            rname = sname[len("model."):]
            if rname in recipe_set:
                mapping[sname] = rname

    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pareto", required=True)
    parser.add_argument("--step", default="knee")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.time()

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
    print(f"[export] Recipe: {dict(sorted(hist.items()))}")
    print(f"[export] Predicted cost: {entry['cost_bytes']/1e9:.2f} GB")

    source_dir = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files from source model
    for fname in source_dir.iterdir():
        if fname.suffix in (".json", ".jinja", ".txt") and fname.name != "model.safetensors.index.json":
            shutil.copy2(fname, output_dir / fname.name)
    print(f"[export] copied config files from {source_dir}", flush=True)

    # Load source safetensors index
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = source_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        source_index = json.load(f)
    weight_map = source_index["weight_map"]

    # Build name mapping
    all_source_names = list(weight_map.keys())
    name_mapping = resolve_name_mapping(all_source_names, recipe)
    print(f"[export] {len(name_mapping)} tensors mapped to recipe "
          f"(out of {len(all_source_names)} source tensors)", flush=True)

    # Classify tensors
    fp4_sources = {}  # source_name → recipe_name
    fp8_sources = {}
    for sname, rname in name_mapping.items():
        bits = recipe[rname]
        if bits <= 4:
            fp4_sources[sname] = rname
        elif bits <= 8:
            fp8_sources[sname] = rname

    # Tensors NOT in recipe (visual, embeddings, norms, biases, etc.) → pass through
    passthrough = set(all_source_names) - set(name_mapping.keys())

    print(f"[export] FP4: {len(fp4_sources)}, FP8: {len(fp8_sources)}, "
          f"passthrough: {len(passthrough)}", flush=True)

    # Build ignore list for quantization config (everything not FP4)
    fp4_target_names = []
    ignore_names = []
    for sname, rname in name_mapping.items():
        mod_name = sname.replace(".weight", "")
        if recipe[rname] <= 4:
            fp4_target_names.append(mod_name)
        else:
            ignore_names.append(mod_name)
    # Also add passthrough Linear layers (visual, embeddings, etc.) to ignore
    for sname in passthrough:
        if sname.endswith(".weight"):
            mod_name = sname.replace(".weight", "")
            ignore_names.append(mod_name)

    # Process all tensors
    all_tensors = {}
    n_quantized = 0
    files_to_load = set(weight_map.values())

    for shard_file in sorted(files_to_load):
        shard_path = source_dir / shard_file
        print(f"[export] processing {shard_file}", flush=True)

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for tname in f.keys():
                tensor = f.get_tensor(tname)

                if tname in fp4_sources:
                    # Quantize to NVFP4
                    result = quantize_to_nvfp4(tensor)
                    prefix = tname.replace(".weight", "")
                    all_tensors[f"{prefix}.weight_packed"] = result["weight_packed"]
                    all_tensors[f"{prefix}.weight_scale"] = result["weight_scale"]
                    all_tensors[f"{prefix}.weight_global_scale"] = result["weight_global_scale"]
                    all_tensors[f"{prefix}.input_global_scale"] = result["input_global_scale"]
                    n_quantized += 1
                elif tname in fp8_sources:
                    # Apply RTN-FP8
                    w_fp8 = rtn_fp8_any_shape(tensor)
                    all_tensors[tname] = w_fp8.to(torch.bfloat16)
                else:
                    # Pass through unchanged
                    all_tensors[tname] = tensor

    print(f"[export] {n_quantized} layers quantized to NVFP4", flush=True)
    print(f"[export] {len(all_tensors)} total tensors", flush=True)

    # Save in shards
    MAX_SHARD_BYTES = 5 * 1024 * 1024 * 1024
    shards = []
    current_shard = {}
    current_size = 0

    for name in sorted(all_tensors.keys()):
        tensor = all_tensors[name]
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > MAX_SHARD_BYTES and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    out_weight_map = {}
    n_shards = len(shards)
    for i, shard in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{n_shards:05d}.safetensors"
        shard_path = output_dir / shard_name
        print(f"[export] saving {shard_name} ({len(shard)} tensors)", flush=True)
        save_file(shard, str(shard_path))
        for name in shard:
            out_weight_map[name] = shard_name

    # Write index
    index = {
        "metadata": {"total_size": sum(
            t.numel() * t.element_size() for t in all_tensors.values()
        )},
        "weight_map": out_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Write quantization_config.json
    quant_config = {
        "config_groups": {
            "group_fp4": {
                "format": "nvfp4-pack-quantized",
                "input_activations": {
                    "dynamic": "local",
                    "group_size": 16,
                    "num_bits": 4,
                    "observer": "static_minmax",
                    "scale_dtype": "torch.float8_e4m3fn",
                    "strategy": "tensor_group",
                    "symmetric": True,
                    "type": "float",
                },
                "weights": {
                    "dynamic": False,
                    "group_size": 16,
                    "num_bits": 4,
                    "observer": "memoryless_minmax",
                    "scale_dtype": "torch.float8_e4m3fn",
                    "strategy": "tensor_group",
                    "symmetric": True,
                    "type": "float",
                },
                "targets": fp4_target_names,
            },
        },
        "format": "nvfp4-pack-quantized",
        "ignore": ignore_names,
        "quant_method": "compressed-tensors",
    }
    with open(output_dir / "quantization_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)

    # Embed quantization_config in config.json
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["quantization_config"] = quant_config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[export] done in {time.time() - t_start:.0f}s", flush=True)
    print(f"[export] saved to {args.output}")
    print(f"[export] serve with: vllm serve {args.output} --trust-remote-code")


if __name__ == "__main__":
    main()
