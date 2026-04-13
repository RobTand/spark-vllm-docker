#!/usr/bin/env python3
"""
dynaquant_vllm.py — vLLM integration for DynaQuant arbitrary-bit models.

Monkey-patches a vLLM model after weight loading to replace Linear layers
with DynaQuantLinear. This avoids patching vLLM source — instead, we
intercept the model post-init and swap layers.

The DynaQuant model format on disk:
    model_dir/
        config.json              — standard HF config
        tokenizer.json           — standard HF tokenizer
        dynaquant_config.json    — DynaQuant recipe + metadata
        packed_weights/          — per-layer packed data
            model.layers.0.mlp.gate_proj.packed  (uint8)
            model.layers.0.mlp.gate_proj.scales  (fp32)
            ...
        model-*.safetensors      — non-quantized weights (embeddings, norms, etc.)
        model.safetensors.index.json

Usage from Python (outside vLLM):
    from dynaquant_vllm import load_dynaquant_model
    model = load_dynaquant_model("/path/to/dynaquant_model")

Usage with vLLM (as a post-load hook):
    # In a custom model loader or startup script
    from dynaquant_vllm import patch_vllm_model
    patch_vllm_model(engine.model, "/path/to/dynaquant_model")
"""
import json
import os
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from dynaquant_linear import DynaQuantLinear


def save_dynaquant_model(
    model: nn.Module,
    recipe: dict,
    output_dir: str,
    source_model: str = None,
    group_size: int = 16,
):
    """Quantize a HF model per recipe and save in DynaQuant format.

    Args:
        model: loaded HF model (bf16/fp32)
        recipe: {param_name: n_bits} — bits per weight tensor
        output_dir: where to save
        source_model: path to source model (for copying config files)
        group_size: quantization group size
    """
    from kernels.pack_utils import pack_Nbit_tensor
    from safetensors.torch import save_file

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    packed_dir = output / "packed_weights"
    packed_dir.mkdir(exist_ok=True)

    # Copy config files from source
    if source_model:
        src = Path(source_model)
        for f in src.iterdir():
            if f.suffix in (".json", ".jinja", ".txt") and "safetensors" not in f.name:
                import shutil
                shutil.copy2(f, output / f.name)

    # Process each parameter — save incrementally to avoid OOM
    layer_configs = {}
    total_packed_bytes = 0
    total_bf16_bytes = 0
    n_passthrough = 0

    # Collect non-quantized weights in shards (max 4 GB each)
    MAX_SHARD = 4 * 1024 * 1024 * 1024
    nq_shard = {}
    nq_shard_size = 0
    nq_shard_idx = 0

    def flush_nq_shard():
        nonlocal nq_shard, nq_shard_size, nq_shard_idx
        if nq_shard:
            fname = f"non_quantized-{nq_shard_idx:03d}.safetensors"
            save_file(nq_shard, str(output / fname))
            nq_shard_idx += 1
            nq_shard = {}
            nq_shard_size = 0

    for name, param in model.named_parameters():
        mod_name = name.replace(".weight", "")
        bits = recipe.get(name, 16)

        if bits >= 16 or not name.endswith(".weight"):
            # Keep as-is — add to current shard
            tensor = param.data.cpu()
            tensor_bytes = tensor.numel() * tensor.element_size()
            if nq_shard_size + tensor_bytes > MAX_SHARD and nq_shard:
                flush_nq_shard()
            nq_shard[name] = tensor
            nq_shard_size += tensor_bytes
            total_bf16_bytes += tensor_bytes
            n_passthrough += 1
        else:
            # Quantize and pack
            w = param.data.cpu().float()
            out_f, in_f = w.shape

            # Pad to group_size
            if in_f % group_size != 0:
                pad = group_size - (in_f % group_size)
                w = torch.nn.functional.pad(w, (0, pad))
                in_f += pad

            packed, scales = pack_Nbit_tensor(w, bits, group_size)

            # Save packed data immediately (no accumulation)
            torch.save(packed, packed_dir / f"{mod_name}.packed")
            torch.save(scales, packed_dir / f"{mod_name}.scales")
            del packed, scales, w

            layer_configs[mod_name] = {
                "n_bits": bits,
                "out_features": out_f,
                "in_features": in_f,
                "original_in_features": param.shape[1],
                "group_size": group_size,
            }
            total_packed_bytes += (out_f * in_f * bits + 7) // 8

    # Flush remaining non-quantized weights
    flush_nq_shard()

    # Save buffers
    buffers = {}
    buf_size = 0
    for name, buf in model.named_buffers():
        buffers[name] = buf.cpu()
        buf_size += buf.numel() * buf.element_size()
    if buffers:
        save_file(buffers, str(output / "buffers.safetensors"))
        total_bf16_bytes += buf_size

    # Save DynaQuant config
    hist = Counter(recipe.get(n, 16) for n in recipe)
    dq_config = {
        "format": "dynaquant",
        "version": 1,
        "group_size": group_size,
        "recipe": recipe,
        "layer_configs": layer_configs,
        "stats": {
            "n_quantized": len(layer_configs),
            "n_passthrough": n_passthrough,
            "packed_bytes": total_packed_bytes,
            "bf16_bytes": total_bf16_bytes,
            "total_bytes": total_packed_bytes + total_bf16_bytes,
            "bits_histogram": {str(k): v for k, v in sorted(hist.items())},
        },
    }
    with open(output / "dynaquant_config.json", "w") as f:
        json.dump(dq_config, f, indent=2)

    print(f"[dynaquant] saved to {output_dir}")
    print(f"[dynaquant] {len(layer_configs)} quantized layers, "
          f"{n_passthrough} passthrough")
    print(f"[dynaquant] packed: {total_packed_bytes/1e9:.2f} GB, "
          f"bf16: {total_bf16_bytes/1e9:.2f} GB, "
          f"total: {(total_packed_bytes + total_bf16_bytes)/1e9:.2f} GB")


def load_dynaquant_weights(
    model: nn.Module,
    model_dir: str,
    device: str = "cuda",
) -> nn.Module:
    """Load DynaQuant packed weights into a model, replacing Linear layers.

    Args:
        model: initialized model (e.g. from AutoModelForCausalLM)
        model_dir: directory with dynaquant_config.json + packed_weights/
        device: target device

    Returns:
        model with DynaQuantLinear layers replacing quantized Linears
    """
    model_dir = Path(model_dir)
    packed_dir = model_dir / "packed_weights"

    with open(model_dir / "dynaquant_config.json") as f:
        dq_config = json.load(f)

    layer_configs = dq_config["layer_configs"]
    group_size = dq_config["group_size"]

    # Load non-quantized weights
    from safetensors.torch import load_file
    nq_shards = sorted(model_dir.glob("non_quantized-*.safetensors"))
    if not nq_shards:
        legacy_nq_path = model_dir / "non_quantized.safetensors"
        if legacy_nq_path.exists():
            nq_shards = [legacy_nq_path]
    for nq_path in nq_shards:
        nq_weights = load_file(str(nq_path), device=device)
        for name, param in model.named_parameters():
            if name in nq_weights:
                param.data.copy_(nq_weights[name])

    # Load buffers
    buf_path = model_dir / "buffers.safetensors"
    if buf_path.exists():
        from safetensors.torch import load_file
        buf_weights = load_file(str(buf_path), device=device)
        for name, buf in model.named_buffers():
            if name in buf_weights:
                buf.data.copy_(buf_weights[name])

    # Replace quantized Linear layers
    # Build lookup of all modules for flexible name resolution
    all_modules = dict(model.named_modules())

    replaced = 0
    for mod_name, cfg in layer_configs.items():
        # Create DynaQuantLinear
        dq = DynaQuantLinear(
            in_features=cfg["in_features"],
            out_features=cfg["out_features"],
            n_bits=cfg["n_bits"],
            group_size=cfg["group_size"],
            bias=False,  # TODO: handle bias
        )

        # Load packed data
        packed = torch.load(packed_dir / f"{mod_name}.packed", weights_only=True)
        scales = torch.load(packed_dir / f"{mod_name}.scales", weights_only=True)

        # Copy into buffers
        dq.packed_weight[:packed.numel()] = packed
        dq.weight_scales.copy_(scales)
        dq = dq.to(device)

        # Resolve module name: try as-is, then strip prefixes
        resolve_name = mod_name
        if resolve_name not in all_modules:
            # Try stripping model.language_model. → model.
            if resolve_name.startswith("model.language_model."):
                resolve_name = "model." + resolve_name[len("model.language_model."):]
            elif resolve_name.startswith("language_model."):
                resolve_name = resolve_name[len("language_model."):]

        if resolve_name not in all_modules:
            print(f"[dynaquant] WARNING: could not find module {mod_name} "
                  f"(tried {resolve_name})")
            continue

        # Replace in model
        parts = resolve_name.rsplit(".", 1)
        if len(parts) == 2:
            parent = all_modules[parts[0]]
            setattr(parent, parts[1], dq)
        else:
            setattr(model, parts[0], dq)
        replaced += 1

    print(f"[dynaquant] loaded {replaced} DynaQuantLinear layers from {model_dir}")
    return model


# ---------------------------------------------------------------------------
# CLI: export a model to DynaQuant format
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export model to DynaQuant format")
    parser.add_argument("--model", required=True, help="HF model path")
    parser.add_argument("--pareto", required=True, help="Pareto JSON from allocator")
    parser.add_argument("--step", default="knee", help="Pareto step or 'knee'")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    from build_rtn_cache import stage_multimodal

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

    # Stage and load model
    staged, cleanup = stage_multimodal(args.model)
    try:
        from transformers import AutoModelForCausalLM
        print(f"[dynaquant] loading {staged}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            staged, dtype=torch.bfloat16,
            device_map="cpu", trust_remote_code=True,
        )

        save_dynaquant_model(model, recipe, args.output, source_model=staged)
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(cleanup, ignore_errors=True)
