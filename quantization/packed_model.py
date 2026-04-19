#!/usr/bin/env python3
"""
packed_model.py — save and load models with N-bit packed weight storage.

Instead of storing quantized weights as bf16 (simulated round-trip), this
module packs each weight matrix at its allocated bit width. A 5-bit Linear
takes 5/16 of the bf16 storage. Per-group scales are stored alongside.

Format on disk:
    model_dir/
    ├── config.json           (standard HF config)
    ├── tokenizer files       (standard HF)
    ├── dynaquant_config.json (recipe: {linear_name: n_bits, ...})
    ├── packed_weights/
    │   ├── model.layers.0.self_attn.q_proj.weight.packed   (N-bit packed codes)
    │   ├── model.layers.0.self_attn.q_proj.weight.scales   (fp32 per-group scales)
    │   └── ...
    └── unpacked_weights.safetensors  (non-quantized params: embeddings, norms)

Loading for inference:
    model = load_dynaquant_model(model_dir)
    # Each quantized Linear is replaced with DynaQuantLinear which
    # stores packed weights and dequantizes on forward().
"""
import gc
import json
import os
import shutil
import struct
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_rtn_cache import (
    stage_multimodal,
    iter_quantizable_tensors,
    rtn_fp4_any_shape,
    rtn_fp8_any_shape,
)
from measure_bit_utility import int_quantize_per_group
from kernels.pack_utils import pack_Nbit_tensor, unpack_Nbit_tensor
from kernels.dequant_gpu import dequant_Nbit_gpu


# ---------------------------------------------------------------------------
# DynaQuantLinear — drop-in replacement for nn.Linear
# ---------------------------------------------------------------------------

class DynaQuantLinear(nn.Module):
    """Linear layer with packed N-bit weight storage.

    Stores weights as packed N-bit codes + per-group fp32 scales.
    On forward(), dequantizes to bf16 and performs the matmul.

    For a fused kernel, the forward() would call a custom CUDA/Triton
    kernel that loads packed data directly. This version uses the
    PyTorch-based GPU dequant as a reference implementation.
    """

    def __init__(self, in_features: int, out_features: int,
                 n_bits: int, group_size: int = 16,
                 bias: bool = False, device=None,
                 cache_weight: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.group_size = group_size
        self.cache_weight = cache_weight
        self._cached_weight: Optional[torch.Tensor] = None

        # Packed weight storage (registered as buffer, not parameter)
        n_values = out_features * in_features
        n_bytes = (n_values * n_bits + 7) // 8
        self.register_buffer('packed_weight',
                             torch.zeros(n_bytes, dtype=torch.uint8, device=device))

        n_groups = in_features // group_size
        self.register_buffer('weight_scales',
                             torch.zeros(out_features, n_groups, dtype=torch.float32,
                                         device=device))

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.bfloat16,
                                                      device=device))
        else:
            self.bias = None

    def pack_weight(self, weight: torch.Tensor):
        """Quantize and pack a bf16/fp32 weight tensor."""
        packed, scales = pack_Nbit_tensor(
            weight.cpu().float(), self.n_bits, self.group_size)
        self.packed_weight.copy_(packed.to(self.packed_weight.device))
        self.weight_scales.copy_(scales.to(self.weight_scales.device))

    def dequantize(self) -> torch.Tensor:
        """Dequantize packed weights to bf16."""
        return dequant_Nbit_gpu(
            self.packed_weight, self.weight_scales,
            self.n_bits, self.out_features, self.in_features,
            self.group_size, output_dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_weight:
            if self._cached_weight is None:
                self._cached_weight = self.dequantize()
            weight = self._cached_weight
        else:
            # Dequantize on every forward (memory-efficient but slower)
            weight = self.dequantize()
        return F.linear(x, weight, self.bias)

    def evict_cache(self):
        """Free the cached dequantized weight to reclaim memory."""
        self._cached_weight = None

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'n_bits={self.n_bits}, bias={self.bias is not None}')


# ---------------------------------------------------------------------------
# Save a DynaQuant-packed model
# ---------------------------------------------------------------------------

def save_packed_model(
    model_path: str,
    recipe: Dict[str, int],
    output_dir: str,
    group_size: int = 16,
):
    """Load a bf16 model, quantize per recipe, save in packed format.

    Args:
        model_path: path to the source bf16 HF model
        recipe: dict of {param_name: n_bits} from the allocator
        output_dir: where to write the packed model
        group_size: quantization group size (default 16 for NVFP4 compat)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    staged, cleanup = stage_multimodal(model_path)
    try:
        print(f"[pack] loading {staged}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            staged, torch_dtype=torch.bfloat16, device_map="cpu",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        packed_dir = out_path / "packed_weights"
        packed_dir.mkdir(exist_ok=True)

        # Save the FULL model first (original weights preserved for tied params
        # like lm_head ↔ embed_tokens). Packed weights take priority at load time.
        print(f"[pack] saving model skeleton + original weights", flush=True)
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        # Quantize and pack each weight per recipe
        quantized_names = set()
        total_original_bytes = 0
        total_packed_bytes = 0

        for full_name, mod, attr in iter_quantizable_tensors(model):
            if full_name not in recipe:
                continue
            n_bits = recipe[full_name]
            if n_bits >= 16:
                continue  # leave at bf16, save normally

            param = getattr(mod, attr)
            weight = param.data.float()

            # Handle 3D (fused MoE) by flattening
            orig_shape = weight.shape
            if weight.dim() == 3:
                E, out_f, in_f = weight.shape
                weight_2d = weight.reshape(E * out_f, in_f)
            else:
                weight_2d = weight
                out_f, in_f = weight.shape

            # Pad input dim to group_size if needed
            if in_f % group_size != 0:
                pad = group_size - (in_f % group_size)
                weight_2d = F.pad(weight_2d, (0, pad))

            packed, scales = pack_Nbit_tensor(weight_2d, n_bits, group_size)

            # Save packed data
            safe_name = full_name.replace("/", "_").replace(".", "_")
            torch.save(packed, packed_dir / f"{safe_name}.packed")
            torch.save(scales, packed_dir / f"{safe_name}.scales")

            # Don't zero weights — they're saved via save_pretrained first,
            # and tied weights (lm_head ↔ embed_tokens) would corrupt both.
            quantized_names.add(full_name)

            original_bytes = weight.numel() * 2  # bf16
            packed_bytes = packed.numel() + scales.numel() * 4
            total_original_bytes += original_bytes
            total_packed_bytes += packed_bytes

        print(f"[pack] packed {len(quantized_names)} weights: "
              f"{total_original_bytes/1e9:.2f} GB → {total_packed_bytes/1e9:.2f} GB "
              f"({total_packed_bytes/total_original_bytes*100:.0f}%)", flush=True)

        # Model already saved above — just write the DynaQuant config

        # Save DynaQuant config
        dynaquant_config = {
            "format": "dynaquant_packed_v1",
            "group_size": group_size,
            "recipe": recipe,
            "quantized_params": list(quantized_names),
            "total_original_bytes": total_original_bytes,
            "total_packed_bytes": total_packed_bytes,
            "compression_ratio": total_original_bytes / max(1, total_packed_bytes),
        }
        with open(out_path / "dynaquant_config.json", "w") as f:
            json.dump(dynaquant_config, f, indent=2)

        print(f"[pack] saved to {output_dir}", flush=True)
        print(f"[pack] compression: {dynaquant_config['compression_ratio']:.1f}×", flush=True)

    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


# ---------------------------------------------------------------------------
# Load a DynaQuant-packed model
# ---------------------------------------------------------------------------

def load_packed_model(model_dir: str, device: str = "cuda"):
    """Load a DynaQuant-packed model for inference.

    Replaces quantized nn.Linear modules with DynaQuantLinear that
    stores packed weights and dequantizes on forward().

    Args:
        model_dir: path to a DynaQuant-packed model directory
        device: target device

    Returns:
        model, tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_dir)
    with open(model_path / "dynaquant_config.json") as f:
        config = json.load(f)

    recipe = config["recipe"]
    group_size = config["group_size"]
    quantized_params = set(config["quantized_params"])
    packed_dir = model_path / "packed_weights"

    # Load the model skeleton (unpacked weights load normally)
    staged, cleanup = stage_multimodal(str(model_path))
    try:
        model = AutoModelForCausalLM.from_pretrained(
            staged, torch_dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)

    # Replace quantized Linears with DynaQuantLinear
    n_replaced = 0
    for full_name, mod, attr in iter_quantizable_tensors(model):
        if full_name not in quantized_params:
            continue

        n_bits = recipe[full_name]
        param = getattr(mod, attr)

        if isinstance(mod, nn.Linear) and attr == "weight":
            # Replace the whole Linear module
            parent_name = full_name.rsplit(".", 1)[0]  # strip ".weight" → "model.layers.N.attn.q_proj"
            parent_parts = parent_name.split(".")
            parent = model
            for part in parent_parts[:-1]:
                parent = getattr(parent, part)

            old_linear = getattr(parent, parent_parts[-1])
            dq_linear = DynaQuantLinear(
                old_linear.in_features, old_linear.out_features,
                n_bits, group_size,
                bias=old_linear.bias is not None,
                device=device,
            )
            if old_linear.bias is not None:
                dq_linear.bias.copy_(old_linear.bias)

            # Load packed data
            safe_name = full_name.replace("/", "_").replace(".", "_")
            packed = torch.load(packed_dir / f"{safe_name}.packed",
                                map_location=device, weights_only=True)
            scales = torch.load(packed_dir / f"{safe_name}.scales",
                                map_location=device, weights_only=True)
            dq_linear.packed_weight.copy_(packed)
            dq_linear.weight_scales.copy_(scales)

            setattr(parent, parent_parts[-1], dq_linear)
            n_replaced += 1

    print(f"[load] replaced {n_replaced} Linears with DynaQuantLinear", flush=True)
    return model, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    pack_p = sub.add_parser("pack", help="Pack a model from a Pareto recipe")
    pack_p.add_argument("--model", required=True)
    pack_p.add_argument("--pareto", required=True)
    pack_p.add_argument("--step", default="knee")
    pack_p.add_argument("--output", required=True)

    load_p = sub.add_parser("test-load", help="Test-load a packed model")
    load_p.add_argument("--model-dir", required=True)

    args = parser.parse_args()

    if args.command == "pack":
        with open(args.pareto) as f:
            pareto_data = json.load(f)
        pareto = pareto_data["pareto"]
        if args.step == "knee":
            entry = min(pareto, key=lambda p: abs(p["step"] - pareto_data["knee_step"]))
        else:
            step = int(args.step)
            entry = min(pareto, key=lambda p: abs(p["step"] - step))
        recipe = entry["recipe"]
        print(f"Recipe: step {entry['step']}, cost {entry['cost_bytes']/1e9:.2f} GB")
        save_packed_model(args.model, recipe, args.output)

    elif args.command == "test-load":
        model, tokenizer = load_packed_model(args.model_dir)
        print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
        # Quick generation test
        inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        print(f"Generation: {tokenizer.decode(out[0])}")
