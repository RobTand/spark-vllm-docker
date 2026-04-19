#!/usr/bin/env python3
"""
joint_hawq_diversity.py — Test (s_bits, g_size) dominance across layer types

Tests:
  - Attention: q_proj, k_proj, v_proj, o_proj
  - MoE gate_up_proj experts
  - MoE down_proj experts
  - Shared experts (if present)
  - Dense layers (embeddings, LM head)

Goal: Validate that s8_g16 dominates across ALL layer types, not just MoE experts.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path
from safetensors import safe_open
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

N_WORKERS = os.cpu_count() or 20


@dataclass
class Config:
    w_bits: int
    s_bits: int
    g_size: int

    def __hash__(self):
        return hash((self.w_bits, self.s_bits, self.g_size))

    def __str__(self):
        return f"w{self.w_bits}_s{self.s_bits}_g{self.g_size}"


@dataclass
class ConfigResult:
    config: Config
    mse: float
    memory_bytes: int
    bits_per_weight: float


W_BITS_RANGE = [3, 4, 5, 6, 8]
S_BITS_RANGE = [4, 8, 16]
G_SIZE_RANGE = [16, 32, 64, 128, 256, 512, 1024, 2048]


def compute_memory(n_elements: int, w_bits: int, s_bits: int, g_size: int) -> int:
    weight_bits = n_elements * w_bits
    weight_bytes = (weight_bits + 7) // 8
    n_groups = (n_elements + g_size - 1) // g_size
    scale_bytes = n_groups * (s_bits // 8)
    return weight_bytes + scale_bytes


def compute_bits_per_weight(w_bits: int, s_bits: int, g_size: int) -> float:
    return w_bits + s_bits / g_size


def quantize_and_measure(weights: torch.Tensor, config: Config) -> float:
    w_bits, s_bits, g_size = config.w_bits, config.s_bits, config.g_size

    weights = weights.flatten().float()
    n = weights.numel()

    if n % g_size != 0:
        pad = g_size - (n % g_size)
        weights = torch.cat([weights, torch.zeros(pad)])
        n = weights.numel()

    n_groups = n // g_size
    groups = weights.view(n_groups, g_size)

    qmax = (1 << (w_bits - 1)) - 1
    total_se = 0.0

    for i in range(n_groups):
        g = groups[i]
        max_abs = g.abs().max().item()
        if max_abs < 1e-10:
            continue

        raw_scale = max_abs / qmax
        if s_bits >= 16:
            scale = torch.tensor(raw_scale, dtype=torch.bfloat16).float().item()
        elif s_bits >= 8:
            if raw_scale > 0:
                exp = np.floor(np.log2(raw_scale + 1e-10))
                mantissa = raw_scale / (2 ** exp)
                mantissa_q = round(mantissa * 8) / 8
                scale = float(mantissa_q * (2 ** exp))
            else:
                scale = 0.0
        else:
            if raw_scale > 0:
                exp = np.floor(np.log2(raw_scale + 1e-10))
                mantissa = raw_scale / (2 ** exp)
                mantissa_q = round(mantissa * 2) / 2
                scale = float(mantissa_q * (2 ** exp))
            else:
                scale = 0.0

        if scale < 1e-10:
            continue

        codes = (g / scale).round().clamp(-qmax - 1, qmax)
        recon = codes * scale
        total_se += ((g - recon) ** 2).sum().item()

    return total_se / weights.numel()


def _eval_single_config(args):
    weights_flat, n_elements, w_bits, s_bits, g_size = args
    config = Config(w_bits, s_bits, g_size)
    weights = torch.from_numpy(weights_flat)
    mse = quantize_and_measure(weights, config)
    memory = compute_memory(n_elements, w_bits, s_bits, g_size)
    bpw = compute_bits_per_weight(w_bits, s_bits, g_size)
    return ConfigResult(config=config, mse=mse, memory_bytes=memory, bits_per_weight=bpw)


def evaluate_layer(weights: torch.Tensor, layer_name: str) -> Tuple[str, List[ConfigResult]]:
    n_elements = weights.numel()
    weights_flat = weights.flatten().float().numpy()

    configs_to_eval = []
    for w_bits in W_BITS_RANGE:
        for s_bits in S_BITS_RANGE:
            for g_size in G_SIZE_RANGE:
                configs_to_eval.append((weights_flat, n_elements, w_bits, s_bits, g_size))

    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_eval_single_config, args) for args in configs_to_eval]
        for future in as_completed(futures):
            results.append(future.result())

    # Compute Pareto frontier
    pareto = []
    for r in results:
        dominated = False
        for other in results:
            if other.mse < r.mse and other.memory_bytes < r.memory_bytes:
                dominated = True
                break
            if other.mse <= r.mse and other.memory_bytes < r.memory_bytes:
                dominated = True
                break
            if other.mse < r.mse and other.memory_bytes <= r.memory_bytes:
                dominated = True
                break
        if not dominated:
            pareto.append(r)

    return layer_name, pareto


def load_diverse_layers(model_path: Path, max_per_type: int = 3) -> Dict[str, List[Tuple[str, torch.Tensor]]]:
    """Load diverse layer types from Qwen3.5-35B."""
    layers_by_type = {
        "attention_qkv": [],
        "attention_o": [],
        "moe_gate_up": [],
        "moe_down": [],
        "shared_expert": [],
        "dense": [],
    }

    st_files = sorted(model_path.glob("*.safetensors"))

    for st_file in st_files:
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # Attention Q/K/V (packed together in some models)
                if "self_attn.q_proj" in key or "self_attn.k_proj" in key or "self_attn.v_proj" in key:
                    if len(layers_by_type["attention_qkv"]) < max_per_type:
                        layers_by_type["attention_qkv"].append((key, tensor))

                # Attention O
                elif "self_attn.o_proj" in key:
                    if len(layers_by_type["attention_o"]) < max_per_type:
                        layers_by_type["attention_o"].append((key, tensor))

                # MoE gate_up_proj
                elif "experts" in key and "gate_up_proj" in key:
                    if len(layers_by_type["moe_gate_up"]) < max_per_type:
                        # Extract single expert
                        if len(tensor.shape) == 3:  # [n_experts, out, in]
                            expert_w = tensor[0]  # First expert
                            layers_by_type["moe_gate_up"].append((f"{key}.expert0", expert_w))
                        else:
                            layers_by_type["moe_gate_up"].append((key, tensor))

                # MoE down_proj
                elif "experts" in key and "down_proj" in key:
                    if len(layers_by_type["moe_down"]) < max_per_type:
                        if len(tensor.shape) == 3:
                            expert_w = tensor[0]
                            layers_by_type["moe_down"].append((f"{key}.expert0", expert_w))
                        else:
                            layers_by_type["moe_down"].append((key, tensor))

                # Shared experts
                elif "shared_expert" in key and "weight" in key:
                    if len(layers_by_type["shared_expert"]) < max_per_type:
                        layers_by_type["shared_expert"].append((key, tensor))

                # Dense layers (embed, lm_head)
                elif "embed_tokens" in key or "lm_head" in key:
                    if len(layers_by_type["dense"]) < max_per_type:
                        layers_by_type["dense"].append((key, tensor))

        # Early exit if we have enough of each type
        all_full = all(len(v) >= max_per_type for v in layers_by_type.values() if len(v) > 0)
        if all_full:
            break

    return layers_by_type


def main():
    t0 = time.time()

    model_path = Path("/models/Qwen3.5-35B-A3B-bf16")

    print("="*70)
    print("Diversity Test: (s_bits, g_size) Dominance Across Layer Types")
    print("="*70)
    print()

    print("Loading diverse layer types...")
    layers_by_type = load_diverse_layers(model_path, max_per_type=3)

    for ltype, layers in layers_by_type.items():
        print(f"  {ltype}: {len(layers)} layers")
    print()

    # Collect Pareto frontiers per type
    pareto_by_type = {}

    for ltype, layers in layers_by_type.items():
        if not layers:
            continue

        print(f"Evaluating {ltype}...")
        type_pareto = []

        for name, weights in layers:
            print(f"  {name}: {weights.numel():,} elements")
            _, pareto = evaluate_layer(weights, name)
            type_pareto.extend(pareto)

        pareto_by_type[ltype] = type_pareto

    print()
    print("="*70)
    print("Results by Layer Type:")
    print("="*70)

    # Analyze (s_bits, g_size) frequency per type
    global_sg_counts = {}

    for ltype, pareto_configs in pareto_by_type.items():
        sg_counts = {}
        for cr in pareto_configs:
            key = f"s{cr.config.s_bits}_g{cr.config.g_size}"
            sg_counts[key] = sg_counts.get(key, 0) + 1
            global_sg_counts[key] = global_sg_counts.get(key, 0) + 1

        print(f"\n{ltype}:")
        top_configs = sorted(sg_counts.items(), key=lambda x: -x[1])[:5]
        for cfg, count in top_configs:
            print(f"  {cfg}: {count}")

    print()
    print("="*70)
    print("Global (s_bits, g_size) Ranking:")
    print("="*70)
    top_global = sorted(global_sg_counts.items(), key=lambda x: -x[1])[:10]
    for cfg, count in top_global:
        print(f"  {cfg}: {count}")

    # Check if one config dominates
    if top_global:
        winner = top_global[0][0]
        winner_count = top_global[0][1]
        total = sum(c for _, c in top_global)
        dominance = winner_count / total * 100
        print(f"\nDominant config: {winner} ({dominance:.1f}% of Pareto appearances)")

        if dominance > 30:
            print(f"\nRECOMMENDATION: Use {winner} as fixed (s_bits, g_size), optimize w_bits only")
        else:
            print("\nNo clear dominant config - may need per-layer (s_bits, g_size) selection")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
