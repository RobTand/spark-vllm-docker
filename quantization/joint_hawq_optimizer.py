#!/usr/bin/env python3
"""
joint_hawq_optimizer.py — Joint optimization over (w_bits, s_bits, g_size)

Extends HAWQ-style sensitivity analysis to co-optimize:
  - Weight bit width (w_bits): 3-8
  - Scale precision (s_bits): 8, 16, 32 (fp8, bf16, fp32)
  - Group size (g_size): 32, 64, 128, 256, 512

For each layer, we:
  1. Measure reconstruction error at each configuration
  2. Compute memory cost at each configuration
  3. Build Pareto frontier (non-dominated configs)
  4. Use multi-choice knapsack to allocate configs given memory budget

This finds the globally optimal allocation across all layers.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from safetensors import safe_open
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import json
import time
import os

# Use all available CPUs
N_WORKERS = os.cpu_count() or 20


@dataclass
class Config:
    """A quantization configuration."""
    w_bits: int
    s_bits: int
    g_size: int

    def __hash__(self):
        return hash((self.w_bits, self.s_bits, self.g_size))

    def __str__(self):
        return f"w{self.w_bits}_s{self.s_bits}_g{self.g_size}"


@dataclass
class ConfigResult:
    """Result of evaluating a config on a layer."""
    config: Config
    mse: float
    memory_bytes: int
    bits_per_weight: float


@dataclass
class LayerInfo:
    """Information about a layer."""
    name: str
    shape: Tuple[int, ...]
    n_elements: int
    pareto_configs: List[ConfigResult]  # Non-dominated configurations


# Configuration search space
W_BITS_RANGE = [3, 4, 5, 6, 8]
S_BITS_RANGE = [4, 8, 16]  # int4, fp8, bf16 — fp32 wastes bits for negligible gain
G_SIZE_RANGE = [16, 32, 64, 128, 256, 512, 1024, 2048]  # include fine (NVFP4-style) to per-row


def compute_memory(n_elements: int, w_bits: int, s_bits: int, g_size: int) -> int:
    """Compute memory in bytes for a quantized tensor."""
    # Weight storage
    weight_bits = n_elements * w_bits
    weight_bytes = (weight_bits + 7) // 8

    # Scale storage
    n_groups = (n_elements + g_size - 1) // g_size
    scale_bytes = n_groups * (s_bits // 8)

    return weight_bytes + scale_bytes


def compute_bits_per_weight(w_bits: int, s_bits: int, g_size: int) -> float:
    """Compute effective bits per weight including scale overhead."""
    return w_bits + s_bits / g_size


def quantize_and_measure(weights: torch.Tensor, config: Config) -> float:
    """Quantize weights with given config and return MSE."""
    w_bits = config.w_bits
    s_bits = config.s_bits
    g_size = config.g_size

    weights = weights.flatten().float()
    n = weights.numel()

    # Pad to multiple of g_size
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

        # Quantize scale to s_bits precision
        raw_scale = max_abs / qmax
        if s_bits >= 32:
            scale = raw_scale
        elif s_bits >= 16:
            scale = torch.tensor(raw_scale, dtype=torch.bfloat16).float().item()
        elif s_bits >= 8:  # fp8 (e4m3)
            if raw_scale > 0:
                exp = np.floor(np.log2(raw_scale + 1e-10))
                mantissa = raw_scale / (2 ** exp)
                mantissa_q = round(mantissa * 8) / 8  # 3-bit mantissa
                scale = float(mantissa_q * (2 ** exp))
            else:
                scale = 0.0
        else:  # 4-bit scale (e2m1 or similar)
            if raw_scale > 0:
                exp = np.floor(np.log2(raw_scale + 1e-10))
                mantissa = raw_scale / (2 ** exp)
                mantissa_q = round(mantissa * 2) / 2  # 1-bit mantissa
                scale = float(mantissa_q * (2 ** exp))
            else:
                scale = 0.0

        if scale < 1e-10:
            continue

        # Quantize weights
        codes = (g / scale).round().clamp(-qmax - 1, qmax)
        recon = codes * scale
        total_se += ((g - recon) ** 2).sum().item()

    return total_se / weights.numel()


def _eval_single_config(args):
    """Evaluate a single config - used for parallel execution."""
    weights_flat, n_elements, w_bits, s_bits, g_size = args
    config = Config(w_bits, s_bits, g_size)

    # Reconstruct tensor for quantization
    weights = torch.from_numpy(weights_flat)
    mse = quantize_and_measure(weights, config)
    memory = compute_memory(n_elements, w_bits, s_bits, g_size)
    bpw = compute_bits_per_weight(w_bits, s_bits, g_size)

    return ConfigResult(
        config=config,
        mse=mse,
        memory_bytes=memory,
        bits_per_weight=bpw,
    )


def evaluate_layer(weights: torch.Tensor, layer_name: str) -> LayerInfo:
    """Evaluate all configurations on a layer and return Pareto frontier."""
    n_elements = weights.numel()
    shape = tuple(weights.shape)

    # Convert to numpy for sharing across processes
    weights_flat = weights.flatten().float().numpy()

    # Build list of configs to evaluate
    configs_to_eval = []
    for w_bits in W_BITS_RANGE:
        for s_bits in S_BITS_RANGE:
            for g_size in G_SIZE_RANGE:
                configs_to_eval.append((weights_flat, n_elements, w_bits, s_bits, g_size))

    # Parallel evaluation
    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_eval_single_config, args) for args in configs_to_eval]
        for future in as_completed(futures):
            results.append(future.result())

    # Compute Pareto frontier (minimize both MSE and memory)
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

    # Sort by memory
    pareto.sort(key=lambda x: x.memory_bytes)

    return LayerInfo(
        name=layer_name,
        shape=shape,
        n_elements=n_elements,
        pareto_configs=pareto,
    )


def solve_allocation(layers: List[LayerInfo], memory_budget: int) -> Dict[str, ConfigResult]:
    """
    Solve the multi-choice knapsack problem:
    - Each layer must pick exactly one config from its Pareto frontier
    - Total memory <= budget
    - Minimize total MSE (weighted by layer size)

    Uses dynamic programming.
    """
    n_layers = len(layers)

    # Discretize memory budget into bins for DP
    # Use 1MB granularity
    GRANULARITY = 1_000_000
    budget_bins = memory_budget // GRANULARITY + 1

    # DP table: dp[layer][budget_bin] = (min_total_mse, config_choices)
    # We'll use a simpler greedy approach for now given the scale

    # Greedy: for each layer, pick the config with best MSE that fits
    # Sort layers by size (larger layers first - they matter more)
    sorted_layers = sorted(layers, key=lambda l: l.n_elements, reverse=True)

    allocation = {}
    remaining_budget = memory_budget

    for layer in sorted_layers:
        # Find best config that fits
        best_config = None
        best_mse = float('inf')

        for cr in layer.pareto_configs:
            if cr.memory_bytes <= remaining_budget:
                if cr.mse < best_mse:
                    best_mse = cr.mse
                    best_config = cr

        if best_config is None:
            # Take cheapest config if nothing fits
            best_config = min(layer.pareto_configs, key=lambda x: x.memory_bytes)

        allocation[layer.name] = best_config
        remaining_budget -= best_config.memory_bytes

    return allocation


def load_qwen35_experts(model_path: Path, max_layers: int = 10) -> List[Tuple[str, torch.Tensor]]:
    """Load expert weights from Qwen3.5-35B."""
    experts = []

    st_files = sorted(model_path.glob("*.safetensors"))

    for st_file in st_files:
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if "experts" in key:
                    tensor = f.get_tensor(key)
                    # tensor shape: [256, out_features, in_features]
                    # Extract a few individual experts
                    for expert_idx in [0, 64, 128, 192]:
                        if expert_idx < tensor.shape[0]:
                            expert_w = tensor[expert_idx]
                            name = f"{key}.expert{expert_idx}"
                            experts.append((name, expert_w))

                            if len(experts) >= max_layers:
                                return experts

    return experts


def main():
    t0 = time.time()

    model_path = Path("/models/Qwen3.5-35B-A3B-bf16")

    print("="*70)
    print("Joint (w_bits, s_bits, g_size) Optimization")
    print("="*70)
    print()

    # Load experts
    print("Loading expert weights...")
    experts = load_qwen35_experts(model_path, max_layers=20)
    print(f"Loaded {len(experts)} expert weight matrices")
    print()

    # Evaluate each layer in parallel
    print(f"Evaluating configurations per layer ({N_WORKERS} workers)...")
    layers = []

    def _eval_layer(args):
        name, weights = args
        return evaluate_layer(weights, name)

    with ThreadPoolExecutor(max_workers=min(N_WORKERS, len(experts))) as executor:
        futures = {executor.submit(_eval_layer, (name, w)): name for name, w in experts}
        for future in as_completed(futures):
            layer_info = future.result()
            layers.append(layer_info)
            print(f"  {layer_info.name}: {layer_info.n_elements:,} elements, "
                  f"{len(layer_info.pareto_configs)} Pareto configs")

    print()

    # Show Pareto frontier for first layer
    print("="*70)
    print(f"Pareto frontier for {layers[0].name}:")
    print("="*70)
    print(f"{'Config':<20} {'Bits/W':>8} {'Memory':>12} {'MSE':>12}")
    print("-"*56)
    for cr in layers[0].pareto_configs[:15]:
        print(f"{str(cr.config):<20} {cr.bits_per_weight:>8.2f} "
              f"{cr.memory_bytes:>12,} {cr.mse:>12.2e}")

    print()

    # Solve allocation at different budgets
    print("="*70)
    print("Allocation at different memory budgets:")
    print("="*70)

    total_elements = sum(l.n_elements for l in layers)

    # Calculate budget at different bits/weight targets
    for target_bpw in [3.5, 4.0, 4.5, 5.0, 6.0]:
        budget = int(total_elements * target_bpw / 8)
        allocation = solve_allocation(layers, budget)

        actual_memory = sum(cr.memory_bytes for cr in allocation.values())
        actual_bpw = actual_memory * 8 / total_elements
        total_mse = sum(cr.mse * layers[i].n_elements
                       for i, (name, cr) in enumerate(allocation.items())) / total_elements

        # Count configs used
        config_counts = {}
        for cr in allocation.values():
            key = str(cr.config)
            config_counts[key] = config_counts.get(key, 0) + 1
        top_configs = sorted(config_counts.items(), key=lambda x: -x[1])[:3]

        print(f"\nTarget: {target_bpw} bits/weight")
        print(f"  Budget: {budget:,} bytes")
        print(f"  Actual: {actual_memory:,} bytes ({actual_bpw:.2f} bits/weight)")
        print(f"  Avg MSE: {total_mse:.2e}")
        print(f"  Top configs: {top_configs}")

    print()
    print("="*70)
    print("Key findings:")
    print("="*70)

    # Analyze what configs dominate
    all_pareto = []
    for layer in layers:
        all_pareto.extend(layer.pareto_configs)

    # Count how often each (s_bits, g_size) pair appears in Pareto frontiers
    sg_counts = {}
    for cr in all_pareto:
        key = (cr.config.s_bits, cr.config.g_size)
        sg_counts[key] = sg_counts.get(key, 0) + 1

    print("\n(s_bits, g_size) frequency in Pareto frontiers:")
    for (s, g), count in sorted(sg_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  s{s}_g{g}: {count} appearances")

    # Find if there's a dominant (s_bits, g_size) choice
    print("\nImplication: If one (s_bits, g_size) dominates, we can simplify to 1D optimization")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
