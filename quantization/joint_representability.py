#!/usr/bin/env python3
"""
joint_representability.py — Measure how well a group of weights can be
jointly represented as (scale × int_code) under varying precision.

The key question: given a group of values and a budget of (scale_bits, weight_bits),
what is the minimum reconstruction error achievable?

This is NOT about individual value quantization. It's about whether a GROUP
can be faithfully represented using a shared scale factor.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class RepresentabilityResult:
    """Result of representability analysis for a weight group."""
    group_size: int
    weight_bits: int
    scale_bits: int  # effective bits for scale (fp32=32, bf16=16, fp8=8)

    # Errors
    mse: float              # mean squared error
    max_error: float        # worst-case error (important for outliers)
    relative_error: float   # mse / variance of original

    # Group characteristics
    group_variance: float   # how spread out the group is
    magnitude_range: float  # max/min ratio (outlier indicator)

    # Memory cost
    bits_per_weight: float  # amortized including scale overhead


def quantize_group(weights: torch.Tensor, weight_bits: int,
                   scale_bits: int) -> Tuple[torch.Tensor, float, float]:
    """
    Quantize a group of weights with given precision for weights and scale.

    Returns: (reconstructed_weights, scale_used, mse)
    """
    # Compute optimal scale (max absolute value)
    max_abs = weights.abs().max().item()
    if max_abs < 1e-10:
        return torch.zeros_like(weights), 0.0, 0.0

    # Quantize the SCALE itself to scale_bits precision
    if scale_bits >= 32:
        scale = max_abs  # fp32, effectively exact
    elif scale_bits >= 16:
        # bf16: truncate mantissa to 7 bits
        scale = torch.tensor(max_abs, dtype=torch.bfloat16).float().item()
    elif scale_bits >= 8:
        # fp8 e4m3: very limited mantissa (3 bits)
        # Simulate by rounding to 3 mantissa bits
        if max_abs > 0:
            exp = np.floor(np.log2(max_abs))
            mantissa = max_abs / (2 ** exp)
            # Round mantissa to 3 bits (8 levels in [1, 2))
            mantissa_q = np.round(mantissa * 8) / 8
            scale = float(mantissa_q * (2 ** exp))
        else:
            scale = 0.0
    else:
        # Very low precision scale (4 bits, etc.) - experimental
        if max_abs > 0:
            exp = np.floor(np.log2(max_abs))
            mantissa = max_abs / (2 ** exp)
            levels = 2 ** (scale_bits - 1)  # half for mantissa
            mantissa_q = np.round(mantissa * levels) / levels
            scale = float(mantissa_q * (2 ** exp))
        else:
            scale = 0.0

    if scale < 1e-10:
        return torch.zeros_like(weights), 0.0, 0.0

    # Quantize weights to integer codes
    qmax = (1 << (weight_bits - 1)) - 1
    codes_float = weights / scale * qmax
    codes_int = codes_float.round().clamp(-qmax - 1, qmax)

    # Reconstruct
    reconstructed = codes_int / qmax * scale

    # Compute error
    mse = ((weights - reconstructed) ** 2).mean().item()

    return reconstructed, scale, mse


def analyze_group(weights: torch.Tensor,
                  weight_bits: int,
                  scale_bits: int,
                  group_size: int) -> RepresentabilityResult:
    """
    Analyze representability of a weight tensor under given configuration.

    Reshapes weights into groups of group_size, quantizes each group,
    and measures reconstruction quality.
    """
    weights = weights.flatten().float()
    n = weights.numel()

    # Pad to multiple of group_size
    if n % group_size != 0:
        pad = group_size - (n % group_size)
        weights = torch.cat([weights, torch.zeros(pad)])

    n_groups = weights.numel() // group_size
    groups = weights.view(n_groups, group_size)

    total_mse = 0.0
    max_errors = []
    group_variances = []
    magnitude_ranges = []

    for g in range(n_groups):
        group = groups[g]
        reconstructed, scale, mse = quantize_group(group, weight_bits, scale_bits)

        total_mse += mse

        errors = (group - reconstructed).abs()
        max_errors.append(errors.max().item())

        var = group.var().item()
        group_variances.append(var)

        abs_vals = group.abs()
        min_abs = abs_vals[abs_vals > 1e-10].min().item() if (abs_vals > 1e-10).any() else 1e-10
        max_abs = abs_vals.max().item()
        magnitude_ranges.append(max_abs / min_abs if min_abs > 0 else float('inf'))

    avg_mse = total_mse / n_groups
    avg_variance = np.mean(group_variances)

    # Memory cost: weight bits + amortized scale bits
    scale_bytes_per_group = scale_bits / 8
    weight_bytes_per_group = group_size * weight_bits / 8
    total_bytes_per_group = scale_bytes_per_group + weight_bytes_per_group
    bits_per_weight = total_bytes_per_group * 8 / group_size

    return RepresentabilityResult(
        group_size=group_size,
        weight_bits=weight_bits,
        scale_bits=scale_bits,
        mse=avg_mse,
        max_error=np.max(max_errors),
        relative_error=avg_mse / avg_variance if avg_variance > 1e-10 else 0.0,
        group_variance=avg_variance,
        magnitude_range=np.median(magnitude_ranges),
        bits_per_weight=bits_per_weight,
    )


def sweep_configurations(weights: torch.Tensor,
                         weight_bits_range: List[int] = [2, 3, 4, 5, 6, 8],
                         scale_bits_range: List[int] = [8, 16, 32],
                         group_size_range: List[int] = [16, 64, 128, 512],
                         ) -> List[RepresentabilityResult]:
    """
    Sweep over all configurations and return results sorted by efficiency.
    """
    results = []

    for g in group_size_range:
        for w in weight_bits_range:
            for s in scale_bits_range:
                result = analyze_group(weights, w, s, g)
                results.append(result)

    return results


def find_pareto_frontier(results: List[RepresentabilityResult]
                         ) -> List[RepresentabilityResult]:
    """
    Find configurations on the Pareto frontier (bits_per_weight vs mse).
    """
    # Sort by bits_per_weight
    sorted_results = sorted(results, key=lambda r: r.bits_per_weight)

    pareto = []
    best_mse = float('inf')

    for r in sorted_results:
        if r.mse < best_mse:
            pareto.append(r)
            best_mse = r.mse

    return pareto


def print_analysis(weights: torch.Tensor, name: str = "weights"):
    """Pretty-print analysis of a weight tensor."""
    print(f"\n{'='*70}")
    print(f"Joint Representability Analysis: {name}")
    print(f"Shape: {list(weights.shape)}, Elements: {weights.numel()}")
    print(f"{'='*70}\n")

    results = sweep_configurations(weights)
    pareto = find_pareto_frontier(results)

    print("Pareto-optimal configurations (bits_per_weight vs MSE):\n")
    print(f"{'Config':<20} {'Bits/W':>8} {'MSE':>12} {'MaxErr':>10} {'RelErr':>10}")
    print("-" * 62)

    for r in pareto:
        config = f"w{r.weight_bits}_s{r.scale_bits}_g{r.group_size}"
        print(f"{config:<20} {r.bits_per_weight:>8.2f} {r.mse:>12.2e} "
              f"{r.max_error:>10.2e} {r.relative_error:>10.4f}")

    print(f"\n{'='*70}")
    print("Key insight: Compare configs at same bits_per_weight")
    print("  - Higher scale_bits + lower weight_bits?")
    print("  - Or lower scale_bits + higher weight_bits?")
    print("  - Answer depends on group variance and outliers")
    print(f"{'='*70}\n")

    # Find interesting comparisons at similar memory cost
    print("Configurations at ~4 bits per weight:\n")
    near_4bit = [r for r in results if 3.5 <= r.bits_per_weight <= 4.5]
    near_4bit.sort(key=lambda r: r.mse)

    print(f"{'Config':<20} {'Bits/W':>8} {'MSE':>12} {'GroupVar':>12}")
    print("-" * 54)
    for r in near_4bit[:8]:
        config = f"w{r.weight_bits}_s{r.scale_bits}_g{r.group_size}"
        print(f"{config:<20} {r.bits_per_weight:>8.2f} {r.mse:>12.2e} "
              f"{r.group_variance:>12.2e}")

    return results, pareto


# Quick test with synthetic data
if __name__ == "__main__":
    torch.manual_seed(42)

    # Test 1: Uniform weights (easy case - scale doesn't matter much)
    print("\n" + "="*70)
    print("TEST 1: Uniform magnitude weights (all similar scale)")
    w_uniform = torch.randn(1536, 3072) * 0.01  # small, uniform
    print_analysis(w_uniform, "Uniform weights")

    # Test 2: Weights with outliers (hard case - scale matters a lot)
    print("\n" + "="*70)
    print("TEST 2: Weights with outliers (some values 100x larger)")
    w_outlier = torch.randn(1536, 3072) * 0.01
    # Add outliers to 1% of values
    outlier_mask = torch.rand_like(w_outlier) < 0.01
    w_outlier[outlier_mask] *= 100
    print_analysis(w_outlier, "Outlier weights")

    # Test 3: Bimodal weights (two distinct scales)
    print("\n" + "="*70)
    print("TEST 3: Bimodal weights (half small, half large)")
    w_bimodal = torch.randn(1536, 3072)
    w_bimodal[:768] *= 0.01
    w_bimodal[768:] *= 1.0
    print_analysis(w_bimodal, "Bimodal weights")
