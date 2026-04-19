#!/usr/bin/env python3
"""
grouping_strategy_test.py — Compare grouping strategies on real MoE weights.

Tests:
  1. Contiguous grouping (current approach)
  2. Sorted grouping (by magnitude)
  3. Clustered grouping (k-means on magnitude)

Goal: Find which strategy gives best MSE at same bits/weight.
"""

import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open
import time


def quantize_with_grouping(weights: torch.Tensor, w_bits: int, g_size: int,
                           strategy: str = "contiguous") -> tuple:
    """
    Quantize weights using specified grouping strategy.

    Returns: (mse, bits_per_weight, extra_overhead_bits)
    """
    weights = weights.flatten().float()
    n = weights.numel()

    # Pad to multiple of g_size
    if n % g_size != 0:
        pad = g_size - (n % g_size)
        weights = torch.cat([weights, torch.zeros(pad)])
        n = weights.numel()

    n_groups = n // g_size

    if strategy == "contiguous":
        groups = weights.view(n_groups, g_size)
        index_overhead = 0

    elif strategy == "sorted":
        # Sort by absolute magnitude
        sorted_idx = weights.abs().argsort()
        sorted_weights = weights[sorted_idx]
        groups = sorted_weights.view(n_groups, g_size)
        # Index overhead: need to store permutation (log2(n) bits per element, or n*log2(n) total)
        # But we can reconstruct from sorting at load time - no storage needed!
        index_overhead = 0  # Reconstructible

    elif strategy == "sorted_store_idx":
        # Sorted with stored indices (if reconstruction is too slow)
        sorted_idx = weights.abs().argsort()
        sorted_weights = weights[sorted_idx]
        groups = sorted_weights.view(n_groups, g_size)
        # Store 16-bit indices
        index_overhead = 16  # bits per weight for index

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Quantize each group
    total_se = 0.0
    qmax = (1 << (w_bits - 1)) - 1

    for i in range(n_groups):
        g = groups[i]
        max_abs = g.abs().max().item()
        if max_abs < 1e-10:
            continue
        scale = max_abs / qmax
        codes = (g / scale).round().clamp(-qmax - 1, qmax)
        recon = codes * scale
        total_se += ((g - recon) ** 2).sum().item()

    mse = total_se / n

    # Bits per weight: w_bits + scale overhead (assuming bf16 scales)
    scale_bits = 16
    bits_per_weight = w_bits + scale_bits / g_size + index_overhead

    return mse, bits_per_weight


def analyze_distribution(weights: torch.Tensor) -> dict:
    """Compute distribution metrics for a weight tensor."""
    w = weights.flatten().float()
    abs_w = w.abs()

    return {
        "mean": w.mean().item(),
        "std": w.std().item(),
        "min": w.min().item(),
        "max": w.max().item(),
        "abs_mean": abs_w.mean().item(),
        "abs_std": abs_w.std().item(),
        "abs_max": abs_w.max().item(),
        "kurtosis": ((w - w.mean()) ** 4).mean().item() / (w.std() ** 4 + 1e-10) - 3,
        "sparsity": (abs_w < 1e-6).float().mean().item(),
        "outlier_ratio": (abs_w > 3 * abs_w.std()).float().mean().item(),
    }


def test_strategies_on_tensor(weights: torch.Tensor, name: str):
    """Test all strategies on a single tensor."""
    print(f"\n{'='*70}")
    print(f"Tensor: {name}")
    print(f"Shape: {list(weights.shape)}, Elements: {weights.numel()}")

    # Distribution analysis
    dist = analyze_distribution(weights)
    print(f"Distribution: std={dist['std']:.4f}, kurtosis={dist['kurtosis']:.2f}, "
          f"outlier_ratio={dist['outlier_ratio']:.4f}")

    print(f"\n{'Strategy':<20} {'w_bits':>6} {'g_size':>8} {'Bits/W':>8} {'MSE':>12} {'Speedup':>10}")
    print("-" * 70)

    results = []

    for w_bits in [3, 4, 5]:
        for g_size in [16, 64, 128, 512]:
            for strategy in ["contiguous", "sorted"]:
                mse, bpw = quantize_with_grouping(weights, w_bits, g_size, strategy)
                results.append({
                    "strategy": strategy,
                    "w_bits": w_bits,
                    "g_size": g_size,
                    "bpw": bpw,
                    "mse": mse,
                })

    # Sort by bits/weight, then by MSE
    results.sort(key=lambda x: (x["bpw"], x["mse"]))

    # Find best MSE for each (w_bits, g_size) pair to compute speedup
    contiguous_mse = {}
    for r in results:
        if r["strategy"] == "contiguous":
            key = (r["w_bits"], r["g_size"])
            contiguous_mse[key] = r["mse"]

    for r in results:
        key = (r["w_bits"], r["g_size"])
        baseline = contiguous_mse.get(key, r["mse"])
        speedup = baseline / r["mse"] if r["mse"] > 0 else float('inf')
        speedup_str = f"{speedup:.2f}x" if r["strategy"] == "sorted" else "-"

        print(f"{r['strategy']:<20} {r['w_bits']:>6} {r['g_size']:>8} "
              f"{r['bpw']:>8.2f} {r['mse']:>12.2e} {speedup_str:>10}")

    # Summary: best config for each strategy at ~4 bits/weight
    print(f"\nBest configs at ~4 bits/weight:")
    for strategy in ["contiguous", "sorted"]:
        near_4 = [r for r in results if r["strategy"] == strategy and 3.5 <= r["bpw"] <= 4.5]
        if near_4:
            best = min(near_4, key=lambda x: x["mse"])
            print(f"  {strategy}: w{best['w_bits']}_g{best['g_size']} → MSE={best['mse']:.2e}")

    return results


def main():
    model_path = Path("/models/Qwen3.5-35B-A3B-bf16")

    # Find safetensor files
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        print(f"No safetensors found in {model_path}")
        return

    print(f"Found {len(st_files)} safetensor files")

    # Load a few expert weights from first shard
    experts_tested = 0
    max_experts = 5

    all_results = []

    for st_file in st_files[:3]:  # Check first 3 shards
        print(f"\nScanning {st_file.name}...")
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                # Look for MoE expert weights
                if "experts" in key and ".weight" in key and "gate" not in key:
                    if experts_tested >= max_experts:
                        break

                    weights = f.get_tensor(key)
                    if weights.dim() == 2:  # Only 2D weight matrices
                        results = test_strategies_on_tensor(weights, key)
                        all_results.extend(results)
                        experts_tested += 1

            if experts_tested >= max_experts:
                break

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    # Aggregate improvement from sorted vs contiguous
    improvements = []
    for r in all_results:
        if r["strategy"] == "sorted":
            key = (r["w_bits"], r["g_size"])
            # Find matching contiguous
            for r2 in all_results:
                if r2["strategy"] == "contiguous" and (r2["w_bits"], r2["g_size"]) == key:
                    if r2["mse"] > 0:
                        improvements.append(r2["mse"] / r["mse"])
                    break

    if improvements:
        print(f"\nSorted vs Contiguous improvement:")
        print(f"  Mean: {np.mean(improvements):.2f}x better MSE")
        print(f"  Min:  {np.min(improvements):.2f}x")
        print(f"  Max:  {np.max(improvements):.2f}x")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
