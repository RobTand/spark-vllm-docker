#!/usr/bin/env python3
"""
joint_allocator_continuous.py — Continuous optimization over (w, s, g)

Treats all three dimensions as continuous variables:
  - w: weight bits (continuous, e.g., 3.5)
  - s: scale bits (continuous, e.g., 6.2)
  - g: group size (continuous, e.g., 48.0)

Uses Lagrangian optimization to find the true optimum, then discretizes
to hardware-supported formats only at export time.

Analytical model:
  MSE(w, s, g) ≈ weight_noise(w) * group_factor(g) + scale_noise(s)

  weight_noise(w) = 1 / (2^(w-1) - 1)²
  scale_noise(s) = 1 / (2^(s-1) - 1)²
  group_factor(g) = sqrt(g / g_ref)  # larger groups = worse fit

  memory(w, s, g) = numel * (w + s/g) / 8
"""

import argparse
import json
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from typing import Dict, List, Tuple


# Bounds for continuous optimization
W_MIN, W_MAX = 2.0, 16.0
S_MIN, S_MAX = 2.0, 16.0
G_MIN, G_MAX = 8.0, 4096.0

# Reference group size for normalization
G_REF = 32.0


def weight_noise(w: float) -> float:
    """Quantization noise from w-bit weights."""
    if w >= 16:
        return 0.0
    qmax = 2 ** (w - 1) - 1
    if qmax <= 0:
        return 1e6  # Very high noise for < 1 bit
    return 1.0 / (qmax ** 2)


def scale_noise(s: float) -> float:
    """Quantization noise from s-bit scales."""
    if s >= 16:
        return 0.0
    qmax = 2 ** (s - 1) - 1
    if qmax <= 0:
        return 1e6
    return 1.0 / (qmax ** 2)


def group_factor(g: float) -> float:
    """How group size affects weight quantization error.

    Larger groups = scale fits local values worse = higher noise.
    """
    return math.sqrt(g / G_REF)


def mse_analytical(w: float, s: float, g: float) -> float:
    """Analytical MSE estimate for continuous (w, s, g)."""
    return weight_noise(w) * group_factor(g) + scale_noise(s) * 0.1


def memory_bytes(numel: int, w: float, s: float, g: float) -> float:
    """Memory in bytes for continuous (w, s, g)."""
    # Weight storage
    weight_bytes = numel * w / 8
    # Scale storage: one scale per group
    n_groups = numel / g
    scale_bytes = n_groups * s / 8
    return weight_bytes + scale_bytes


def bpw(w: float, s: float, g: float) -> float:
    """Effective bits per weight."""
    return w + s / g


def optimal_config_for_bpw(target_bpw: float, numel: int, sensitivity: float) -> Tuple[float, float, float, float]:
    """Find optimal (w, s, g) for a given target bpw.

    Given constraint: w + s/g = target_bpw
    Minimize: sensitivity * MSE(w, s, g)

    Returns: (w, s, g, mse)
    """
    def objective(params):
        w, s = params
        # Solve for g from constraint: g = s / (target_bpw - w)
        if target_bpw <= w:
            return 1e10  # Infeasible
        g = s / (target_bpw - w)
        if g < G_MIN or g > G_MAX:
            return 1e10  # Out of bounds
        return mse_analytical(w, s, g)

    best_mse = float('inf')
    best_config = (W_MIN, S_MIN, G_MIN)

    # Grid search over w and s, solve for g
    for w in np.linspace(W_MIN, min(W_MAX, target_bpw - 0.1), 20):
        for s in np.linspace(S_MIN, S_MAX, 20):
            if target_bpw <= w:
                continue
            g = s / (target_bpw - w)
            if g < G_MIN or g > G_MAX:
                continue
            mse = mse_analytical(w, s, g)
            if mse < best_mse:
                best_mse = mse
                best_config = (w, s, g)

    return (*best_config, best_mse)


def load_hawq_sensitivity(path: str) -> List[Dict]:
    """Load HAWQ sensitivity data."""
    with open(path) as f:
        data = json.load(f)

    layers = []
    if "sensitivity" in data:
        for name, entry in data["sensitivity"].items():
            h_trace = entry.get("h_trace", 0.0)
            w_norm_sq = entry.get("w_norm_sq", 0.0)
            numel = entry.get("numel", 1)
            sensitivity = h_trace * w_norm_sq / max(1, numel)

            layers.append({
                "name": name,
                "numel": numel,
                "sensitivity": sensitivity,
            })
    else:
        raise ValueError(f"Unknown format in {path}")

    return layers


def optimal_layer_config(L: Dict, lambda_val: float) -> Tuple[float, float, float, float, float]:
    """Find optimal (w, s, g) for a layer given Lagrange multiplier lambda.

    Minimizes: sensitivity * MSE(w,s,g) + lambda * memory(w,s,g)

    Uses vectorized numpy for speed.
    Returns: (w, s, g, mse, memory)
    """
    numel = L["numel"]
    sens = L["sensitivity"]

    # Vectorized grid
    w_vals = np.linspace(W_MIN, W_MAX, 15)
    s_vals = np.linspace(S_MIN, S_MAX, 10)
    g_vals = np.logspace(np.log10(G_MIN), np.log10(G_MAX), 10)

    # Create meshgrid
    W, S, G = np.meshgrid(w_vals, s_vals, g_vals, indexing='ij')

    # Vectorized MSE
    qmax_w = 2 ** (W - 1) - 1
    qmax_s = 2 ** (S - 1) - 1
    w_noise = 1.0 / (qmax_w ** 2)
    s_noise = 1.0 / (qmax_s ** 2)
    g_factor = np.sqrt(G / G_REF)
    MSE = w_noise * g_factor + s_noise * 0.1

    # Vectorized memory
    MEM = numel * (W + S / G) / 8

    # Cost function
    COST = sens * MSE + lambda_val * MEM

    # Find minimum
    idx = np.argmin(COST)
    w = W.flat[idx]
    s = S.flat[idx]
    g = G.flat[idx]

    return w, s, g, MSE.flat[idx], MEM.flat[idx]


def water_fill_continuous(layers: List[Dict], n_points: int = 50) -> List[Dict]:
    """Lagrangian optimization over continuous (w, s, g) space.

    Uses binary search on lambda (memory price) to hit different
    total memory budgets, producing the Pareto frontier.
    """
    total_numel = sum(L["numel"] for L in layers)

    # Normalize lambda by total model size so it's scale-invariant
    # lambda_norm is "error reduction per bit of memory"
    pareto_curve = []

    # Sweep lambda from high (prefer low memory) to low (prefer low error)
    for lambda_norm in np.logspace(4, -6, n_points):
        # Scale lambda by inverse of total elements
        lambda_val = lambda_norm / total_numel
        total_error = 0.0
        total_mem = 0.0
        recipe = {}

        for L in layers:
            w, s, g, mse, mem = optimal_layer_config(L, lambda_val)
            total_error += L["sensitivity"] * mse
            total_mem += mem
            recipe[L["name"]] = (round(w, 2), round(s, 2), round(g, 1))

        pareto_curve.append({
            "lambda": lambda_val,
            "actual_bpw": round(total_mem * 8 / total_numel, 2),
            "cost_bytes": int(total_mem),
            "weighted_error": total_error,
            "recipe": recipe,
        })

    # Sort by memory
    pareto_curve.sort(key=lambda p: p["cost_bytes"])

    return pareto_curve


def discretize_config(w: float, s: float, g: float,
                      w_options: List[int] = None,
                      s_options: List[int] = None,
                      g_options: List[int] = None) -> Tuple[int, int, int]:
    """Round continuous config to nearest hardware-supported values."""
    if w_options is None:
        w_options = [2, 3, 4, 5, 6, 8, 16]
    if s_options is None:
        s_options = [4, 8, 16]
    if g_options is None:
        g_options = [16, 32, 64, 128, 256, 512, 1024, 2048]

    w_disc = min(w_options, key=lambda x: abs(x - w))
    s_disc = min(s_options, key=lambda x: abs(x - s))
    g_disc = min(g_options, key=lambda x: abs(x - g))

    return w_disc, s_disc, g_disc


def main():
    parser = argparse.ArgumentParser(description="Continuous 3D optimizer")
    parser.add_argument("--sensitivity", required=True, help="HAWQ sensitivity JSON")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument("--bpw-step", type=float, default=0.1, help="BPW step size")
    args = parser.parse_args()

    print("=" * 70)
    print("Continuous (w, s, g) Optimizer")
    print("=" * 70)

    print(f"\nLoading HAWQ sensitivity from {args.sensitivity}...")
    layers = load_hawq_sensitivity(args.sensitivity)
    print(f"Loaded {len(layers)} layers")

    total_numel = sum(L["numel"] for L in layers)
    print(f"Total elements: {total_numel:,}")

    print(f"\nOptimizing over continuous space...")
    pareto_curve = water_fill_continuous(layers, n_points=50)
    print(f"Generated {len(pareto_curve)} Pareto points")

    # Show results
    print("\n" + "=" * 70)
    print("CONTINUOUS PARETO FRONTIER")
    print("=" * 70)
    print(f"{'BPW':>6} {'Memory (GB)':>12} {'Error':>12}")
    print("-" * 34)

    for i, p in enumerate(pareto_curve):
        if i % 10 == 0 or i == len(pareto_curve) - 1:
            mem_gb = p['cost_bytes'] / 1e9
            print(f"{p['actual_bpw']:>6.2f} {mem_gb:>12.2f} {p['weighted_error']:>12.2e}")

    # Show example configs at key points
    print("\n" + "=" * 70)
    print("OPTIMAL CONTINUOUS CONFIGS (sample layers)")
    print("=" * 70)

    for target in [3.0, 4.0, 5.0, 6.0]:
        closest = min(pareto_curve, key=lambda p: abs(p['actual_bpw'] - target))
        print(f"\nAt ~{closest['actual_bpw']:.1f} bpw:")

        # Show a few example configs
        for name, cfg in list(closest['recipe'].items())[:5]:
            w, s, g = cfg
            w_d, s_d, g_d = discretize_config(w, s, g)
            print(f"  {name[:50]:<50}")
            print(f"    continuous: w={w:.1f}, s={s:.1f}, g={g:.0f}")
            print(f"    discretized: w={w_d}, s={s_d}, g={g_d}")

    # Save
    with open(args.output, "w") as f:
        json.dump({
            "n_layers": len(layers),
            "total_elements": total_numel,
            "pareto_curve": pareto_curve,
        }, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
