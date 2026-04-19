#!/usr/bin/env python3
"""
joint_allocator.py — Water-fill allocator over (w_bits, s_bits, g_size) using
analytical noise model and HAWQ sensitivity.

Extends allocate_bits.py to 3D config space:
  - w_bits: 3, 4, 5, 6, 8
  - s_bits: 4, 8, 16 (int4, fp8, bf16 scales)
  - g_size: 16, 32, 64, 128, 256, 512, 1024, 2048

Uses analytical MSE model:
  MSE(w,s,g) ≈ noise_var(w) + scale_noise(s) × scale_factor(g)

where:
  noise_var(w) = 1 / (2^(w-1) - 1)²  (weight quantization noise)
  scale_noise(s) = precision_loss(s_bits)
  scale_factor(g) = depends on weight distribution within groups

The key insight: we DON'T need to evaluate every config by actually quantizing.
We model the noise analytically and use HAWQ sensitivity to weight layers.

Usage:
    python3 joint_allocator.py \\
        --sensitivity /tmp/curves/qwen35-4b-hawq.json \\
        --output /tmp/pareto/qwen35-4b-joint.json
"""

import argparse
import heapq
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


# Config space
W_BITS_OPTIONS = list(range(3, 17))  # [3, 4, 5, ..., 16]
S_BITS_OPTIONS = list(range(3, 17))  # [3, 4, 5, ..., 16]
G_SIZE_OPTIONS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def weight_noise_var(w_bits: int) -> float:
    """Noise variance from weight quantization (uniform symmetric)."""
    if w_bits >= 16:
        return 0.0
    qmax = (2 ** (w_bits - 1)) - 1
    # Uniform quantization noise: var = (scale/qmax)² / 12
    # Normalized by scale², this is 1 / (12 * qmax²)
    return 1.0 / (qmax ** 2)


def scale_noise_factor(s_bits: int) -> float:
    """Relative precision loss from scale quantization."""
    if s_bits >= 16:
        return 0.0  # bf16 scales - negligible error
    elif s_bits == 8:
        # fp8 e4m3: 3 mantissa bits = 1/8 relative precision
        return 0.125 ** 2
    else:
        # int4 with scale-of-scale: ~1/8 relative precision after SoS
        return 0.125 ** 2


def group_size_factor(g_size: int, w_max_abs: float, w_norm_sq: float, numel: int) -> float:
    """Factor capturing how group size affects quantization error.

    Smaller groups = scale fits local values better = lower noise.
    We model this as: larger groups have more variance in max-abs across groups.
    """
    # Approximate: noise scales with sqrt(g_size) due to central limit effects
    # on how well the group scale fits individual values
    # Normalized so g=16 has factor 1.0
    return math.sqrt(g_size / 16)


def compute_memory(numel: int, w_bits: int, s_bits: int, g_size: int) -> int:
    """Memory in bytes for quantized tensor."""
    weight_bytes = (numel * w_bits + 7) // 8
    n_groups = (numel + g_size - 1) // g_size
    if s_bits == 4:
        scale_bytes = (n_groups * 4 + 7) // 8 + 4  # +4 for scale-of-scale
    else:
        scale_bytes = n_groups * (s_bits // 8)
    return weight_bytes + scale_bytes


def compute_bpw(numel: int, w_bits: int, s_bits: int, g_size: int) -> float:
    """Effective bits per weight including scale overhead."""
    mem = compute_memory(numel, w_bits, s_bits, g_size)
    return mem * 8 / numel


def analytical_mse(w_bits: int, s_bits: int, g_size: int,
                   w_max_abs: float, w_norm_sq: float, numel: int) -> float:
    """Analytical MSE estimate for a config.

    MSE ≈ sensitivity × (weight_noise + scale_noise × group_factor)

    The sensitivity weighting happens at the allocator level.
    Here we just compute the raw noise contribution.
    """
    w_noise = weight_noise_var(w_bits)
    s_noise = scale_noise_factor(s_bits)
    g_factor = group_size_factor(g_size, w_max_abs, w_norm_sq, numel)

    # Combined noise: weight noise dominates, scale noise adds overhead
    # Scale noise is multiplied by expected |Q|² which is ~1 for normalized weights
    return w_noise * g_factor + s_noise


def build_config_ladder(numel: int, w_max_abs: float, w_norm_sq: float) -> List[Tuple]:
    """Build sorted list of (memory, mse, config) for a layer.

    Returns configs sorted by memory (ascending), with dominated configs removed.
    This is the layer's Pareto frontier.
    """
    configs = []
    for w in W_BITS_OPTIONS:
        for s in S_BITS_OPTIONS:
            for g in G_SIZE_OPTIONS:
                mem = compute_memory(numel, w, s, g)
                mse = analytical_mse(w, s, g, w_max_abs, w_norm_sq, numel)
                configs.append((mem, mse, (w, s, g)))

    # Sort by memory
    configs.sort(key=lambda x: x[0])

    # Remove dominated configs (higher memory AND higher MSE)
    pareto = []
    min_mse = float('inf')
    for mem, mse, cfg in configs:
        if mse < min_mse:
            pareto.append((mem, mse, cfg))
            min_mse = mse

    return pareto


def load_hawq_sensitivity(path: str) -> List[Dict]:
    """Load HAWQ sensitivity and build layer list."""
    with open(path) as f:
        data = json.load(f)

    layers = []
    if "sensitivity" in data:
        for name, entry in data["sensitivity"].items():
            h_trace = entry.get("h_trace", 0.0)
            w_norm_sq = entry.get("w_norm_sq", 0.0)
            w_max_abs = entry.get("w_max_abs", 1.0)
            numel = entry.get("numel", 1)

            # Sensitivity = h_trace × w_norm_sq / numel (validated correlation)
            sensitivity = h_trace * w_norm_sq / max(1, numel)

            # Build config ladder for this layer
            ladder = build_config_ladder(numel, w_max_abs, w_norm_sq)

            layers.append({
                "name": name,
                "numel": numel,
                "sensitivity": sensitivity,
                "w_max_abs": w_max_abs,
                "w_norm_sq": w_norm_sq,
                "ladder": ladder,  # List of (mem, mse, (w,s,g)) sorted by mem
            })
    else:
        raise ValueError(f"Unknown sensitivity format in {path}")

    return layers


def water_fill_3d(layers: List[Dict]) -> List[Dict]:
    """Water-fill over 3D config ladders to produce global Pareto frontier.

    Each layer has a ladder of configs sorted by memory.
    We start at config 0 (minimum memory) and upgrade layers greedily
    based on marginal utility per cost.
    """
    # State: current config index per layer
    state = {L["name"]: 0 for L in layers}
    lookup = {L["name"]: L for L in layers}

    def get_config(name: str) -> Tuple:
        L = lookup[name]
        idx = state[name]
        return L["ladder"][idx]  # (mem, mse, (w,s,g))

    def total_cost() -> int:
        return sum(get_config(n)[0] for n in state)

    def total_error() -> float:
        return sum(lookup[n]["sensitivity"] * get_config(n)[1] for n in state)

    def total_elements() -> int:
        return sum(lookup[n]["numel"] for n in state)

    def marginal_score(name: str) -> Tuple[float, int]:
        """Score for upgrading to next config. Returns (neg_score, next_idx)."""
        L = lookup[name]
        cur_idx = state[name]

        if cur_idx >= len(L["ladder"]) - 1:
            return (0.0, cur_idx)  # Already at max

        cur_mem, cur_mse, _ = L["ladder"][cur_idx]
        nxt_mem, nxt_mse, _ = L["ladder"][cur_idx + 1]

        d_error = L["sensitivity"] * (cur_mse - nxt_mse)  # Error reduction
        d_cost = nxt_mem - cur_mem  # Memory increase

        if d_cost <= 0:
            return (-float('inf'), cur_idx + 1)  # Free upgrade

        return (-d_error / d_cost, cur_idx + 1)  # Negative for max-heap

    # Build initial heap
    heap = []
    for L in layers:
        neg_score, nxt_idx = marginal_score(L["name"])
        if nxt_idx > state[L["name"]]:
            heapq.heappush(heap, (neg_score, L["name"], nxt_idx))

    # Record starting state
    total_numel = total_elements()
    pareto_curve = []
    pareto_curve.append({
        "step": 0,
        "cost_bytes": total_cost(),
        "weighted_error": total_error(),
        "avg_bpw": total_cost() * 8 / total_numel,
        "recipe": {n: get_config(n)[2] for n in state},
    })

    step = 0
    while heap:
        neg_score, name, target_idx = heapq.heappop(heap)

        # Skip stale entries
        if state[name] >= target_idx:
            continue

        # Upgrade this layer
        state[name] = target_idx
        step += 1

        # Record new state
        pareto_curve.append({
            "step": step,
            "cost_bytes": total_cost(),
            "weighted_error": total_error(),
            "avg_bpw": total_cost() * 8 / total_numel,
            "recipe": {n: get_config(n)[2] for n in state},
        })

        # Push next upgrade for this layer
        neg_score, nxt_idx = marginal_score(name)
        if nxt_idx > state[name]:
            heapq.heappush(heap, (neg_score, name, nxt_idx))

    return pareto_curve


def find_pareto_knee(pareto_curve: List[Dict]) -> Tuple[Dict, Dict]:
    """Find knee points: best value AND diminishing returns.

    Returns two points:
    1. "value_knee" - best bang-for-buck (max distance below diagonal)
    2. "elbow_knee" - where diminishing returns kick in (max curvature)

    The elbow is typically what users want for production - the point
    where adding more bits gives rapidly diminishing error reduction.
    """
    if len(pareto_curve) < 5:
        mid = pareto_curve[len(pareto_curve) // 2]
        return mid, mid

    # Extract (bpw, log_error) pairs
    points = [(p['avg_bpw'], math.log(max(p['weighted_error'], 1e-20))) for p in pareto_curve]

    # Normalize to [0, 1]
    bpw_min = min(p[0] for p in points)
    bpw_max = max(p[0] for p in points)
    err_min = min(p[1] for p in points)
    err_max = max(p[1] for p in points)

    if bpw_max == bpw_min or err_max == err_min:
        mid = pareto_curve[len(pareto_curve) // 2]
        return mid, mid

    normalized = [
        ((p[0] - bpw_min) / (bpw_max - bpw_min),
         (p[1] - err_min) / (err_max - err_min))
        for p in points
    ]

    # 1. Best value knee: max distance below diagonal y = 1 - x
    max_dist = -1
    value_idx = 0
    for i, (x, y) in enumerate(normalized):
        dist = (1 - x - y) / math.sqrt(2)
        if dist > max_dist:
            max_dist = dist
            value_idx = i

    # 2. Elbow knee: max curvature (where slope changes most rapidly)
    # Use second derivative: d²(log_error)/d(bpw)²
    # Sample at intervals to reduce noise
    step = max(1, len(pareto_curve) // 100)
    sampled_indices = list(range(0, len(pareto_curve), step))
    if sampled_indices[-1] != len(pareto_curve) - 1:
        sampled_indices.append(len(pareto_curve) - 1)

    max_curvature = -1
    elbow_idx = len(pareto_curve) // 2

    for i in range(1, len(sampled_indices) - 1):
        idx_prev = sampled_indices[i - 1]
        idx_curr = sampled_indices[i]
        idx_next = sampled_indices[i + 1]

        x0, y0 = normalized[idx_prev]
        x1, y1 = normalized[idx_curr]
        x2, y2 = normalized[idx_next]

        # First derivatives (slopes)
        dx1 = x1 - x0
        dx2 = x2 - x1
        if dx1 < 1e-10 or dx2 < 1e-10:
            continue

        dy1 = (y1 - y0) / dx1  # slope before
        dy2 = (y2 - y1) / dx2  # slope after

        # Second derivative (curvature proxy)
        # Negative curvature means concave down (error decreasing less rapidly)
        d2y = (dy2 - dy1) / ((dx1 + dx2) / 2)

        # We want positive curvature (slope going from negative to less negative)
        # This is where error reduction slows down
        if d2y > max_curvature:
            max_curvature = d2y
            elbow_idx = idx_curr

    return pareto_curve[value_idx], pareto_curve[elbow_idx]


def main():
    parser = argparse.ArgumentParser(description="3D water-fill allocator with analytical MSE")
    parser.add_argument("--sensitivity", required=True, help="HAWQ sensitivity JSON")
    parser.add_argument("--output", required=True, help="Output Pareto curve JSON")
    args = parser.parse_args()

    print("=" * 70)
    print("Joint 3D Water-Fill Allocator (Analytical)")
    print("=" * 70)

    # Load HAWQ sensitivity
    print(f"\nLoading HAWQ sensitivity from {args.sensitivity}...")
    layers = load_hawq_sensitivity(args.sensitivity)
    print(f"Loaded {len(layers)} layers")

    total_numel = sum(L["numel"] for L in layers)
    print(f"Total elements: {total_numel:,}")

    # Water-fill
    print("\nWater-filling over 3D config space...")
    pareto_curve = water_fill_3d(layers)
    print(f"Generated {len(pareto_curve)} Pareto points")

    # Show sampled results
    print("\n" + "=" * 70)
    print("PARETO FRONTIER (sampled)")
    print("=" * 70)
    print(f"{'Step':>6} {'BPW':>8} {'Memory':>14} {'Error':>12}")
    print("-" * 44)

    n_points = len(pareto_curve)
    sample_indices = [0] + [int(i * (n_points-1) / 9) for i in range(1, 9)] + [n_points - 1]
    sample_indices = sorted(set(sample_indices))

    for idx in sample_indices:
        p = pareto_curve[idx]
        print(f"{p['step']:>6} {p['avg_bpw']:>8.2f} {p['cost_bytes']:>14,} {p['weighted_error']:>12.2e}")

    # Config distribution at key points
    print("\n" + "=" * 70)
    print("CONFIG DISTRIBUTION AT KEY POINTS")
    print("=" * 70)

    for bpw_target in [3.0, 4.0, 5.0, 6.0, 8.0]:
        closest = min(pareto_curve, key=lambda p: abs(p['avg_bpw'] - bpw_target))
        if abs(closest['avg_bpw'] - bpw_target) > 0.5:
            continue

        print(f"\nAt ~{closest['avg_bpw']:.1f} bpw (step {closest['step']}):")

        # Count configs
        config_counts = {}
        for cfg in closest['recipe'].values():
            key = f"w{cfg[0]}_s{cfg[1]}_g{cfg[2]}"
            config_counts[key] = config_counts.get(key, 0) + 1

        for cfg, count in sorted(config_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {cfg}: {count} layers")

    # Find optimal knee points
    value_knee, elbow_knee = find_pareto_knee(pareto_curve)

    print("\n" + "=" * 70)
    print("KNEE POINTS")
    print("=" * 70)

    print("\n1. BEST VALUE (maximum efficiency - aggressive compression):")
    print(f"   BPW: {value_knee['avg_bpw']:.2f}")
    print(f"   Memory: {value_knee['cost_bytes']:,} bytes ({value_knee['cost_bytes'] / 1e9:.2f} GB)")
    print(f"   Weighted Error: {value_knee['weighted_error']:.2e}")

    print("\n2. ELBOW (diminishing returns - balanced tradeoff):")
    print(f"   BPW: {elbow_knee['avg_bpw']:.2f}")
    print(f"   Memory: {elbow_knee['cost_bytes']:,} bytes ({elbow_knee['cost_bytes'] / 1e9:.2f} GB)")
    print(f"   Weighted Error: {elbow_knee['weighted_error']:.2e}")

    # Show config distribution at elbow (recommended)
    config_counts = {}
    for cfg in elbow_knee['recipe'].values():
        key = f"w{cfg[0]}_s{cfg[1]}_g{cfg[2]}"
        config_counts[key] = config_counts.get(key, 0) + 1

    print("\nConfig distribution at elbow (recommended):")
    for cfg, count in sorted(config_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cfg}: {count} layers")

    # Save output
    with open(args.output, "w") as f:
        json.dump({
            "n_layers": len(layers),
            "total_elements": total_numel,
            "pareto_curve": pareto_curve,
            "min_bpw": pareto_curve[0]['avg_bpw'],
            "max_bpw": pareto_curve[-1]['avg_bpw'],
            "value_knee": {
                "bpw": value_knee['avg_bpw'],
                "cost_bytes": value_knee['cost_bytes'],
                "weighted_error": value_knee['weighted_error'],
                "step": value_knee['step'],
                "recipe": value_knee['recipe'],
            },
            "elbow_knee": {
                "bpw": elbow_knee['avg_bpw'],
                "cost_bytes": elbow_knee['cost_bytes'],
                "weighted_error": elbow_knee['weighted_error'],
                "step": elbow_knee['step'],
                "recipe": elbow_knee['recipe'],
            },
        }, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
