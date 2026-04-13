#!/usr/bin/env python3
"""
joint_knapsack_optimizer.py — Optimal (w_bits, s_bits, g_size) allocation via DP

Full HAWQ-style optimization extended to 3D:
  1. Measure sensitivity per layer (Hessian trace proxy)
  2. Build Pareto frontier per layer over (w_bits, s_bits, g_size)
  3. Solve sensitivity-weighted multi-choice knapsack

Objective: minimize Σ(sensitivity_i × MSE_i)
Constraint: Σ(memory_i) ≤ budget

Uses ProcessPoolExecutor for true parallelism (bypasses GIL).
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from safetensors import safe_open
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import hashlib
import time
import os
import json
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

N_WORKERS = os.cpu_count() or 20

# Cache directory for Pareto frontiers
CACHE_DIR = Path.home() / ".cache" / "dynaquant" / "pareto"


def load_hawq_sensitivity(path: str) -> Dict[str, float]:
    """Load HAWQ sensitivity from measure_hawq_sensitivity.py output.

    Returns dict mapping layer name -> sensitivity scalar.
    Uses h_trace * w_norm_sq / numel (highest correlation in validation).
    """
    with open(path) as f:
        data = json.load(f)

    sensitivities = {}
    if "sensitivity" in data:
        # HAWQ format
        for name, entry in data["sensitivity"].items():
            h_trace = entry.get("h_trace", 0.0)
            w_norm_sq = entry.get("w_norm_sq", 0.0)
            numel = max(1, entry.get("numel", 1))
            sensitivities[name] = h_trace * w_norm_sq / numel
    else:
        raise ValueError(f"Unknown sensitivity format in {path}")

    return sensitivities


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


@dataclass
class LayerInfo:
    name: str
    shape: Tuple[int, ...]
    n_elements: int
    sensitivity: float
    pareto_configs: List[ConfigResult]
    stats: Dict[str, float]


# Default search space: wide weight search, narrow scale search, moderate groups.
DEFAULT_W_BITS_RANGE = list(range(3, 17))
DEFAULT_S_BITS_RANGE = [8, 16]
DEFAULT_G_SIZE_RANGE = [16, 32, 64, 128, 256]
DEFAULT_BASELINE_CONFIG = (4, 8, 16)


def get_shape_hash(shape: Tuple[int, ...], weights_sample: Optional[np.ndarray] = None) -> str:
    """Compute a cache key from shape plus sampled content.

    Shape alone is not enough: same-shape layers can have different outlier and
    scale distributions, which changes their Pareto frontiers.
    """
    h = hashlib.md5(str(shape).encode())
    if weights_sample is not None:
        flat = np.asarray(weights_sample, dtype=np.float32).reshape(-1)
        if flat.size:
            sample_n = min(1024, flat.size)
            if flat.size > sample_n:
                idx = np.linspace(0, flat.size - 1, sample_n, dtype=np.int64)
                flat = flat[idx]
            h.update(flat.tobytes())
    return h.hexdigest()[:12]


def load_cached_frontier(shape: Tuple[int, ...], weights_sample: Optional[np.ndarray] = None) -> Optional[List[Tuple]]:
    """Load cached Pareto frontier for a shape if available."""
    if not CACHE_DIR.exists():
        return None
    cache_file = CACHE_DIR / f"{get_shape_hash(shape, weights_sample)}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_cached_frontier(shape: Tuple[int, ...], pareto: List[Tuple],
                         weights_sample: Optional[np.ndarray] = None):
    """Save Pareto frontier to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{get_shape_hash(shape, weights_sample)}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump(pareto, f)
    except Exception:
        pass  # Cache failures are non-fatal


def compute_memory(n_elements: int, w_bits: int, s_bits: int, g_size: int) -> int:
    """Compute memory in bytes for quantized tensor.

    For int4 scales (s_bits=4), adds 4 bytes for scale-of-scale per tensor.
    """
    weight_bits = n_elements * w_bits
    weight_bytes = (weight_bits + 7) // 8
    n_groups = (n_elements + g_size - 1) // g_size

    if s_bits == 4:
        # int4 scales: 4 bits per scale + 4 bytes scale-of-scale
        scale_bytes = (n_groups * 4 + 7) // 8 + 4
    else:
        scale_bytes = n_groups * (s_bits // 8)

    return weight_bytes + scale_bytes


def compute_bits_per_weight(w_bits: int, s_bits: int, g_size: int) -> float:
    return w_bits + s_bits / g_size


def scale_bits_per_weight(s_bits: int, g_size: int) -> float:
    return s_bits / g_size


def quantize_scale_fp8_e4m3(scales: np.ndarray) -> np.ndarray:
    """Quantize scales to fp8 e4m3fn format (4-bit exp, 3-bit mantissa)."""
    result = np.zeros_like(scales)
    nz = scales > 0

    if not nz.any():
        return result

    # fp8 e4m3: bias=7, 4 exponent bits, 3 mantissa bits
    # Range: ~1e-9 to 448
    vals = scales[nz]

    # Clamp to fp8 range
    vals = np.clip(vals, 2**-9, 448)

    # Extract exponent and mantissa
    exp = np.floor(np.log2(vals))
    mantissa = vals / (2.0 ** exp)  # In range [1, 2)

    # Quantize mantissa to 3 bits (8 levels in [1, 2))
    mantissa_q = np.round((mantissa - 1.0) * 8) / 8 + 1.0

    result[nz] = mantissa_q * (2.0 ** exp)
    return result


def quantize_scale_int4_symmetric(scales: np.ndarray) -> np.ndarray:
    """Quantize scales using symmetric int4 with per-tensor scale-of-scale."""
    if scales.max() < 1e-10:
        return scales

    # Scale-of-scale: single float per tensor
    sos = scales.max() / 7.0  # int4 symmetric: -8 to 7, use positive range

    # Quantize to int4
    q_scales = np.round(scales / sos).clip(0, 7)  # Scales are positive

    # Dequantize
    return q_scales * sos


def quantize_and_measure_single(weights_flat: np.ndarray, w_bits: int, s_bits: int, g_size: int) -> float:
    """Quantize weights with given config and return MSE. Optimized numpy version."""
    original_n = len(weights_flat)
    n = original_n

    # Pad to multiple of g_size
    if n % g_size != 0:
        pad = g_size - (n % g_size)
        weights_flat = np.concatenate([weights_flat, np.zeros(pad, dtype=np.float32)])
        n = len(weights_flat)

    n_groups = n // g_size
    groups = weights_flat.reshape(n_groups, g_size)

    qmax = (1 << (w_bits - 1)) - 1

    # Vectorized per-group processing
    max_abs = np.abs(groups).max(axis=1)  # [n_groups]

    # Compute raw scales
    valid = max_abs > 1e-10
    raw_scales = np.zeros(n_groups, dtype=np.float32)
    raw_scales[valid] = max_abs[valid] / qmax

    # Quantize scales based on s_bits
    if s_bits >= 16:
        # bf16 - use as-is (bf16 has enough precision for scales)
        scales = raw_scales
    elif s_bits == 8:
        # fp8 e4m3fn
        scales = quantize_scale_fp8_e4m3(raw_scales)
    else:
        # int4 symmetric with per-tensor scale-of-scale
        scales = quantize_scale_int4_symmetric(raw_scales)

    # Quantize weights and reconstruct
    scales_expanded = scales[:, np.newaxis]  # [n_groups, 1]
    scales_safe = np.where(scales_expanded > 1e-10, scales_expanded, 1.0)

    codes = np.clip(np.round(groups / scales_safe), -qmax - 1, qmax)
    recon = codes * scales_expanded

    # Zero out groups with zero scale
    recon[scales < 1e-10] = 0

    # MSE over original length (exclude synthetic padding).
    recon_flat = recon.reshape(-1)[:original_n]
    se = ((weights_flat[:original_n] - recon_flat) ** 2).sum()
    return float(se / original_n)


def parse_int_list(spec: str) -> List[int]:
    """Parse comma-separated ints and closed ranges like 3-16."""
    values = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            for v in range(start, end + step, step):
                values.add(v)
        else:
            values.add(int(part))
    return sorted(values)


def parse_config(spec: str) -> Tuple[int, int, int]:
    parts = [int(p.strip()) for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected w_bits,s_bits,g_size but got: {spec}")
    return tuple(parts)  # type: ignore[return-value]


def build_configs(w_bits: List[int], s_bits: List[int], g_sizes: List[int]) -> List[Tuple[int, int, int]]:
    return [(w, s, g) for w in w_bits for s in s_bits for g in g_sizes]


def compute_layer_stats(weights_flat: np.ndarray) -> Dict[str, float]:
    w = np.asarray(weights_flat, dtype=np.float32).reshape(-1)
    if w.size == 0:
        return {"std": 0.0, "kurtosis": 0.0, "outlier_ratio": 0.0, "max_abs": 0.0}
    mean = float(np.mean(w))
    centered = w - mean
    var = float(np.mean(centered ** 2))
    std = float(np.sqrt(max(var, 1e-20)))
    kurtosis = float(np.mean(centered ** 4) / max(var ** 2, 1e-20))
    outlier_ratio = float(np.mean(np.abs(centered) > (3.0 * std)))
    max_abs = float(np.max(np.abs(w)))
    return {
        "std": std,
        "kurtosis": kurtosis,
        "outlier_ratio": outlier_ratio,
        "max_abs": max_abs,
    }


def evaluate_layer_worker(args):
    """Worker function for ProcessPoolExecutor - evaluates one layer.

    Uses shape-based caching: layers with identical shapes share Pareto frontiers.
    Sensitivity is from HAWQ if provided, else falls back to RMS proxy.
    """
    name, weights_flat, shape, n_elements, use_cache, hawq_sensitivity, all_configs = args

    # Use HAWQ sensitivity if provided, else fall back to RMS proxy
    if hawq_sensitivity is not None:
        sensitivity = hawq_sensitivity
    else:
        # Fallback: RMS of weights (weak proxy, warns user)
        sensitivity = float(np.sqrt(np.mean(weights_flat ** 2)))

    # Try to load cached frontier
    if use_cache:
        cached = load_cached_frontier(shape, weights_flat)
        if cached is not None:
            return (name, shape, n_elements, sensitivity, compute_layer_stats(weights_flat), cached, True)

    # Evaluate all configs
    results = []
    for w_bits, s_bits, g_size in all_configs:
        mse = quantize_and_measure_single(weights_flat, w_bits, s_bits, g_size)
        memory = compute_memory(n_elements, w_bits, s_bits, g_size)
        bpw = compute_bits_per_weight(w_bits, s_bits, g_size)
        results.append((w_bits, s_bits, g_size, mse, memory, bpw))

    # Build Pareto frontier
    pareto_indices = []
    for i, (w1, s1, g1, mse1, mem1, bpw1) in enumerate(results):
        dominated = False
        for j, (w2, s2, g2, mse2, mem2, bpw2) in enumerate(results):
            if i == j:
                continue
            if mse2 < mse1 and mem2 < mem1:
                dominated = True
                break
            if mse2 <= mse1 and mem2 < mem1:
                dominated = True
                break
            if mse2 < mse1 and mem2 <= mem1:
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)

    pareto = [results[i] for i in pareto_indices]
    pareto.sort(key=lambda x: x[4])  # Sort by memory

    # Cache the frontier
    if use_cache:
        save_cached_frontier(shape, pareto, weights_flat)

    return (name, shape, n_elements, sensitivity, compute_layer_stats(weights_flat), pareto, False)


def water_fill_pareto(layers: List[LayerInfo]) -> List[Dict]:
    """
    Water-filling over 3D Pareto frontiers to compute global Pareto curve.

    Each layer has a Pareto frontier of (w_bits, s_bits, g_size) configs.
    We start all layers at their minimum-memory config, then iteratively
    upgrade the layer with best marginal (sensitivity × MSE_reduction) / memory_increase.

    Returns list of (cost_bytes, weighted_error, recipe) at each step.
    """
    import heapq

    # Initialize: each layer at its cheapest config (index 0, sorted by memory)
    state = {layer.name: 0 for layer in layers}  # config index per layer
    lookup = {layer.name: layer for layer in layers}

    def total_cost() -> int:
        return sum(lookup[n].pareto_configs[state[n]].memory_bytes for n in state)

    def total_error() -> float:
        return sum(lookup[n].sensitivity * lookup[n].pareto_configs[state[n]].mse
                   for n in state)

    def marginal_score(name: str) -> Tuple[float, int]:
        """Score for upgrading this layer to next Pareto config.
        Returns (negative_score, next_idx) for max-heap behavior."""
        layer = lookup[name]
        cur_idx = state[name]

        if cur_idx >= len(layer.pareto_configs) - 1:
            return (0.0, cur_idx)  # Already at max

        cur_cfg = layer.pareto_configs[cur_idx]
        nxt_cfg = layer.pareto_configs[cur_idx + 1]

        d_error = layer.sensitivity * (cur_cfg.mse - nxt_cfg.mse)  # Error reduction (positive)
        d_cost = nxt_cfg.memory_bytes - cur_cfg.memory_bytes  # Cost increase (positive)

        if d_cost <= 0:
            return (-float('inf'), cur_idx + 1)  # Free upgrade, take it

        score = d_error / d_cost  # Marginal utility per byte
        return (-score, cur_idx + 1)  # Negative for max-heap

    # Build initial heap
    heap = []
    for layer in layers:
        neg_score, nxt_idx = marginal_score(layer.name)
        if nxt_idx > state[layer.name]:
            heapq.heappush(heap, (neg_score, layer.name, nxt_idx))

    # Record starting state (all at minimum)
    pareto_curve = []
    pareto_curve.append({
        "step": 0,
        "cost_bytes": total_cost(),
        "weighted_error": total_error(),
        "avg_bpw": total_cost() * 8 / sum(l.n_elements for l in layers),
        "recipe": {n: str(lookup[n].pareto_configs[state[n]].config) for n in state},
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
            "avg_bpw": total_cost() * 8 / sum(l.n_elements for l in layers),
            "recipe": {n: str(lookup[n].pareto_configs[state[n]].config) for n in state},
        })

        # Push next upgrade for this layer if available
        neg_score, nxt_idx = marginal_score(name)
        if nxt_idx > state[name]:
            heapq.heappush(heap, (neg_score, name, nxt_idx))

    return pareto_curve


def find_config_index(layer: LayerInfo, target: Tuple[int, int, int]) -> int:
    """Find exact target config or the closest not-smaller fallback."""
    exact = None
    fallback = None
    fallback_key = None
    for idx, cfg in enumerate(layer.pareto_configs):
        c = cfg.config
        if (c.w_bits, c.s_bits, c.g_size) == target:
            exact = idx
            break
        if c.w_bits >= target[0] and c.s_bits >= target[1] and c.g_size <= target[2]:
            key = (c.w_bits - target[0], c.s_bits - target[1], target[2] - c.g_size, cfg.memory_bytes)
            if fallback_key is None or key < fallback_key:
                fallback = idx
                fallback_key = key
    if exact is not None:
        return exact
    if fallback is not None:
        return fallback
    # Last resort: closest by Manhattan distance in config space.
    return min(
        range(len(layer.pareto_configs)),
        key=lambda i: (
            abs(layer.pareto_configs[i].config.w_bits - target[0]) +
            abs(layer.pareto_configs[i].config.s_bits - target[1]) +
            abs(layer.pareto_configs[i].config.g_size - target[2])
        ),
    )


def build_promotion_curve(layers: List[LayerInfo], baseline_config: Tuple[int, int, int]) -> List[Dict]:
    """Promotion ladder anchored at a shared NVFP4-like baseline."""
    import heapq

    state = {layer.name: find_config_index(layer, baseline_config) for layer in layers}
    lookup = {layer.name: layer for layer in layers}
    total_elems = sum(layer.n_elements for layer in layers)

    def total_cost() -> int:
        return sum(lookup[n].pareto_configs[state[n]].memory_bytes for n in state)

    def total_error() -> float:
        return sum(
            lookup[n].sensitivity * lookup[n].pareto_configs[state[n]].mse
            for n in state
        )

    def marginal_score(name: str) -> Tuple[float, int]:
        layer = lookup[name]
        cur_idx = state[name]
        if cur_idx >= len(layer.pareto_configs) - 1:
            return (0.0, cur_idx)
        cur_cfg = layer.pareto_configs[cur_idx]
        best = None
        for nxt_idx in range(cur_idx + 1, len(layer.pareto_configs)):
            nxt_cfg = layer.pareto_configs[nxt_idx]
            d_error = layer.sensitivity * (cur_cfg.mse - nxt_cfg.mse)
            d_cost = nxt_cfg.memory_bytes - cur_cfg.memory_bytes
            if d_cost <= 0:
                score = float("inf")
            else:
                score = d_error / d_cost
            if best is None or score > best[0]:
                best = (score, nxt_idx)
        if best is None:
            return (0.0, cur_idx)
        score, idx = best
        return (-score, idx)

    heap = []
    for layer in layers:
        neg_score, nxt_idx = marginal_score(layer.name)
        if nxt_idx > state[layer.name]:
            heapq.heappush(heap, (neg_score, layer.name, nxt_idx))

    curve = [{
        "step": 0,
        "cost_bytes": total_cost(),
        "weighted_error": total_error(),
        "avg_bpw": total_cost() * 8 / total_elems,
        "recipe": {n: str(lookup[n].pareto_configs[state[n]].config) for n in state},
    }]

    step = 0
    while heap:
        neg_score, name, target_idx = heapq.heappop(heap)
        if state[name] >= target_idx:
            continue
        state[name] = target_idx
        step += 1
        curve.append({
            "step": step,
            "cost_bytes": total_cost(),
            "weighted_error": total_error(),
            "avg_bpw": total_cost() * 8 / total_elems,
            "recipe": {n: str(lookup[n].pareto_configs[state[n]].config) for n in state},
        })
        neg_score, nxt_idx = marginal_score(name)
        if nxt_idx > state[name]:
            heapq.heappush(heap, (neg_score, name, nxt_idx))

    return curve


def extract_outliers(
    layers: List[LayerInfo],
    baseline_config: Tuple[int, int, int],
    rescue_min_w_bits: int = 8,
) -> List[Dict]:
    """Identify tensors whose compressed-baseline damage is disproportionate.

    Score combines end-to-end weighted recovery from escaping the baseline with
    distributional pathology indicators. This keeps the global allocator focused
    on the normal case and surfaces layers that deserve bespoke handling.
    """
    rows = []
    for layer in layers:
        base_idx = find_config_index(layer, baseline_config)
        base_cfg = layer.pareto_configs[base_idx]
        rescue_candidates = [
            cfg for cfg in layer.pareto_configs
            if cfg.config.w_bits >= rescue_min_w_bits
        ]
        if rescue_candidates:
            rescue_cfg = min(rescue_candidates, key=lambda cfg: cfg.mse)
        else:
            rescue_cfg = min(layer.pareto_configs, key=lambda cfg: cfg.mse)
        raw_recovery = max(0.0, base_cfg.mse - rescue_cfg.mse)
        weighted_recovery = layer.sensitivity * raw_recovery
        pathology = 1.0 + layer.stats["outlier_ratio"] * 8.0 + max(0.0, layer.stats["kurtosis"] - 3.0) * 0.05
        score = weighted_recovery * pathology
        rows.append({
            "name": layer.name,
            "baseline_config": str(base_cfg.config),
            "rescue_config": str(rescue_cfg.config),
            "baseline_bpw": base_cfg.bits_per_weight,
            "rescue_bpw": rescue_cfg.bits_per_weight,
            "baseline_scale_bpw": scale_bits_per_weight(base_cfg.config.s_bits, base_cfg.config.g_size),
            "rescue_scale_bpw": scale_bits_per_weight(rescue_cfg.config.s_bits, rescue_cfg.config.g_size),
            "baseline_mse": base_cfg.mse,
            "rescue_mse": rescue_cfg.mse,
            "raw_recovery": raw_recovery,
            "weighted_recovery": weighted_recovery,
            "sensitivity": layer.sensitivity,
            "outlier_ratio": layer.stats["outlier_ratio"],
            "kurtosis": layer.stats["kurtosis"],
            "score": score,
        })

    rows.sort(key=lambda row: row["score"], reverse=True)
    if not rows:
        return rows

    cumsum = []
    running = 0.0
    for row in rows:
        running += row["score"]
        cumsum.append(running)

    if len(cumsum) < 3:
        cutoff = len(rows)
    else:
        xs = list(range(len(cumsum)))
        ys = cumsum
        x_min, x_max = 0, len(cumsum) - 1
        y_min, y_max = min(ys), max(ys)
        xr = x_max - x_min or 1
        yr = y_max - y_min or 1.0
        norm = [((x - x_min) / xr, (y - y_min) / yr) for x, y in zip(xs, ys)]
        x1, y1 = norm[0]
        x2, y2 = norm[-1]
        denom = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5 or 1.0
        cutoff = 1
        best_d = -1.0
        for i, (x, y) in enumerate(norm):
            d = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
            if d > best_d:
                best_d = d
                cutoff = i + 1

    top_score = rows[0]["score"]
    for i, row in enumerate(rows):
        row["is_outlier"] = i < cutoff and row["score"] > 0 and row["score"] >= 0.1 * top_score
        row["rank"] = i + 1
    return rows


def load_model_layers(model_path: Path, max_layers: int = None) -> List[Tuple[str, np.ndarray, Tuple, int]]:
    """Load weight tensors as numpy arrays for multiprocessing.

    Returns layers sorted by size (largest first) for better load balancing.
    """
    layers = []
    st_files = sorted(model_path.glob("*.safetensors"))

    for st_file in st_files:
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if ".weight" not in key:
                    continue
                if "layernorm" in key.lower() or "norm" in key.lower():
                    continue

                tensor = f.get_tensor(key)

                # For MoE experts, extract individual experts
                if "experts" in key and len(tensor.shape) == 3:
                    for exp_idx in [0, tensor.shape[0] // 2, tensor.shape[0] - 1]:
                        if exp_idx < tensor.shape[0]:
                            t = tensor[exp_idx]
                            layers.append((
                                f"{key}.expert{exp_idx}",
                                t.flatten().float().numpy(),
                                tuple(t.shape),
                                t.numel()
                            ))
                else:
                    layers.append((
                        key,
                        tensor.flatten().float().numpy(),
                        tuple(tensor.shape),
                        tensor.numel()
                    ))

                if max_layers and len(layers) >= max_layers:
                    break
        if max_layers and len(layers) >= max_layers:
            break

    # Sort by size descending for better load balancing (big layers first)
    layers.sort(key=lambda x: -x[3])
    return layers


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Joint (w_bits, s_bits, g_size) knapsack optimizer")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--sensitivity", type=str, default=None,
                        help="Path to HAWQ sensitivity JSON from measure_hawq_sensitivity.py. "
                             "If not provided, falls back to RMS-of-weights proxy (less accurate).")
    parser.add_argument("--target-bpw", type=float, default=4.5, help="Target bits per weight")
    parser.add_argument("--max-layers", type=int, default=None, help="Max layers to process")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--no-cache", action="store_true", help="Disable Pareto frontier caching")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")
    parser.add_argument("--skip-large", type=int, default=None,
                        help="Skip layers larger than N million elements (e.g. --skip-large 100 to skip embed/lm_head)")
    parser.add_argument("--w-bits", type=str, default="3-16",
                        help="Weight bit search space, e.g. 3-16 or 3,4,5,8,16")
    parser.add_argument("--s-bits", type=str, default="8,16",
                        help="Scale bit search space, e.g. 8,16")
    parser.add_argument("--g-sizes", type=str, default="16,32,64,128,256",
                        help="Group sizes, e.g. 16,32,64,128,256")
    parser.add_argument("--baseline-config", type=str, default="4,8,16",
                        help="Reference config for promotion-ladder and outlier analysis")
    parser.add_argument("--rescue-min-w-bits", type=int, default=8,
                        help="Minimum weight bits considered an outlier rescue config")
    args = parser.parse_args()

    # Handle cache
    if args.clear_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache at {CACHE_DIR}")

    t0 = time.time()
    model_path = Path(args.model)

    print("=" * 70)
    print("Joint 3D Water-Fill Optimizer")
    print(f"Model: {model_path}")
    print(f"Workers: {N_WORKERS}")
    w_bits = parse_int_list(args.w_bits)
    s_bits = parse_int_list(args.s_bits)
    g_sizes = parse_int_list(args.g_sizes)
    baseline_config = parse_config(args.baseline_config)
    all_configs = build_configs(w_bits, s_bits, g_sizes)
    print(f"Configs per layer: {len(all_configs)}")
    print(f"w_bits={w_bits}")
    print(f"s_bits={s_bits}")
    print(f"g_sizes={g_sizes}")
    print(f"baseline={baseline_config}")
    print("=" * 70)
    print(flush=True)

    # Load HAWQ sensitivity if provided
    hawq_sens = None
    if args.sensitivity:
        print(f"Loading HAWQ sensitivity from {args.sensitivity}...")
        hawq_sens = load_hawq_sensitivity(args.sensitivity)
        print(f"  Loaded sensitivity for {len(hawq_sens)} layers")
    else:
        print("WARNING: No --sensitivity provided, using RMS-of-weights proxy (less accurate)")
        print("         Run measure_hawq_sensitivity.py first for proper Fisher-based sensitivity")

    # Load layers
    print("\nLoading model weights...")
    t_load = time.time()
    raw_layers = load_model_layers(model_path, args.max_layers)
    print(f"Loaded {len(raw_layers)} weight tensors in {time.time() - t_load:.1f}s")

    # Filter out large layers if requested (for fast iteration)
    if args.skip_large:
        threshold = args.skip_large * 1_000_000
        before = len(raw_layers)
        raw_layers = [(n, w, s, e) for n, w, s, e in raw_layers if e < threshold]
        print(f"Skipped {before - len(raw_layers)} layers > {args.skip_large}M elements")

    total_elements = sum(n for _, _, _, n in raw_layers)
    print(f"Total elements: {total_elements:,}")
    print(flush=True)

    # Build Pareto frontiers in parallel
    print(f"\nBuilding Pareto frontiers ({N_WORKERS} processes)...")
    t_pareto = time.time()

    layers = []
    completed = 0
    cache_hits = 0

    # Add use_cache flag and HAWQ sensitivity to each layer
    use_cache = not args.no_cache
    raw_layers_with_cache = []
    for name, weights, shape, n in raw_layers:
        # Look up HAWQ sensitivity for this layer
        layer_sens = None
        if hawq_sens:
            # Try exact match first, then with/without .weight suffix
            layer_sens = hawq_sens.get(name)
            if layer_sens is None and name.endswith(".weight"):
                layer_sens = hawq_sens.get(name[:-7])  # strip .weight
            if layer_sens is None and not name.endswith(".weight"):
                layer_sens = hawq_sens.get(name + ".weight")
        raw_layers_with_cache.append((name, weights, shape, n, use_cache, layer_sens, all_configs))

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(evaluate_layer_worker, layer_data): layer_data[0]
                   for layer_data in raw_layers_with_cache}

        for future in as_completed(futures):
            name, shape, n_elements, sensitivity, stats, pareto_tuples, was_cached = future.result()
            if was_cached:
                cache_hits += 1

            # Convert tuples back to ConfigResult
            pareto_configs = [
                ConfigResult(
                    config=Config(w, s, g),
                    mse=mse,
                    memory_bytes=mem,
                    bits_per_weight=bpw
                )
                for w, s, g, mse, mem, bpw in pareto_tuples
            ]

            layers.append(LayerInfo(
                name=name,
                shape=shape,
                n_elements=n_elements,
                sensitivity=sensitivity,
                pareto_configs=pareto_configs,
                stats=stats,
            ))

            completed += 1
            if completed % 10 == 0 or completed == len(raw_layers):
                print(f"  {completed}/{len(raw_layers)} layers processed", flush=True)

    print(f"Pareto frontiers built in {time.time() - t_pareto:.1f}s (cache hits: {cache_hits}/{len(raw_layers)})")
    print(flush=True)

    # Water-fill to get full Pareto curve
    print("\nWater-filling over 3D Pareto frontiers...")
    t_wf = time.time()
    pareto_curve = water_fill_pareto(layers)
    print(f"Water-fill time: {time.time() - t_wf:.2f}s")
    print(f"Pareto curve has {len(pareto_curve)} points")
    print(flush=True)

    print("\nBuilding promotion ladder above baseline...")
    t_promote = time.time()
    promotion_curve = build_promotion_curve(layers, baseline_config)
    print(f"Promotion ladder time: {time.time() - t_promote:.2f}s")
    print(f"Promotion ladder has {len(promotion_curve)} points")

    print("\nExtracting outlier layers...")
    outliers = extract_outliers(
        layers,
        baseline_config=baseline_config,
        rescue_min_w_bits=args.rescue_min_w_bits,
    )
    promoted_outliers = [row for row in outliers if row["is_outlier"]]
    print(f"Outlier set size: {len(promoted_outliers)}/{len(outliers)}")
    for row in promoted_outliers[:10]:
        print(
            f"  {row['rank']:>3d}. {row['name']}  "
            f"{row['baseline_config']} -> {row['rescue_config']}  "
            f"score={row['score']:.2e}  wr={row['weighted_recovery']:.2e}"
        )

    # Results
    print("\n" + "=" * 70)
    print("PARETO FRONTIER (sampled)")
    print("=" * 70)
    print(f"{'Step':>6} {'BPW':>8} {'Memory':>14} {'Error':>12}")
    print("-" * 44)

    # Sample ~10 points from the curve
    n_points = len(pareto_curve)
    sample_indices = [0] + [int(i * (n_points-1) / 9) for i in range(1, 9)] + [n_points - 1]
    sample_indices = sorted(set(sample_indices))

    for idx in sample_indices:
        p = pareto_curve[idx]
        print(f"{p['step']:>6} {p['avg_bpw']:>8.2f} {p['cost_bytes']:>14,} {p['weighted_error']:>12.2e}")

    print("\n" + "=" * 70)
    print("PROMOTION LADDER ABOVE BASELINE (sampled)")
    print("=" * 70)
    print(f"{'Step':>6} {'BPW':>8} {'Memory':>14} {'Error':>12}")
    print("-" * 44)
    n_points = len(promotion_curve)
    sample_indices = [0] + [int(i * (n_points - 1) / 9) for i in range(1, 9)] + [n_points - 1]
    sample_indices = sorted(set(sample_indices))
    for idx in sample_indices:
        p = promotion_curve[idx]
        print(f"{p['step']:>6} {p['avg_bpw']:>8.2f} {p['cost_bytes']:>14,} {p['weighted_error']:>12.2e}")

    # Show config distribution at a few key points
    print("\n" + "=" * 70)
    print("CONFIG DISTRIBUTION AT KEY POINTS")
    print("=" * 70)

    for bpw_target in [3.0, 4.0, 5.0, 6.0]:
        # Find closest point
        closest = min(pareto_curve, key=lambda p: abs(p['avg_bpw'] - bpw_target))
        if abs(closest['avg_bpw'] - bpw_target) > 0.5:
            continue

        print(f"\nAt ~{closest['avg_bpw']:.1f} bpw (step {closest['step']}):")

        # Count configs
        config_counts = {}
        for cfg_str in closest['recipe'].values():
            config_counts[cfg_str] = config_counts.get(cfg_str, 0) + 1

        for cfg, count in sorted(config_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {cfg}: {count} layers")

    # Save output
    if args.output:
        output_data = {
            "model": str(model_path),
            "total_elements": total_elements,
            "n_layers": len(layers),
            "search_space": {
                "w_bits": w_bits,
                "s_bits": s_bits,
                "g_sizes": g_sizes,
            },
            "baseline_config": baseline_config,
            "pareto_curve": pareto_curve,
            "promotion_curve": promotion_curve,
            "outliers": outliers,
            # Also save summary stats
            "min_bpw": pareto_curve[0]['avg_bpw'],
            "max_bpw": pareto_curve[-1]['avg_bpw'],
            "min_error": pareto_curve[-1]['weighted_error'],
            "max_error": pareto_curve[0]['weighted_error'],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved full Pareto curve to {args.output}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
