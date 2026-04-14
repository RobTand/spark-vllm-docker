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
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from safetensors import safe_open
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import multiprocessing as mp
import hashlib
import time
import os
import json
import sys
import re
import math

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


def load_minimax_gate_priors(model_path: Path) -> Dict[str, float]:
    """Derive a lightweight routing prior from MiniMax router weights.

    This is not exact routing frequency. It is a checkpoint-only proxy that
    scores each MoE layer by how concentrated its router appears to be:
      mean(top-k(router_row_norm + correction_bias))) / mean(all experts)

    We normalize the resulting multipliers to have mean 1.0 and apply them only
    to MoE expert-family items.
    """
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = json.load(f)
    top_k = int(cfg.get("num_experts_per_tok", 8))

    gate_weights: Dict[str, torch.Tensor] = {}
    gate_biases: Dict[str, torch.Tensor] = {}
    for st_file in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(".block_sparse_moe.gate.weight"):
                    gate_weights[key] = f.get_tensor(key).float()
                elif key.endswith(".block_sparse_moe.e_score_correction_bias"):
                    gate_biases[key] = f.get_tensor(key).float()

    priors: Dict[str, float] = {}
    layer_scores: List[float] = []
    for gate_name, gate_weight in gate_weights.items():
        bias_name = gate_name.replace(".gate.weight", ".e_score_correction_bias")
        bias = gate_biases.get(bias_name)
        row_score = gate_weight.norm(dim=1)
        if bias is not None:
            row_score = row_score + bias
        score = float((row_score.topk(min(top_k, row_score.numel())).values.mean() / row_score.mean()).item())
        layer_name = gate_name[: gate_name.rfind(".block_sparse_moe.gate.weight")]
        priors[f"{layer_name}.block_sparse_moe.experts.*.w1.weight"] = score
        priors[f"{layer_name}.block_sparse_moe.experts.*.w2.weight"] = score
        priors[f"{layer_name}.block_sparse_moe.experts.*.w3.weight"] = score
        layer_scores.append(score)

    if not layer_scores:
        return priors

    mean_score = float(sum(layer_scores) / len(layer_scores))
    if mean_score <= 0:
        return priors
    for key in list(priors.keys()):
        priors[key] = priors[key] / mean_score
    return priors


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
    members: List[str] = field(default_factory=list)


# Default search space: wide weight search, narrow scale search, moderate groups.
DEFAULT_W_BITS_RANGE = list(range(3, 17))
DEFAULT_S_BITS_RANGE = [8, 16]
DEFAULT_G_SIZE_RANGE = [16, 32, 64, 128, 256]
DEFAULT_BASELINE_CONFIG = (4, 8, 16)


def get_shape_hash(shape: Tuple[int, ...], weights_sample: Optional[np.ndarray] = None,
                   config_signature: Optional[str] = None) -> str:
    """Compute a cache key from shape plus sampled content.

    Shape alone is not enough: same-shape layers can have different outlier and
    scale distributions, which changes their Pareto frontiers.
    """
    h = hashlib.md5(str(shape).encode())
    if config_signature:
        h.update(config_signature.encode())
    if weights_sample is not None:
        flat = np.asarray(weights_sample, dtype=np.float32).reshape(-1)
        if flat.size:
            sample_n = min(1024, flat.size)
            if flat.size > sample_n:
                idx = np.linspace(0, flat.size - 1, sample_n, dtype=np.int64)
                flat = flat[idx]
            h.update(flat.tobytes())
    return h.hexdigest()[:12]


def load_cached_frontier(shape: Tuple[int, ...], weights_sample: Optional[np.ndarray] = None,
                         config_signature: Optional[str] = None) -> Optional[List[Tuple]]:
    """Load cached Pareto frontier for a shape if available."""
    if not CACHE_DIR.exists():
        return None
    cache_file = CACHE_DIR / f"{get_shape_hash(shape, weights_sample, config_signature)}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_cached_frontier(shape: Tuple[int, ...], pareto: List[Tuple],
                         weights_sample: Optional[np.ndarray] = None,
                         config_signature: Optional[str] = None):
    """Save Pareto frontier to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{get_shape_hash(shape, weights_sample, config_signature)}.json"
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
    n_groups_est = (len(weights_flat) + g_size - 1) // g_size
    if n_groups_est > 1_000_000:
        return quantize_and_measure_single_chunked(weights_flat, w_bits, s_bits, g_size)

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


def quantize_and_measure_single_chunked(weights_flat: np.ndarray, w_bits: int, s_bits: int,
                                        g_size: int, chunk_groups: int = 262_144) -> float:
    """Chunked version for very large tensors (for example MoE expert tensors)."""
    weights_flat = np.asarray(weights_flat, dtype=np.float32).reshape(-1)
    original_n = len(weights_flat)
    qmax = (1 << (w_bits - 1)) - 1
    total_se = 0.0
    processed = 0

    for start_group in range(0, (original_n + g_size - 1) // g_size, chunk_groups):
        start = start_group * g_size
        end = min(original_n, (start_group + chunk_groups) * g_size)
        chunk = weights_flat[start:end]
        n = len(chunk)
        if n % g_size != 0:
            pad = g_size - (n % g_size)
            chunk = np.concatenate([chunk, np.zeros(pad, dtype=np.float32)])
            n = len(chunk)

        groups = chunk.reshape(n // g_size, g_size)
        max_abs = np.abs(groups).max(axis=1)
        valid = max_abs > 1e-10
        raw_scales = np.zeros(groups.shape[0], dtype=np.float32)
        raw_scales[valid] = max_abs[valid] / qmax

        if s_bits >= 16:
            scales = raw_scales
        elif s_bits == 8:
            scales = quantize_scale_fp8_e4m3(raw_scales)
        else:
            scales = quantize_scale_int4_symmetric(raw_scales)

        scales_expanded = scales[:, np.newaxis]
        scales_safe = np.where(scales_expanded > 1e-10, scales_expanded, 1.0)
        codes = np.clip(np.round(groups / scales_safe), -qmax - 1, qmax)
        recon = codes * scales_expanded
        recon[scales < 1e-10] = 0

        valid_n = min(end - start, n)
        diff = chunk[:valid_n] - recon.reshape(-1)[:valid_n]
        total_se += float(np.dot(diff, diff))
        processed += valid_n

    return total_se / max(processed, 1)


def quantize_scale_fp8_e4m3_torch(scales: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(scales)
    nz = scales > 0
    if not torch.any(nz):
        return result
    vals = torch.clamp(scales[nz], min=2 ** -9, max=448.0)
    exp = torch.floor(torch.log2(vals))
    mantissa = vals / torch.pow(torch.tensor(2.0, device=vals.device, dtype=vals.dtype), exp)
    mantissa_q = torch.round((mantissa - 1.0) * 8.0) / 8.0 + 1.0
    result[nz] = mantissa_q * torch.pow(torch.tensor(2.0, device=vals.device, dtype=vals.dtype), exp)
    return result


def quantize_scale_int4_symmetric_torch(scales: torch.Tensor) -> torch.Tensor:
    if torch.max(scales) < 1e-10:
        return scales
    sos = torch.max(scales) / 7.0
    q_scales = torch.clamp(torch.round(scales / sos), 0, 7)
    return q_scales * sos


def quantize_and_measure_single_torch(weights_flat: torch.Tensor, w_bits: int, s_bits: int, g_size: int) -> float:
    """Torch/GPU version of quantize_and_measure_single."""
    n_groups_est = (weights_flat.numel() + g_size - 1) // g_size
    if n_groups_est > 1_000_000:
        return quantize_and_measure_single_torch_chunked(weights_flat, w_bits, s_bits, g_size)

    original_n = weights_flat.numel()
    w = weights_flat.float()
    if original_n % g_size != 0:
        pad = g_size - (original_n % g_size)
        w = torch.nn.functional.pad(w, (0, pad))
    n = w.numel()
    groups = w.view(n // g_size, g_size)
    qmax = (1 << (w_bits - 1)) - 1
    max_abs = groups.abs().amax(dim=1)
    raw_scales = torch.zeros_like(max_abs)
    valid = max_abs > 1e-10
    raw_scales[valid] = max_abs[valid] / qmax
    if s_bits >= 16:
        scales = raw_scales
    elif s_bits == 8:
        scales = quantize_scale_fp8_e4m3_torch(raw_scales)
    else:
        scales = quantize_scale_int4_symmetric_torch(raw_scales)
    scales_expanded = scales[:, None]
    scales_safe = torch.where(scales_expanded > 1e-10, scales_expanded, torch.ones_like(scales_expanded))
    codes = torch.clamp(torch.round(groups / scales_safe), -qmax - 1, qmax)
    recon = codes * scales_expanded
    recon[scales < 1e-10] = 0
    recon_flat = recon.reshape(-1)[:original_n]
    se = torch.sum((weights_flat[:original_n] - recon_flat) ** 2)
    return float(se.item() / original_n)


def quantize_and_measure_single_torch_chunked(
    weights_flat: torch.Tensor,
    w_bits: int,
    s_bits: int,
    g_size: int,
    chunk_groups: int = 262_144,
) -> float:
    """GPU chunked quantization error path for very large tensors.

    Keeps only one chunk on device at a time. This is slower than the fully
    vectorized path on small tensors, but it avoids VRAM spikes on huge fused
    MoE tensors while still using the GPU for the heavy inner loop.
    """
    original_n = weights_flat.numel()
    qmax = (1 << (w_bits - 1)) - 1
    total_se = 0.0
    processed = 0

    device = torch.device("cuda")
    source = weights_flat
    if source.is_cuda:
        source = source.detach().to(device="cpu", dtype=torch.float32)
    else:
        source = source.detach().to(dtype=torch.float32)

    with torch.no_grad():
        total_groups = (original_n + g_size - 1) // g_size
        for start_group in range(0, total_groups, chunk_groups):
            start = start_group * g_size
            end = min(original_n, (start_group + chunk_groups) * g_size)
            chunk_cpu = source[start:end]
            n = chunk_cpu.numel()
            if n % g_size != 0:
                pad = g_size - (n % g_size)
                chunk_cpu = torch.nn.functional.pad(chunk_cpu, (0, pad))
                n = chunk_cpu.numel()

            chunk = chunk_cpu.to(device=device, dtype=torch.float32, non_blocking=False)
            groups = chunk.view(n // g_size, g_size)
            max_abs = groups.abs().amax(dim=1)
            raw_scales = torch.zeros_like(max_abs)
            valid = max_abs > 1e-10
            raw_scales[valid] = max_abs[valid] / qmax

            if s_bits >= 16:
                scales = raw_scales
            elif s_bits == 8:
                scales = quantize_scale_fp8_e4m3_torch(raw_scales)
            else:
                scales = quantize_scale_int4_symmetric_torch(raw_scales)

            scales_expanded = scales[:, None]
            scales_safe = torch.where(
                scales_expanded > 1e-10, scales_expanded, torch.ones_like(scales_expanded)
            )
            codes = torch.clamp(torch.round(groups / scales_safe), -qmax - 1, qmax)
            recon = codes * scales_expanded
            recon[scales < 1e-10] = 0

            valid_n = end - start
            diff = chunk[:valid_n] - recon.reshape(-1)[:valid_n]
            total_se += float(torch.sum(diff * diff).item())
            processed += valid_n

            del chunk, groups, max_abs, raw_scales, scales, scales_expanded, scales_safe, codes, recon, diff

    torch.cuda.empty_cache()
    return total_se / max(processed, 1)


def quantize_and_measure_many_torch_chunked(
    weights_flat: torch.Tensor,
    configs: List[Tuple[int, int, int]],
    target_chunk_elems: Optional[int] = None,
) -> Dict[Tuple[int, int, int], float]:
    """Evaluate many configs over one large tensor with shared chunk passes.

    For huge MoE tensors, the dominant cost was repeatedly reloading the same
    tensor for every config. This function groups configs by ``g_size`` and
    reuses each chunk across all bit/scale variants for that group size.
    """
    original_n = weights_flat.numel()
    device = torch.device("cuda")
    source = weights_flat
    if source.is_cuda:
        source = source.detach().to(device="cpu", dtype=torch.float32)
    else:
        source = source.detach().to(dtype=torch.float32)

    if target_chunk_elems is None:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        # Chunk working set is several tensor-sized buffers at once. Using ~25%
        # of currently free memory keeps utilization up without flirting with OOM.
        target_chunk_elems = int(max(8_388_608, min(134_217_728, (free_bytes * 0.25) // 24)))

    configs_by_g: Dict[int, List[Tuple[int, int, int]]] = {}
    for cfg in configs:
        configs_by_g.setdefault(cfg[2], []).append(cfg)

    total_se = {cfg: 0.0 for cfg in configs}

    with torch.no_grad():
        for g_size, g_configs in configs_by_g.items():
            chunk_groups = max(1, target_chunk_elems // g_size)
            unique_w_bits = sorted({w for w, _, _ in g_configs})
            for start_group in range(0, (original_n + g_size - 1) // g_size, chunk_groups):
                start = start_group * g_size
                end = min(original_n, (start_group + chunk_groups) * g_size)
                chunk_cpu = source[start:end]
                valid_n = chunk_cpu.numel()
                if valid_n % g_size != 0:
                    pad = g_size - (valid_n % g_size)
                    chunk_cpu = torch.nn.functional.pad(chunk_cpu, (0, pad))
                chunk = chunk_cpu.to(device=device, dtype=torch.float32, non_blocking=False)
                groups = chunk.view(-1, g_size)
                max_abs = groups.abs().amax(dim=1)

                raw_scales_by_w = {}
                for w_bits in unique_w_bits:
                    qmax = (1 << (w_bits - 1)) - 1
                    raw_scales = torch.zeros_like(max_abs)
                    valid = max_abs > 1e-10
                    raw_scales[valid] = max_abs[valid] / qmax
                    raw_scales_by_w[w_bits] = raw_scales

                for w_bits, s_bits, _ in g_configs:
                    qmax = (1 << (w_bits - 1)) - 1
                    raw_scales = raw_scales_by_w[w_bits]
                    if s_bits >= 16:
                        scales = raw_scales
                    elif s_bits == 8:
                        scales = quantize_scale_fp8_e4m3_torch(raw_scales)
                    else:
                        scales = quantize_scale_int4_symmetric_torch(raw_scales)

                    scales_expanded = scales[:, None]
                    scales_safe = torch.where(
                        scales_expanded > 1e-10, scales_expanded, torch.ones_like(scales_expanded)
                    )
                    codes = torch.clamp(torch.round(groups / scales_safe), -qmax - 1, qmax)
                    recon = codes * scales_expanded
                    recon[scales < 1e-10] = 0
                    diff = chunk[:valid_n] - recon.reshape(-1)[:valid_n]
                    total_se[(w_bits, s_bits, g_size)] += float(torch.sum(diff * diff).item())

                    del scales, scales_expanded, scales_safe, codes, recon, diff

                del chunk, groups, max_abs, raw_scales_by_w

    torch.cuda.empty_cache()
    return {cfg: total_se[cfg] / max(original_n, 1) for cfg in configs}


def quantize_and_measure_many_torch_streaming(
    weights_tensor: torch.Tensor,
    configs: List[Tuple[int, int, int]],
    target_chunk_elems: Optional[int] = None,
    whole_tensor_gpu_threshold_bytes: Optional[int] = None,
) -> Dict[Tuple[int, int, int], float]:
    """GPU-first evaluator that streams from a CPU tensor source.

    Small tensors are moved to GPU once and evaluated there. Large tensors stay
    on CPU in their source dtype and are chunk-copied to GPU, reusing each chunk
    across all configs with the same group size.
    """
    flat_cpu = weights_tensor.reshape(-1).contiguous()
    source_bytes = flat_cpu.numel() * flat_cpu.element_size()
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    if whole_tensor_gpu_threshold_bytes is None:
        # If a tensor is comfortably below ~20% of currently free memory,
        # move it wholesale and use the simpler fast path.
        whole_tensor_gpu_threshold_bytes = max(
            512 * 1024 * 1024,
            int(free_bytes * 0.20),
        )

    if target_chunk_elems is None:
        # The streaming path was previously too conservative, leaving the GPU
        # underutilized. Budget about 35% of free memory to the active chunk,
        # assuming roughly 24 bytes/element of transient working set.
        target_chunk_elems = int(max(
            16_777_216,
            min(268_435_456, (free_bytes * 0.35) // 24),
        ))

    if source_bytes <= whole_tensor_gpu_threshold_bytes:
        flat_gpu = flat_cpu.to(device="cuda", dtype=torch.float32, non_blocking=False)
        out = {}
        for cfg in configs:
            out[cfg] = quantize_and_measure_single_torch(flat_gpu, *cfg)
        del flat_gpu
        torch.cuda.empty_cache()
        return out

    original_n = flat_cpu.numel()
    device = torch.device("cuda")
    configs_by_g: Dict[int, List[Tuple[int, int, int]]] = {}
    for cfg in configs:
        configs_by_g.setdefault(cfg[2], []).append(cfg)

    total_se = {cfg: 0.0 for cfg in configs}

    with torch.no_grad():
        for g_size, g_configs in configs_by_g.items():
            chunk_groups = max(1, target_chunk_elems // g_size)
            unique_w_bits = sorted({w for w, _, _ in g_configs})
            total_groups = (original_n + g_size - 1) // g_size
            for start_group in range(0, total_groups, chunk_groups):
                start = start_group * g_size
                end = min(original_n, (start_group + chunk_groups) * g_size)
                chunk_cpu = flat_cpu[start:end]
                valid_n = chunk_cpu.numel()
                if valid_n % g_size != 0:
                    pad = g_size - (valid_n % g_size)
                    chunk_cpu = torch.nn.functional.pad(chunk_cpu, (0, pad))

                if not chunk_cpu.is_pinned():
                    chunk_cpu = chunk_cpu.pin_memory()
                chunk = chunk_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
                groups = chunk.view(-1, g_size)
                max_abs = groups.abs().amax(dim=1)

                raw_scales_by_w = {}
                valid = max_abs > 1e-10
                for w_bits in unique_w_bits:
                    qmax = (1 << (w_bits - 1)) - 1
                    raw_scales = torch.zeros_like(max_abs)
                    raw_scales[valid] = max_abs[valid] / qmax
                    raw_scales_by_w[w_bits] = raw_scales

                for w_bits, s_bits, _ in g_configs:
                    qmax = (1 << (w_bits - 1)) - 1
                    raw_scales = raw_scales_by_w[w_bits]
                    if s_bits >= 16:
                        scales = raw_scales
                    elif s_bits == 8:
                        scales = quantize_scale_fp8_e4m3_torch(raw_scales)
                    else:
                        scales = quantize_scale_int4_symmetric_torch(raw_scales)

                    scales_expanded = scales[:, None]
                    scales_safe = torch.where(
                        scales_expanded > 1e-10, scales_expanded, torch.ones_like(scales_expanded)
                    )
                    codes = torch.clamp(torch.round(groups / scales_safe), -qmax - 1, qmax)
                    recon = codes * scales_expanded
                    recon[scales < 1e-10] = 0
                    diff = chunk[:valid_n] - recon.reshape(-1)[:valid_n]
                    total_se[(w_bits, s_bits, g_size)] += float(torch.sum(diff * diff).item())

                    del scales, scales_expanded, scales_safe, codes, recon, diff

                del chunk, groups, max_abs, raw_scales_by_w, chunk_cpu

    torch.cuda.empty_cache()
    return {cfg: total_se[cfg] / max(original_n, 1) for cfg in configs}


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


CONFIG_RE = re.compile(r"^w(?P<w>\d+)_s(?P<s>\d+)_g(?P<g>\d+)$")


def parse_recipe_config_str(spec: str) -> Tuple[int, int, int]:
    m = CONFIG_RE.match(spec)
    if not m:
        raise ValueError(f"invalid config string: {spec}")
    return int(m.group("w")), int(m.group("s")), int(m.group("g"))


def build_configs(w_bits: List[int], s_bits: List[int], g_sizes: List[int]) -> List[Tuple[int, int, int]]:
    return [(w, s, g) for w in w_bits for s in s_bits for g in g_sizes]


def config_is_allowed(w_bits: int, s_bits: int, g_size: int,
                      enforce_nvfp4_fp4: bool) -> bool:
    """Filter configs that violate deployment-side format constraints.

    Blackwell NVFP4 is tied to 16-element micro-block scaling. When the search
    includes 4-bit weights and we intend to materialize those as NVFP4, keep
    FP4 on g=16 only. Other bit widths remain storage-side choices.
    """
    if enforce_nvfp4_fp4 and w_bits == 4 and g_size != 16:
        return False
    return True


def compute_layer_stats(weights_flat) -> Dict[str, float]:
    if isinstance(weights_flat, torch.Tensor):
        w = weights_flat.detach().float().reshape(-1)
        if w.numel() == 0:
            return {"std": 0.0, "kurtosis": 0.0, "outlier_ratio": 0.0, "max_abs": 0.0}
        mean = float(torch.mean(w).item())
        centered = w - mean
        var = float(torch.mean(centered ** 2).item())
        std = float(np.sqrt(max(var, 1e-20)))
        kurtosis = float((torch.mean(centered ** 4).item()) / max(var ** 2, 1e-20))
        outlier_ratio = float(torch.mean((torch.abs(centered) > (3.0 * std)).float()).item())
        max_abs = float(torch.max(torch.abs(w)).item())
        return {
            "std": std,
            "kurtosis": kurtosis,
            "outlier_ratio": outlier_ratio,
            "max_abs": max_abs,
        }

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


def config_signature(all_configs: List[Tuple[int, int, int]]) -> str:
    return "|".join(f"{w}:{s}:{g}" for w, s, g in all_configs)


def candidate_sensitivity_names(name: str) -> List[str]:
    """Generate plausible aliases across checkpoint/module naming conventions."""
    variants = [name]
    prefixes = [
        "model.language_model.",
        "language_model.model.",
        "language_model.",
        "model.",
    ]
    changed = True
    while changed:
        changed = False
        for cur in list(variants):
            for prefix in prefixes:
                if cur.startswith(prefix):
                    alt = cur[len(prefix):]
                    if alt not in variants:
                        variants.append(alt)
                        changed = True
    for cur in list(variants):
        if cur.endswith(".weight"):
            alt = cur[:-7]
            if alt not in variants:
                variants.append(alt)
        else:
            alt = cur + ".weight"
            if alt not in variants:
                variants.append(alt)
    return variants


def load_layer_tensor_cpu(layer_ref: Dict) -> torch.Tensor:
    """Load one tensor as a flattened CPU float32 torch tensor."""
    with safe_open(layer_ref["st_file"], framework="pt", device="cpu") as f:
        tensor = f.get_tensor(layer_ref["tensor_key"])
    if layer_ref["expert_idx"] is not None:
        tensor = tensor[layer_ref["expert_idx"]]
    return tensor.flatten().float()


def weight_sample_bytes(weights_flat, sample_n: int = 1024) -> np.ndarray:
    """Small content sample for cache-key stability without huge transfers."""
    if isinstance(weights_flat, torch.Tensor):
        flat = weights_flat.detach().reshape(-1)
        if flat.numel() == 0:
            return np.empty((0,), dtype=np.float32)
        if flat.numel() > sample_n:
            idx = torch.linspace(0, flat.numel() - 1, sample_n, dtype=torch.long)
            flat = flat[idx]
        return flat.cpu().numpy().astype(np.float32, copy=False)
    flat = np.asarray(weights_flat, dtype=np.float32).reshape(-1)
    if flat.size > sample_n:
        idx = np.linspace(0, flat.size - 1, sample_n, dtype=np.int64)
        flat = flat[idx]
    return flat


def evaluate_layer_worker(args):
    """Worker function for ProcessPoolExecutor - evaluates one layer.

    Uses shape-based caching: layers with identical shapes share Pareto frontiers.
    Sensitivity is from HAWQ if provided, else falls back to RMS proxy.
    """
    name, weights_flat, shape, n_elements, members, use_cache, hawq_sensitivity, all_configs, eval_backend = args
    cfg_sig = config_signature(all_configs)

    # Use HAWQ sensitivity if provided, else fall back to RMS proxy
    if hawq_sensitivity is not None:
        sensitivity = hawq_sensitivity
    else:
        # Fallback: RMS of weights (weak proxy, warns user)
        sensitivity = float(np.sqrt(np.mean(weights_flat ** 2)))

    sample = weight_sample_bytes(weights_flat)

    # Try to load cached frontier
    if use_cache:
        cached = load_cached_frontier(shape, sample, cfg_sig)
        if cached is not None:
            return (name, shape, n_elements, members, sensitivity, compute_layer_stats(weights_flat), cached, True)

    # Evaluate all configs
    results = []
    weights_torch = None
    if eval_backend == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("eval_backend=gpu requested but CUDA is not available")
        min_g = min(g for _, _, g in all_configs)
        n_groups_est = (len(weights_flat) + min_g - 1) // min_g
        # Avoid staging giant MoE tensors fully on device. The chunked GPU path
        # will stream them from CPU.
        if n_groups_est <= 4_000_000:
            weights_torch = torch.from_numpy(weights_flat).to(device="cuda", dtype=torch.float32)
    batched_mses = None
    if eval_backend == "gpu" and weights_torch is None:
        batched_mses = quantize_and_measure_many_torch_chunked(torch.from_numpy(weights_flat), all_configs)
    for w_bits, s_bits, g_size in all_configs:
        if eval_backend == "gpu":
            if batched_mses is not None:
                mse = batched_mses[(w_bits, s_bits, g_size)]
            else:
                mse = quantize_and_measure_single_torch(weights_torch, w_bits, s_bits, g_size)
        else:
            mse = quantize_and_measure_single(weights_flat, w_bits, s_bits, g_size)
        memory = compute_memory(n_elements, w_bits, s_bits, g_size)
        bpw = compute_bits_per_weight(w_bits, s_bits, g_size)
        results.append((w_bits, s_bits, g_size, mse, memory, bpw))
    if weights_torch is not None:
        del weights_torch
        torch.cuda.empty_cache()

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
        save_cached_frontier(shape, pareto, sample, cfg_sig)

    return (name, shape, n_elements, members, sensitivity, compute_layer_stats(weights_flat), pareto, False)


def evaluate_layer_local(layer_ref: Dict, use_cache_flag: bool, layer_sens: Optional[float],
                         all_configs: List[Tuple[int, int, int]], eval_backend: str):
    """Single-process evaluator used by the streaming GPU path."""
    if eval_backend != "gpu":
        weights = load_layer_array(layer_ref)
        return evaluate_layer_worker((
            layer_ref["name"],
            weights,
            layer_ref["shape"],
            layer_ref["n_elements"],
            list(layer_ref.get("members", [])),
            use_cache_flag,
            layer_sens,
            all_configs,
            eval_backend,
        ))

    name = layer_ref["name"]
    shape = layer_ref["shape"]
    n_elements = layer_ref["n_elements"]
    members = list(layer_ref.get("members", []))
    cfg_sig = config_signature(all_configs)
    tensor = load_layer_tensor(layer_ref)
    sample = sample_tensor_values(tensor)

    if layer_sens is not None:
        sensitivity = layer_sens
    else:
        sensitivity = float(np.sqrt(np.mean(sample ** 2))) if sample.size else 0.0

    if use_cache_flag:
        cached = load_cached_frontier(shape, sample, cfg_sig)
        if cached is not None:
            return (name, shape, n_elements, members, sensitivity, compute_layer_stats_sampled(sample), cached, True)

    mses = quantize_and_measure_many_torch_streaming(tensor, all_configs)
    results = []
    for w_bits, s_bits, g_size in all_configs:
        memory = compute_memory(n_elements, w_bits, s_bits, g_size)
        bpw = compute_bits_per_weight(w_bits, s_bits, g_size)
        results.append((w_bits, s_bits, g_size, mses[(w_bits, s_bits, g_size)], memory, bpw))

    pareto_indices = []
    for i, (_w1, _s1, _g1, mse1, mem1, _bpw1) in enumerate(results):
        dominated = False
        for j, (_w2, _s2, _g2, mse2, mem2, _bpw2) in enumerate(results):
            if i == j:
                continue
            if (mse2 < mse1 and mem2 < mem1) or (mse2 <= mse1 and mem2 < mem1) or (mse2 < mse1 and mem2 <= mem1):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)

    pareto = [results[i] for i in pareto_indices]
    pareto.sort(key=lambda x: x[4])

    if use_cache_flag:
        save_cached_frontier(shape, pareto, sample, cfg_sig)

    return (name, shape, n_elements, members, sensitivity, compute_layer_stats_sampled(sample), pareto, False)


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


def expert_family_key(name: str) -> Optional[str]:
    if ".mlp.experts.gate_up_proj" in name or ".mlp.experts.w13_" in name:
        return "moe_gate_up"
    if ".mlp.experts.down_proj" in name or ".mlp.experts.w2_" in name:
        return "moe_down"
    m = re.search(r"\.block_sparse_moe\.experts\.\d+\.(w[123])\.weight$", name)
    if m:
        return f"moe_{m.group(1)}"
    return None


def recipe_cost_error(recipe: Dict[str, str], lookup: Dict[str, LayerInfo]) -> Tuple[int, float, float]:
    total_elems = sum(layer.n_elements for layer in lookup.values())
    total_cost = 0
    total_error = 0.0
    for name, cfg_str in recipe.items():
        layer = lookup[name]
        chosen = None
        for cfg in layer.pareto_configs:
            if str(cfg.config) == cfg_str:
                chosen = cfg
                break
        if chosen is None:
            raise KeyError(f"config {cfg_str} not found for layer {name}")
        total_cost += chosen.memory_bytes
        total_error += layer.sensitivity * chosen.mse
    avg_bpw = total_cost * 8 / total_elems
    return total_cost, total_error, avg_bpw


def aggregate_expert_families_for_optimization(
    layers: List[LayerInfo],
) -> Tuple[List[LayerInfo], Dict[str, List[str]]]:
    """Collapse MoE expert families into shared optimization items.

    This turns all expert gate_up tensors into one decision variable and all
    expert down tensors into another. The optimizer can then trade off expert
    precision against the rest of the model directly, instead of projecting a
    mixed-width solution onto a shared-config constraint afterward.
    """
    lookup = {layer.name: layer for layer in layers}
    family_members: Dict[str, List[str]] = {}
    passthrough: List[LayerInfo] = []

    for layer in layers:
        fam = expert_family_key(layer.name)
        if fam:
            family_members.setdefault(fam, []).append(layer.name)
        else:
            passthrough.append(layer)

    optimized_layers = list(passthrough)
    aggregate_map: Dict[str, List[str]] = {}

    for fam, names in family_members.items():
        if len(names) <= 1:
            optimized_layers.extend(lookup[name] for name in names)
            continue

        per_member_cfgs = []
        config_intersection = None
        for name in names:
            layer = lookup[name]
            cfg_map = {str(cfg.config): cfg for cfg in layer.pareto_configs}
            per_member_cfgs.append((name, layer, cfg_map))
            cfg_keys = set(cfg_map.keys())
            config_intersection = cfg_keys if config_intersection is None else (config_intersection & cfg_keys)

        if not config_intersection:
            optimized_layers.extend(lookup[name] for name in names)
            continue

        total_elements = sum(lookup[name].n_elements for name in names)
        mean_stats = {
            key: float(np.mean([lookup[name].stats.get(key, 0.0) for name in names]))
            for key in ("std", "kurtosis", "outlier_ratio", "max_abs")
        }

        aggregate_cfgs: List[ConfigResult] = []
        for cfg_str in sorted(config_intersection, key=parse_recipe_config_str):
            w_bits, s_bits, g_size = parse_recipe_config_str(cfg_str)
            total_memory = 0
            total_weighted_error = 0.0
            for _name, layer, cfg_map in per_member_cfgs:
                cfg = cfg_map[cfg_str]
                total_memory += cfg.memory_bytes
                total_weighted_error += layer.sensitivity * cfg.mse
            aggregate_cfgs.append(
                ConfigResult(
                    config=Config(w_bits, s_bits, g_size),
                    mse=total_weighted_error,
                    memory_bytes=total_memory,
                    bits_per_weight=total_memory * 8 / total_elements,
                )
            )

        optimized_layers.append(
            LayerInfo(
                name=fam,
                shape=(total_elements,),
                n_elements=total_elements,
                sensitivity=1.0,
                pareto_configs=aggregate_cfgs,
                stats=mean_stats,
                members=list(names),
            )
        )
        aggregate_map[fam] = list(names)

    optimized_layers.sort(key=lambda layer: -layer.n_elements)
    return optimized_layers, aggregate_map


def expand_recipe_from_aggregates(
    recipe: Dict[str, str],
    aggregate_map: Dict[str, List[str]],
) -> Dict[str, str]:
    """Expand aggregate expert-family recipe entries back to real tensor names."""
    expanded: Dict[str, str] = {}
    for name, cfg_str in recipe.items():
        members = aggregate_map.get(name)
        if members:
            for member in members:
                expanded[member] = cfg_str
        else:
            expanded[name] = cfg_str
    return expanded


def expand_curve_from_aggregates(
    curve: List[Dict],
    aggregate_map: Dict[str, List[str]],
) -> List[Dict]:
    expanded_curve = []
    for point in curve:
        new_point = dict(point)
        new_point["recipe"] = expand_recipe_from_aggregates(point["recipe"], aggregate_map)
        expanded_curve.append(new_point)
    return expanded_curve


def apply_expert_consensus_to_recipe(
    recipe: Dict[str, str],
    lookup: Dict[str, LayerInfo],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Force a shared config per MoE expert family near the current recipe size.

    Consensus is chosen per family (gate_up/down) by matching the current
    average memory target as closely as possible, then preferring lower total
    weighted error among ties.
    """
    families: Dict[str, List[str]] = {}
    for name in recipe:
        fam = expert_family_key(name)
        if fam:
            families.setdefault(fam, []).append(name)

    if not families:
        return dict(recipe), {}

    adjusted = dict(recipe)
    chosen_configs: Dict[str, str] = {}

    for fam, names in families.items():
        if len(names) <= 1:
            continue

        target_avg_memory = 0.0
        per_layer_cfg_maps = []
        config_intersection = None

        for name in names:
            layer = lookup[name]
            cfg_map = {str(cfg.config): cfg for cfg in layer.pareto_configs}
            per_layer_cfg_maps.append((name, layer, cfg_map))
            config_keys = set(cfg_map.keys())
            config_intersection = config_keys if config_intersection is None else (config_intersection & config_keys)
            current_cfg = cfg_map[recipe[name]]
            target_avg_memory += current_cfg.memory_bytes

        if not config_intersection:
            continue

        target_avg_memory /= len(names)
        best = None
        for cfg_str in sorted(config_intersection):
            avg_memory = 0.0
            total_weighted_error = 0.0
            for _name, layer, cfg_map in per_layer_cfg_maps:
                cfg = cfg_map[cfg_str]
                avg_memory += cfg.memory_bytes
                total_weighted_error += layer.sensitivity * cfg.mse
            avg_memory /= len(names)
            key = (
                abs(avg_memory - target_avg_memory),
                total_weighted_error,
                avg_memory,
                parse_recipe_config_str(cfg_str),
            )
            if best is None or key < best[0]:
                best = (key, cfg_str)

        if best is None:
            continue

        cfg_str = best[1]
        chosen_configs[fam] = cfg_str
        for name in names:
            adjusted[name] = cfg_str

    return adjusted, chosen_configs


def apply_expert_consensus_to_curve(curve: List[Dict], lookup: Dict[str, LayerInfo]) -> Tuple[List[Dict], Dict[str, str]]:
    adjusted_curve = []
    last_consensus: Dict[str, str] = {}
    for point in curve:
        recipe, consensus = apply_expert_consensus_to_recipe(point["recipe"], lookup)
        cost_bytes, weighted_error, avg_bpw = recipe_cost_error(recipe, lookup)
        new_point = dict(point)
        new_point["recipe"] = recipe
        new_point["cost_bytes"] = cost_bytes
        new_point["weighted_error"] = weighted_error
        new_point["avg_bpw"] = avg_bpw
        if consensus:
            new_point["expert_consensus"] = consensus
            last_consensus = consensus
        adjusted_curve.append(new_point)
    return adjusted_curve, last_consensus


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


def find_curve_knee(curve: List[Dict], x_key: str = "cost_bytes",
                    y_key: str = "weighted_error") -> int:
    """Kneedle-style knee index for a monotone frontier."""
    if len(curve) < 3:
        return max(0, len(curve) - 1)
    xs = [pt[x_key] for pt in curve]
    ys = [pt[y_key] for pt in curve]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    xr = (x1 - x0) or 1.0
    yr = (y1 - y0) or 1.0
    norm = [((x - x0) / xr, (y - y0) / yr) for x, y in zip(xs, ys)]
    a = norm[0]
    b = norm[-1]
    denom = ((b[1] - a[1]) ** 2 + (b[0] - a[0]) ** 2) ** 0.5 or 1.0
    best_i = 0
    best_d = -1.0
    for i, (x, y) in enumerate(norm):
        d = abs((b[1] - a[1]) * x - (b[0] - a[0]) * y + b[0] * a[1] - b[1] * a[0]) / denom
        if d > best_d:
            best_i = i
            best_d = d
    return best_i


MINIMAX_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.block_sparse_moe\.experts\.)"
    r"(?P<expert>\d+)"
    r"(?P<suffix>\.(w[123])\.weight)$"
)


def _collapse_expert_refs(
    raw_refs: List[Dict],
    experts_per_family_sample: int,
) -> List[Dict]:
    """Collapse repeated MoE expert tensors into per-layer family items.

    MiniMax stores each expert as its own 2D weight tensor, which explodes the
    search space. For local runs, we collapse tensors like:
      model.layers.7.block_sparse_moe.experts.{0..255}.w1.weight
    into one family item keyed as:
      model.layers.7.block_sparse_moe.experts.*.w1.weight

    The family item evaluates a small set of evenly spaced expert samples while
    charging the full family memory cost. This keeps the run local and much more
    representative than a tiny global HAWQ sample.
    """
    passthrough: List[Dict] = []
    families: Dict[str, List[Tuple[int, Dict]]] = {}

    for ref in raw_refs:
        m = MINIMAX_EXPERT_RE.match(ref["name"])
        if not m:
            passthrough.append(ref)
            continue
        family_name = f"{m.group('prefix')}*{m.group('suffix')}"
        expert_idx = int(m.group("expert"))
        families.setdefault(family_name, []).append((expert_idx, ref))

    collapsed = list(passthrough)
    for family_name, members in sorted(families.items()):
        members.sort(key=lambda item: item[0])
        all_refs = [ref for _idx, ref in members]
        sample_count = min(max(1, experts_per_family_sample), len(all_refs))
        if sample_count >= len(all_refs):
            sampled_refs = all_refs
        else:
            sample_positions = np.linspace(0, len(all_refs) - 1, sample_count, dtype=np.int64)
            sampled_refs = [all_refs[int(pos)] for pos in sample_positions]

        representative_shape = sampled_refs[0]["shape"]
        representative_elements = sampled_refs[0]["n_elements"]
        collapsed.append({
            "name": family_name,
            "shape": representative_shape,
            "n_elements": representative_elements * len(all_refs),
            "st_file": sampled_refs[0]["st_file"],
            "tensor_key": sampled_refs[0]["tensor_key"],
            "expert_idx": None,
            "sample_expert_indices": [ref["name"] for ref in sampled_refs],
            "sample_tensor_refs": [
                {"st_file": ref["st_file"], "tensor_key": ref["tensor_key"]}
                for ref in sampled_refs
            ],
            "members": [ref["name"] for ref in all_refs],
        })

    collapsed.sort(key=lambda x: -x["n_elements"])
    return collapsed


def discover_model_layers(model_path: Path, max_layers: int = None,
                         modality_policy: str = "text-only",
                         collapse_expert_families: bool = False,
                         experts_per_family_sample: int = 1) -> List[Dict]:
    """Discover quantizable weight tensors without loading them all into RAM."""
    if modality_policy != "text-only":
        raise NotImplementedError(
            f"Unsupported modality_policy={modality_policy!r}. "
            "Only text-only is supported right now."
        )
    layers = []
    st_files = sorted(model_path.glob("*.safetensors"))

    for st_file in st_files:
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if "layernorm" in key.lower() or "norm" in key.lower():
                    continue
                if not (
                    key.startswith("model.language_model.")
                    or key.startswith("language_model.")
                    or key.startswith("model.layers.")
                    or key.startswith("layers.")
                    or key.startswith("model.embed_tokens.")
                    or key.startswith("embed_tokens.")
                    or key.startswith("lm_head.")
                    or key.startswith("model.lm_head.")
                    or key.startswith("language_model.lm_head.")
                ):
                    continue

                try:
                    shape = tuple(f.get_slice(key).get_shape())
                except Exception:
                    shape = tuple(f.get_tensor(key).shape)

                if len(shape) not in (2, 3):
                    continue
                if len(shape) == 1:
                    continue
                if not key.endswith(".weight"):
                    continue

                n_elements = 1
                for dim in shape:
                    n_elements *= dim

                # MoE expert tensors participate as full fused tensors. This is
                # slower than proxy sampling, but it keeps the recipe faithful to
                # the actual storage object that will be exported.
                if "experts" in key and len(shape) == 3:
                    layers.append({
                        "name": key,
                        "shape": shape,
                        "n_elements": n_elements,
                        "st_file": str(st_file),
                        "tensor_key": key,
                        "expert_idx": None,
                        "sample_expert_indices": None,
                    })
                else:
                    layers.append({
                        "name": key,
                        "shape": shape,
                        "n_elements": n_elements,
                        "st_file": str(st_file),
                        "tensor_key": key,
                        "expert_idx": None,
                        "sample_expert_indices": None,
                    })

                if max_layers and len(layers) >= max_layers:
                    break
        if max_layers and len(layers) >= max_layers:
            break

    if collapse_expert_families:
        layers = _collapse_expert_refs(layers, experts_per_family_sample)

    # Sort by size descending for better load balancing (big layers first)
    layers.sort(key=lambda x: -x["n_elements"])
    if max_layers:
        layers = layers[:max_layers]
    return layers


def load_layer_array(layer_ref: Dict) -> np.ndarray:
    """Load one tensor (or expert slice) as a flattened float32 numpy array."""
    sample_refs = layer_ref.get("sample_tensor_refs")
    if sample_refs:
        arrays = []
        for ref in sample_refs:
            with safe_open(ref["st_file"], framework="pt", device="cpu") as f:
                tensor = f.get_tensor(ref["tensor_key"])
            arrays.append(tensor.flatten().float().numpy())
        return np.concatenate(arrays, axis=0)
    with safe_open(layer_ref["st_file"], framework="pt", device="cpu") as f:
        tensor = f.get_tensor(layer_ref["tensor_key"])
    if layer_ref["expert_idx"] is not None:
        tensor = tensor[layer_ref["expert_idx"]]
    return tensor.flatten().float().numpy()


def load_layer_tensor(layer_ref: Dict) -> torch.Tensor:
    """Load one tensor as a CPU torch tensor without full float32 expansion."""
    sample_refs = layer_ref.get("sample_tensor_refs")
    if sample_refs:
        chunks = []
        for ref in sample_refs:
            with safe_open(ref["st_file"], framework="pt", device="cpu") as f:
                tensor = f.get_tensor(ref["tensor_key"])
            chunks.append(tensor.reshape(-1).contiguous())
        return torch.cat(chunks, dim=0)
    with safe_open(layer_ref["st_file"], framework="pt", device="cpu") as f:
        tensor = f.get_tensor(layer_ref["tensor_key"])
    if layer_ref["expert_idx"] is not None:
        tensor = tensor[layer_ref["expert_idx"]]
    return tensor.contiguous()


def sample_tensor_values(tensor: torch.Tensor, sample_n: int = 4096) -> np.ndarray:
    """Cheap content sample for cache keys and approximate stats."""
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return np.empty(0, dtype=np.float32)
    if flat.numel() <= sample_n:
        return flat.float().cpu().numpy()
    idx = torch.linspace(0, flat.numel() - 1, sample_n, dtype=torch.int64)
    return flat.index_select(0, idx).float().cpu().numpy()


def compute_layer_stats_sampled(sample: np.ndarray) -> Dict[str, float]:
    """Approximate stats from a representative sample."""
    return compute_layer_stats(sample)


def maybe_write_progress(output_path: Optional[str], payload: Dict):
    """Best-effort checkpoint write for long incremental runs."""
    if not output_path:
        return
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, output_path)


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
    parser.add_argument("--no-expert-consensus", action="store_true",
                        help="Disable shared-config consensus for MoE expert families in output recipes")
    parser.add_argument("--modality-policy", type=str, default="text-only",
                        help="Tensor selection policy. Only text-only is supported right now.")
    parser.add_argument("--enforce-nvfp4-fp4", action="store_true", default=True,
                        help="Require 4-bit configs to use g=16 so FP4 remains NVFP4-aligned")
    parser.add_argument("--allow-non-nvfp4-fp4", action="store_true",
                        help="Disable the NVFP4 g=16 constraint for 4-bit configs")
    parser.add_argument("--workers", type=int, default=None,
                        help="Process workers for frontier evaluation. Default is memory-safe, not max CPU.")
    parser.add_argument("--max-pending-jobs", type=int, default=None,
                        help="Maximum queued layer jobs kept in memory at once. Default = 2x workers.")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Write a progress checkpoint to --output every N completed layers.")
    parser.add_argument("--eval-backend", choices=["cpu", "gpu"], default="cpu",
                        help="Local frontier evaluation backend. cpu is default; gpu is a fast path for smoke tests.")
    parser.add_argument("--collapse-expert-families", action="store_true",
                        help="Collapse repeated MoE expert tensors into one per-layer family item before frontier building.")
    parser.add_argument("--experts-per-family-sample", type=int, default=4,
                        help="When collapsing expert families, evaluate this many evenly spaced experts per family.")
    parser.add_argument("--routing-prior", choices=["none", "minimax-gate"], default="none",
                        help="Optional checkpoint-only routing prior to reweight MoE expert families.")
    parser.add_argument("--routing-prior-strength", type=float, default=1.0,
                        help="Exponent applied to the routing prior multiplier before sensitivity weighting.")
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
    w_bits = parse_int_list(args.w_bits)
    s_bits = parse_int_list(args.s_bits)
    g_sizes = parse_int_list(args.g_sizes)
    baseline_config = parse_config(args.baseline_config)
    enforce_nvfp4_fp4 = args.enforce_nvfp4_fp4 and not args.allow_non_nvfp4_fp4
    all_configs = [
        cfg for cfg in build_configs(w_bits, s_bits, g_sizes)
        if config_is_allowed(*cfg, enforce_nvfp4_fp4=enforce_nvfp4_fp4)
    ]
    worker_count = args.workers or min(N_WORKERS, 8)
    max_pending_jobs = args.max_pending_jobs or max(1, worker_count * 2)
    print(f"Workers: {worker_count}")
    print(f"Max pending jobs: {max_pending_jobs}")
    print(f"Configs per layer: {len(all_configs)}")
    print(f"w_bits={w_bits}")
    print(f"s_bits={s_bits}")
    print(f"g_sizes={g_sizes}")
    print(f"baseline={baseline_config}")
    print(f"enforce_nvfp4_fp4={enforce_nvfp4_fp4}")
    print(f"modality_policy={args.modality_policy}")
    print(f"eval_backend={args.eval_backend}")
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

    routing_prior = {}
    if args.routing_prior == "minimax-gate":
        print("Loading MiniMax gate-based routing prior...")
        routing_prior = load_minimax_gate_priors(model_path)
        if routing_prior:
            vals = list(routing_prior.values())
            print(f"  Loaded priors for {len(routing_prior)} expert families "
                  f"(min={min(vals):.3f} mean={sum(vals)/len(vals):.3f} max={max(vals):.3f})")
        else:
            print("  No routing priors found")

    # Load layers
    print("\nLoading model weights...")
    t_load = time.time()
    raw_layers = discover_model_layers(
        model_path,
        args.max_layers,
        args.modality_policy,
        collapse_expert_families=args.collapse_expert_families,
        experts_per_family_sample=args.experts_per_family_sample,
    )
    print(f"Discovered {len(raw_layers)} weight tensors in {time.time() - t_load:.1f}s")
    if args.collapse_expert_families:
        collapsed = sum(1 for layer in raw_layers if layer.get("sample_tensor_refs"))
        print(f"Collapsed expert families: {collapsed} items "
              f"(samples per family: {args.experts_per_family_sample})")

    # Filter out large layers if requested (for fast iteration)
    if args.skip_large:
        threshold = args.skip_large * 1_000_000
        before = len(raw_layers)
        raw_layers = [layer for layer in raw_layers if layer["n_elements"] < threshold]
        print(f"Skipped {before - len(raw_layers)} layers > {args.skip_large}M elements")

    total_elements = sum(layer["n_elements"] for layer in raw_layers)
    print(f"Total elements: {total_elements:,}")
    print(flush=True)

    if not raw_layers or total_elements == 0:
        raise RuntimeError(
            f"No weight tensors were discovered under {model_path}. "
            "Check the model path and safetensors contents."
        )

    # Build Pareto frontiers
    if args.eval_backend == "gpu":
        print(f"\nBuilding Pareto frontiers (single-process streaming GPU)...")
    else:
        print(f"\nBuilding Pareto frontiers ({worker_count} processes)...")
    t_pareto = time.time()

    layers = []
    completed = 0
    cache_hits = 0

    # Add use_cache flag and HAWQ sensitivity to each layer
    use_cache = not args.no_cache
    layer_jobs = []
    for layer in raw_layers:
        name = layer["name"]
        # Look up HAWQ sensitivity for this layer
        layer_sens = None
        if hawq_sens:
            for candidate in candidate_sensitivity_names(name):
                layer_sens = hawq_sens.get(candidate)
                if layer_sens is not None:
                    break
        if routing_prior and name in routing_prior:
            if layer_sens is None:
                layer_sens = 1.0
            layer_sens *= routing_prior[name] ** args.routing_prior_strength
        layer_jobs.append((layer, use_cache, layer_sens))

    def record_result(result):
        nonlocal completed, cache_hits
        name, shape, n_elements, members, sensitivity, stats, pareto_tuples, was_cached = result
        if was_cached:
            cache_hits += 1

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
            members=members,
        ))

        completed += 1
        if completed % 10 == 0 or completed == len(raw_layers):
            print(f"  {completed}/{len(raw_layers)} layers processed", flush=True)
        if args.output and (completed % args.checkpoint_every == 0 or completed == len(raw_layers)):
            maybe_write_progress(args.output, {
                "model": str(model_path),
                "phase": "frontier_build",
                "completed_layers": completed,
                "total_layers": len(raw_layers),
                "cache_hits": cache_hits,
                "search_space": {
                    "w_bits": w_bits,
                    "s_bits": s_bits,
                    "g_sizes": g_sizes,
                },
                "baseline_config": baseline_config,
            })

    if args.eval_backend == "gpu":
        # One process owns the GPU. This avoids Python multiprocessing and lets
        # us stream tensors/configs directly through one device-resident path.
        for layer_ref, use_cache_flag, layer_sens in layer_jobs:
            result = evaluate_layer_local(layer_ref, use_cache_flag, layer_sens, all_configs, args.eval_backend)
            record_result(result)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {}
            job_idx = 0

            def submit_one():
                nonlocal job_idx
                if job_idx >= len(layer_jobs):
                    return False
                layer_ref, use_cache_flag, layer_sens = layer_jobs[job_idx]
                weights = load_layer_array(layer_ref)
                payload = (
                    layer_ref["name"],
                    weights,
                    layer_ref["shape"],
                    layer_ref["n_elements"],
                    list(layer_ref.get("members", [])),
                    use_cache_flag,
                    layer_sens,
                    all_configs,
                    args.eval_backend,
                )
                fut = executor.submit(evaluate_layer_worker, payload)
                futures[fut] = layer_ref["name"]
                job_idx += 1
                return True

            while len(futures) < max_pending_jobs and submit_one():
                pass

            while futures:
                done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    futures.pop(future, None)
                    record_result(result)

                    while len(futures) < max_pending_jobs and submit_one():
                        pass

    print(f"Pareto frontiers built in {time.time() - t_pareto:.1f}s (cache hits: {cache_hits}/{len(raw_layers)})")
    print(flush=True)

    optimization_layers = layers
    aggregate_map: Dict[str, List[str]] = {}
    if not args.no_expert_consensus:
        optimization_layers, aggregate_map = aggregate_expert_families_for_optimization(layers)
        if aggregate_map:
            print("\nApplying expert-family shared-config constraint during optimization:")
            for fam, members in sorted(aggregate_map.items()):
                print(f"  {fam}: {len(members)} tensors")
            print(f"Reduced optimization items: {len(layers)} -> {len(optimization_layers)}")
            print(flush=True)

    # Water-fill to get full Pareto curve
    print("\nWater-filling over 3D Pareto frontiers...")
    t_wf = time.time()
    pareto_curve = water_fill_pareto(optimization_layers)
    if aggregate_map:
        pareto_curve = expand_curve_from_aggregates(pareto_curve, aggregate_map)
    print(f"Water-fill time: {time.time() - t_wf:.2f}s")
    print(f"Pareto curve has {len(pareto_curve)} points")
    print(flush=True)

    print("\nBuilding promotion ladder above baseline...")
    t_promote = time.time()
    promotion_curve = build_promotion_curve(optimization_layers, baseline_config)
    if aggregate_map:
        promotion_curve = expand_curve_from_aggregates(promotion_curve, aggregate_map)
    lookup = {layer.name: layer for layer in layers}
    pareto_consensus = {}
    promotion_consensus = {}
    promotion_knee_idx = find_curve_knee(promotion_curve)
    promotion_knee = promotion_curve[promotion_knee_idx]
    if aggregate_map:
        promotion_consensus = {
            fam: promotion_knee["recipe"][members[0]]
            for fam, members in sorted(aggregate_map.items())
            if members and members[0] in promotion_knee["recipe"]
        }
        pareto_consensus = {
            fam: pareto_curve[-1]["recipe"][members[0]]
            for fam, members in sorted(aggregate_map.items())
            if members and members[0] in pareto_curve[-1]["recipe"]
        }
        if promotion_consensus:
            print("Applied expert-family constraint inside optimizer:")
            for fam, cfg_str in sorted(promotion_consensus.items()):
                print(f"  {fam}: {cfg_str}")
    print(f"Promotion ladder time: {time.time() - t_promote:.2f}s")
    print(f"Promotion ladder has {len(promotion_curve)} points")
    print(
        f"Promotion kneedle: step {promotion_knee['step']}  "
        f"bpw={promotion_knee['avg_bpw']:.3f}  "
        f"memory={promotion_knee['cost_bytes']:,}  "
        f"error={promotion_knee['weighted_error']:.3e}"
    )

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
            "promotion_knee": promotion_knee,
            "outliers": outliers,
            "expert_consensus": {
                "enabled": not args.no_expert_consensus,
                "pareto": pareto_consensus,
                "promotion": promotion_consensus,
            },
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
