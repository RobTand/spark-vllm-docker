#!/usr/bin/env python3
"""
Compare native deployment buckets directly on a model checkpoint.

This is separate from joint_knapsack_optimizer.py on purpose: native formats
like NVFP4 / MXFP4 / MXFP6 / MXFP8 are not faithfully described by the generic
(w_bits, s_bits, g_size) abstraction. This script compares them as explicit
deployment buckets with their own quantization semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantization.joint_knapsack_optimizer import (
    candidate_sensitivity_names,
    compute_layer_stats,
    discover_model_layers,
    find_curve_knee,
    load_hawq_sensitivity,
    load_layer_tensor,
)
from quantization.build_rtn_cache import rtn_fp8_any_shape


@dataclass(frozen=True)
class NativeBucket:
    name: str
    group_size: int
    scale_mode: str
    codebook: str
    bits_per_weight: float


@dataclass
class BucketResult:
    bucket: NativeBucket
    mse: float
    memory_bytes: int
    bits_per_weight: float


@dataclass
class LayerStudy:
    name: str
    n_elements: int
    sensitivity: float
    stats: Dict[str, float]
    buckets: List[BucketResult]


BUCKETS: Dict[str, NativeBucket] = {
    "nvfp4": NativeBucket("nvfp4", group_size=16, scale_mode="fp8_e4m3", codebook="fp4_e2m1", bits_per_weight=4.5),
    "mxfp4": NativeBucket("mxfp4", group_size=32, scale_mode="e8m0_pow2", codebook="fp4_e2m1", bits_per_weight=4.25),
    "mxfp6_e2m3": NativeBucket("mxfp6_e2m3", group_size=32, scale_mode="e8m0_pow2", codebook="fp6_e2m3", bits_per_weight=6.25),
    "mxfp6_e3m2": NativeBucket("mxfp6_e3m2", group_size=32, scale_mode="e8m0_pow2", codebook="fp6_e3m2", bits_per_weight=6.25),
    "fp8_e4m3": NativeBucket("fp8_e4m3", group_size=1, scale_mode="bf16", codebook="fp8_e4m3", bits_per_weight=8.0),
    "fp8_e5m2": NativeBucket("fp8_e5m2", group_size=1, scale_mode="bf16", codebook="fp8_e5m2", bits_per_weight=8.0),
    "mxfp8_e4m3": NativeBucket("mxfp8_e4m3", group_size=32, scale_mode="e8m0_pow2", codebook="fp8_e4m3", bits_per_weight=8.25),
    "mxfp8_e5m2": NativeBucket("mxfp8_e5m2", group_size=32, scale_mode="e8m0_pow2", codebook="fp8_e5m2", bits_per_weight=8.25),
    "bf16": NativeBucket("bf16", group_size=1, scale_mode="bf16", codebook="bf16", bits_per_weight=16.0),
}


def build_float_codebook(exp_bits: int, mant_bits: int) -> torch.Tensor:
    bias = (1 << (exp_bits - 1)) - 1
    values = {0.0}
    max_exp = (1 << exp_bits) - 1
    for sign in (-1.0, 1.0):
        for exp in range(max_exp):
            if exp == max_exp:
                continue
            if exp == 0:
                exponent = 1 - bias
                for mant in range(1, 1 << mant_bits):
                    frac = mant / float(1 << mant_bits)
                    values.add(sign * frac * (2.0 ** exponent))
            else:
                exponent = exp - bias
                for mant in range(0, 1 << mant_bits):
                    frac = 1.0 + mant / float(1 << mant_bits)
                    values.add(sign * frac * (2.0 ** exponent))
    out = torch.tensor(sorted(values), dtype=torch.float32)
    return out


CODEBOOKS: Dict[str, torch.Tensor] = {
    "fp4_e2m1": torch.tensor(sorted({0.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}), dtype=torch.float32),
    "fp6_e2m3": build_float_codebook(2, 3),
    "fp6_e3m2": build_float_codebook(3, 2),
    "fp8_e4m3": build_float_codebook(4, 3),
    "fp8_e5m2": build_float_codebook(5, 2),
}


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


def quantize_scale_e8m0_pow2_torch(scales: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(scales)
    nz = scales > 0
    if not torch.any(nz):
        return result
    vals = torch.clamp(scales[nz], min=2.0 ** -127, max=2.0 ** 127)
    result[nz] = torch.pow(torch.tensor(2.0, device=vals.device, dtype=vals.dtype), torch.round(torch.log2(vals)))
    return result


def quantize_scales(scales: torch.Tensor, scale_mode: str) -> torch.Tensor:
    if scale_mode == "bf16":
        return scales
    if scale_mode == "fp8_e4m3":
        return quantize_scale_fp8_e4m3_torch(scales)
    if scale_mode == "e8m0_pow2":
        return quantize_scale_e8m0_pow2_torch(scales)
    raise ValueError(f"unsupported scale_mode={scale_mode}")


def apply_codebook(x: torch.Tensor, codebook_name: str) -> torch.Tensor:
    if codebook_name == "bf16":
        return x.to(torch.bfloat16).float()
    if codebook_name == "fp8_e4m3":
        finfo = torch.finfo(torch.float8_e4m3fn)
        return torch.clamp(x, min=-finfo.max, max=finfo.max).to(torch.float8_e4m3fn).float()
    if codebook_name == "fp8_e5m2":
        finfo = torch.finfo(torch.float8_e5m2)
        return torch.clamp(x, min=-finfo.max, max=finfo.max).to(torch.float8_e5m2).float()
    codebook = CODEBOOKS[codebook_name].to(device=x.device)
    view = x.reshape(-1, 1)
    idx = torch.argmin(torch.abs(view - codebook.view(1, -1)), dim=1)
    return codebook.index_select(0, idx).reshape_as(x)


def bucket_memory_bytes(n_elements: int, bucket: NativeBucket) -> int:
    if bucket.name == "bf16":
        return n_elements * 2
    if bucket.name.startswith("fp8_"):
        return n_elements
    weight_bytes = math.ceil(n_elements * (bucket.bits_per_weight - (8.0 / bucket.group_size)) / 8.0)
    scale_bytes = math.ceil((n_elements / bucket.group_size))
    if bucket.name == "nvfp4":
        scale_bytes += 4
    return weight_bytes + scale_bytes


def quantize_tensor_to_bucket(weights: torch.Tensor, bucket: NativeBucket, chunk_groups: int = 262_144) -> torch.Tensor:
    """Round-trip a tensor through a native bucket approximation."""
    if bucket.name == "fp8_e4m3":
        return rtn_fp8_any_shape(weights.to(torch.float32)).to(dtype=weights.dtype)
    if bucket.name == "fp8_e5m2":
        w = weights.to(torch.float32)
        if w.dim() == 2:
            max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            finfo = torch.finfo(torch.float8_e5m2)
            scale = max_abs / finfo.max
            q = torch.clamp(w / scale, min=-finfo.max, max=finfo.max).to(torch.float8_e5m2).float()
            return (q * scale).to(dtype=weights.dtype)
        if w.dim() == 3:
            e, out_f, in_f = w.shape
            flat = w.reshape(e * out_f, in_f)
            max_abs = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            finfo = torch.finfo(torch.float8_e5m2)
            scale = max_abs / finfo.max
            q = torch.clamp(flat / scale, min=-finfo.max, max=finfo.max).to(torch.float8_e5m2).float()
            return (q * scale).reshape_as(w).to(dtype=weights.dtype)
        return torch.clamp(w, min=-torch.finfo(torch.float8_e5m2).max, max=torch.finfo(torch.float8_e5m2).max).to(torch.float8_e5m2).float().to(dtype=weights.dtype)
    original_shape = tuple(weights.shape)
    flat_cpu = weights.reshape(-1).to(dtype=torch.float32, device="cpu")
    original_n = flat_cpu.numel()
    device = torch.device("cuda")
    chunks: List[torch.Tensor] = []

    with torch.no_grad():
        total_groups = (original_n + bucket.group_size - 1) // bucket.group_size
        for start_group in range(0, total_groups, chunk_groups):
            start = start_group * bucket.group_size
            end = min(original_n, (start_group + chunk_groups) * bucket.group_size)
            chunk_cpu = flat_cpu[start:end]
            valid_n = chunk_cpu.numel()
            if valid_n % bucket.group_size != 0:
                pad = bucket.group_size - (valid_n % bucket.group_size)
                chunk_cpu = torch.nn.functional.pad(chunk_cpu, (0, pad))
            if not chunk_cpu.is_pinned():
                chunk_cpu = chunk_cpu.pin_memory()
            chunk = chunk_cpu.to(device=device, dtype=torch.float32, non_blocking=True)

            if bucket.name == "bf16":
                recon = chunk.to(torch.bfloat16).float()
            else:
                groups = chunk.view(-1, bucket.group_size)
                max_abs = groups.abs().amax(dim=1)
                valid = max_abs > 1e-10
                raw_scales = torch.zeros_like(max_abs)
                vmax = torch.max(torch.abs(CODEBOOKS[bucket.codebook]))
                raw_scales[valid] = max_abs[valid] / vmax
                scales = quantize_scales(raw_scales, bucket.scale_mode)
                scales_expanded = scales[:, None]
                scales_safe = torch.where(scales_expanded > 1e-10, scales_expanded, torch.ones_like(scales_expanded))
                normalized = groups / scales_safe
                q = apply_codebook(normalized, bucket.codebook)
                recon = (q * scales_expanded).reshape(-1)

            chunks.append(recon[:valid_n].to(device="cpu", dtype=weights.dtype))
            del chunk, recon, chunk_cpu

    torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0).reshape(original_shape).contiguous()


def measure_bucket_mse(weights: torch.Tensor, bucket: NativeBucket, chunk_groups: int = 262_144) -> float:
    recon = quantize_tensor_to_bucket(weights, bucket, chunk_groups=chunk_groups).reshape(-1).float()
    ref = weights.reshape(-1).float().cpu()
    diff = ref - recon.cpu()
    return float(torch.mean(diff * diff).item()) if diff.numel() else 0.0


def build_frontier(results: List[BucketResult]) -> List[BucketResult]:
    keep = []
    for i, r1 in enumerate(results):
        dominated = False
        for j, r2 in enumerate(results):
            if i == j:
                continue
            if (r2.mse < r1.mse and r2.memory_bytes <= r1.memory_bytes) or (
                r2.mse <= r1.mse and r2.memory_bytes < r1.memory_bytes
            ):
                dominated = True
                break
        if not dominated:
            keep.append(r1)
    keep.sort(key=lambda r: r.memory_bytes)
    return keep


def water_fill(layers: List[LayerStudy]) -> List[Dict]:
    lookup = {layer.name: layer for layer in layers}
    state = {layer.name: 0 for layer in layers}
    total_elems = sum(layer.n_elements for layer in layers)

    def total_cost() -> int:
        return sum(lookup[n].buckets[idx].memory_bytes for n, idx in state.items())

    def total_error() -> float:
        return sum(lookup[n].sensitivity * lookup[n].buckets[idx].mse for n, idx in state.items())

    import heapq

    def marginal_score(name: str):
        layer = lookup[name]
        cur = state[name]
        if cur + 1 >= len(layer.buckets):
            return None
        cur_cfg = layer.buckets[cur]
        nxt_cfg = layer.buckets[cur + 1]
        dcost = nxt_cfg.memory_bytes - cur_cfg.memory_bytes
        derr = layer.sensitivity * (cur_cfg.mse - nxt_cfg.mse)
        if dcost <= 0 or derr <= 0:
            return None
        return derr / dcost, cur + 1

    heap = []
    for layer in layers:
        m = marginal_score(layer.name)
        if m is not None:
            score, nxt = m
            heapq.heappush(heap, (-score, layer.name, nxt))

    curve = [{
        "step": 0,
        "cost_bytes": total_cost(),
        "weighted_error": total_error(),
        "avg_bpw": total_cost() * 8 / total_elems,
        "recipe": {n: lookup[n].buckets[state[n]].bucket.name for n in state},
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
            "recipe": {n: lookup[n].buckets[state[n]].bucket.name for n in state},
        })
        m = marginal_score(name)
        if m is not None:
            score, nxt = m
            heapq.heappush(heap, (-score, name, nxt))

    return curve


def promotion_curve(layers: List[LayerStudy], baseline: str) -> List[Dict]:
    lookup = {layer.name: layer for layer in layers}
    state = {}
    total_elems = sum(layer.n_elements for layer in layers)
    import heapq

    for layer in layers:
        base_idx = next((i for i, cfg in enumerate(layer.buckets) if cfg.bucket.name == baseline), 0)
        state[layer.name] = base_idx

    def total_cost() -> int:
        return sum(lookup[n].buckets[idx].memory_bytes for n, idx in state.items())

    def total_error() -> float:
        return sum(lookup[n].sensitivity * lookup[n].buckets[idx].mse for n, idx in state.items())

    def marginal_score(name: str):
        layer = lookup[name]
        cur = state[name]
        best = None
        for nxt in range(cur + 1, len(layer.buckets)):
            cur_cfg = layer.buckets[cur]
            nxt_cfg = layer.buckets[nxt]
            dcost = nxt_cfg.memory_bytes - cur_cfg.memory_bytes
            derr = layer.sensitivity * (cur_cfg.mse - nxt_cfg.mse)
            if dcost <= 0 or derr <= 0:
                continue
            score = derr / dcost
            if best is None or score > best[0]:
                best = (score, nxt)
        return best

    heap = []
    for layer in layers:
        m = marginal_score(layer.name)
        if m is not None:
            score, nxt = m
            heapq.heappush(heap, (-score, layer.name, nxt))

    curve = [{
        "step": 0,
        "cost_bytes": total_cost(),
        "weighted_error": total_error(),
        "avg_bpw": total_cost() * 8 / total_elems,
        "recipe": {n: lookup[n].buckets[state[n]].bucket.name for n in state},
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
            "recipe": {n: lookup[n].buckets[state[n]].bucket.name for n in state},
        })
        m = marginal_score(name)
        if m is not None:
            score, nxt = m
            heapq.heappush(heap, (-score, name, nxt))

    return curve


def main():
    parser = argparse.ArgumentParser(description="Native format study")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sensitivity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--modality-policy", default="text-only")
    parser.add_argument("--baseline", default="nvfp4")
    parser.add_argument(
        "--buckets",
        default="nvfp4,mxfp4,mxfp6_e2m3,mxfp6_e3m2,fp8_e4m3,fp8_e5m2,mxfp8_e4m3,mxfp8_e5m2,bf16",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    hawq = load_hawq_sensitivity(args.sensitivity)
    bucket_names = [b.strip() for b in args.buckets.split(",") if b.strip()]
    buckets = [BUCKETS[name] for name in bucket_names]

    print(f"[native-study] model={model_path}", flush=True)
    print(f"[native-study] buckets={bucket_names}", flush=True)

    raw_layers = discover_model_layers(model_path, args.max_layers, args.modality_policy)
    print(f"[native-study] discovered {len(raw_layers)} tensors", flush=True)

    layers: List[LayerStudy] = []
    for idx, layer_ref in enumerate(raw_layers, start=1):
        name = layer_ref["name"]
        layer_sens = None
        for candidate in candidate_sensitivity_names(name):
            layer_sens = hawq.get(candidate)
            if layer_sens is not None:
                break
        if layer_sens is None:
            weights_sample = load_layer_tensor(layer_ref).reshape(-1)[:4096].float()
            layer_sens = float(torch.sqrt(torch.mean(weights_sample ** 2)).item())
        tensor = load_layer_tensor(layer_ref)
        stats = compute_layer_stats(tensor)
        results = []
        for bucket in buckets:
            mse = measure_bucket_mse(tensor, bucket)
            mem = bucket_memory_bytes(layer_ref["n_elements"], bucket)
            results.append(BucketResult(bucket=bucket, mse=mse, memory_bytes=mem, bits_per_weight=bucket.bits_per_weight))
        frontier = build_frontier(results)
        layers.append(LayerStudy(name=name, n_elements=layer_ref["n_elements"], sensitivity=layer_sens, stats=stats, buckets=frontier))
        if idx % 10 == 0 or idx == len(raw_layers):
            print(f"[native-study] {idx}/{len(raw_layers)} tensors", flush=True)

    pareto = water_fill(layers)
    promo = promotion_curve(layers, args.baseline)
    knee_idx = find_curve_knee(promo)
    out = {
        "model": str(model_path),
        "buckets": bucket_names,
        "n_layers": len(layers),
        "pareto_curve": pareto,
        "promotion_curve": promo,
        "promotion_knee": promo[knee_idx],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[native-study] promotion kneedle: step {promo[knee_idx]['step']}  bpw={promo[knee_idx]['avg_bpw']:.3f}  "
          f"memory={promo[knee_idx]['cost_bytes']:,}  error={promo[knee_idx]['weighted_error']:.3e}", flush=True)
    print(f"[native-study] saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
