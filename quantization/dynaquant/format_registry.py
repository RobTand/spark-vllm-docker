"""format_registry.py — extensible quantization format catalog.

A FormatSpec describes everything the pipeline needs to treat a format
uniformly:

  - name                    canonical identifier
  - weight_bits             weight element width
  - group_size              per-group scale granularity; 0 = per-channel
  - weight_element_dtype    torch dtype used on disk
  - scale_bits              bits consumed by per-group scales
  - scale_dtype_name        human-readable scale dtype
  - effective_bits          total bits/param (weight + scale amortized)
  - autoround_config()      dict AutoRound consumes via --layer_config
  - quantize_dequantize(w)  apply RTN in-place (for closed-loop MSE)
  - min_capability_sm       minimum SM arch (useful for hardware filter)

Users register new formats with @register_format. New hardware formats
(e.g. a future MXFP6 variant, Ada W4A8, INT3) can be added without
touching core code.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class FormatSpec:
    name: str
    weight_bits: int
    group_size: int          # 0 means per-channel; >=1 = block size
    scale_bits: int          # bits per scale element
    scale_dtype_name: str    # "fp8_e4m3", "uint8_e8m0", "fp32", ...
    weight_element_dtype: str   # "fp4_e2m1", "fp8_e4m3", "fp6_e3m2", "int4", ...
    act_bits: int | None = None   # None = no activation quant (W8A16)
    act_dtype_name: str | None = None
    act_group_size: int | None = None
    family: str = "generic"       # "nv", "mx", "int", "fp"
    min_capability_sm: int = 80   # minimum CUDA compute capability
    # The autoround layer_config dict. Leave as callable for lazy config.
    autoround_config: Callable[[], dict] = field(default=lambda: {})
    # RTN quantize+dequantize, returns the rounded tensor (same shape+dtype).
    quantize_dequantize: Callable[[torch.Tensor], torch.Tensor] = field(default=lambda x: x)

    @property
    def effective_bits(self) -> float:
        """Average bits per parameter accounting for scales."""
        if self.group_size == 0:
            # per-channel: scales cost is tiny (out_features × scale_bits / weights)
            # caller can override if needed; default assumes amortized negligible
            return float(self.weight_bits) + 0.02
        return float(self.weight_bits) + float(self.scale_bits) / self.group_size


REGISTRY: dict[str, FormatSpec] = {}


def register_format(spec: FormatSpec) -> FormatSpec:
    REGISTRY[spec.name] = spec
    return spec


# -----------------------------------------------------------------------
# Reference format implementations
# -----------------------------------------------------------------------
# Helpers for RTN quantization reference impls.

def _rtn_uniform_int(w: torch.Tensor, bits: int, group_size: int,
                     symmetric: bool = True) -> torch.Tensor:
    """Round-to-nearest uniform-integer quantizer with optional group scaling."""
    orig_shape = w.shape
    out_f, in_f = w.shape[-2], w.shape[-1]
    w2 = w.reshape(-1, in_f).float()
    if group_size > 0 and group_size < in_f:
        w2 = w2.reshape(-1, in_f // group_size, group_size)
    else:
        w2 = w2.unsqueeze(1)
    # Per-group max for scale
    max_abs = w2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    if symmetric:
        levels = (1 << (bits - 1)) - 1
        scale = max_abs / levels
        q = torch.round(w2 / scale).clamp(-levels - 1, levels)
        w_rec = q * scale
    else:
        levels = (1 << bits) - 1
        w_min = w2.amin(dim=-1, keepdim=True)
        w_max = w2.amax(dim=-1, keepdim=True)
        scale = (w_max - w_min) / levels
        zp = torch.round(-w_min / scale.clamp_min(1e-8))
        q = torch.round(w2 / scale.clamp_min(1e-8) + zp).clamp(0, levels)
        w_rec = (q - zp) * scale
    return w_rec.reshape(orig_shape).to(w.dtype)


def _rtn_fp_codebook(w: torch.Tensor, codebook: torch.Tensor,
                     group_size: int) -> torch.Tensor:
    """Round to nearest value in a small FP codebook, with per-group scaling.

    Vectorized via torch.bucketize on the sorted codebook. For each scaled
    weight value x, we binary-search the codebook to find the two bracketing
    entries and pick the closer one. O(N log K) instead of the O(N * K)
    pairwise-distance approach, with 0 extra-dim allocations.
    """
    orig_shape = w.shape
    in_f = w.shape[-1]
    w2 = w.reshape(-1, in_f).float()
    if group_size > 0 and group_size < in_f:
        w2 = w2.reshape(-1, in_f // group_size, group_size)
    else:
        w2 = w2.unsqueeze(1)

    cb = codebook.to(device=w2.device, dtype=torch.float32).contiguous()
    cmax = float(cb.abs().max().item())
    max_abs = w2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = max_abs / cmax
    x = w2 / scale                                    # shape (..., group)

    # Bucketize returns the insertion index: cb[idx-1] <= x < cb[idx].
    idx = torch.bucketize(x.contiguous(), cb)
    idx_lo = (idx - 1).clamp_min(0)
    idx_hi = idx.clamp_max(cb.numel() - 1)
    lo = cb[idx_lo]
    hi = cb[idx_hi]
    choose_hi = (hi - x).abs() < (x - lo).abs()
    q = torch.where(choose_hi, hi, lo)

    w_rec = q * scale
    return w_rec.reshape(orig_shape).to(w.dtype)


# FP codebooks
def _e2m1_codebook() -> torch.Tensor:
    # 4-bit: 1 sign + 2 exp + 1 mantissa.  Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    signed = [0.0] + [+v for v in vals[1:]] + [-v for v in vals[1:]]
    return torch.tensor(sorted(set(signed)), dtype=torch.float32)


def _e3m2_codebook() -> torch.Tensor:
    # 6-bit FP e3m2: 1s + 3 exp + 2 mantissa. 64 codes.
    codes = set([0.0])
    for exp in range(8):
        for m in range(4):
            val = (1 + m / 4) * (2 ** (exp - 3))
            codes.add(+val); codes.add(-val)
    return torch.tensor(sorted(codes), dtype=torch.float32)


def _e2m3_codebook() -> torch.Tensor:
    # 6-bit FP e2m3: 1s + 2 exp + 3 mantissa. 64 codes.
    codes = set([0.0])
    for exp in range(4):
        for m in range(8):
            val = (1 + m / 8) * (2 ** (exp - 1))
            codes.add(+val); codes.add(-val)
    return torch.tensor(sorted(codes), dtype=torch.float32)


def _e4m3_codebook() -> torch.Tensor:
    # 8-bit FP e4m3 (no inf). 256 codes, covering ±448.
    # We compute it analytically from the OCP FP8 spec (nan reserved).
    codes = set([0.0])
    for exp in range(16):
        for m in range(8):
            if exp == 0:
                val = (m / 8) * (2 ** -6)  # subnormals
            else:
                val = (1 + m / 8) * (2 ** (exp - 7))
            codes.add(+val); codes.add(-val)
    # Clip to ±448 per spec (remove overflowing exponents)
    return torch.tensor(sorted(c for c in codes if abs(c) <= 448.0),
                        dtype=torch.float32)


def _e5m2_codebook() -> torch.Tensor:
    # 8-bit FP e5m2. Wider range, less mantissa precision.
    codes = set([0.0])
    for exp in range(32):
        for m in range(4):
            if exp == 0:
                val = (m / 4) * (2 ** -14)
            else:
                val = (1 + m / 4) * (2 ** (exp - 15))
            codes.add(+val); codes.add(-val)
    return torch.tensor(sorted(codes), dtype=torch.float32)


_CODEBOOKS = {
    "fp4_e2m1": _e2m1_codebook(),
    "fp6_e3m2": _e3m2_codebook(),
    "fp6_e2m3": _e2m3_codebook(),
    "fp8_e4m3": _e4m3_codebook(),
    "fp8_e5m2": _e5m2_codebook(),
}


def _make_rtn(codebook_name: str, group_size: int):
    cb = _CODEBOOKS[codebook_name]
    def f(w: torch.Tensor) -> torch.Tensor:
        return _rtn_fp_codebook(w, cb, group_size)
    return f


# -----------------------------------------------------------------------
# Built-in format registrations
# -----------------------------------------------------------------------
# AutoRound layer_config entries match what AutoRound expects for its
# internal QuantizationScheme.  See auto_round.compressors.utils for the
# canonical fields.  Feel free to extend as new formats are added.

def _nv_autoround(bits=4, gsize=16, act_bits=4):
    return dict(
        bits=bits, group_size=gsize, sym=True, data_type="nv_fp",
        act_bits=act_bits, act_group_size=gsize, act_sym=True,
        act_data_type="nv_fp4_with_static_gs" if bits == 4 else "nv_fp",
        act_dynamic=True,
    )


def _mx_autoround(bits=8, gsize=32, act_bits=8, elt="fp8_e4m3"):
    return dict(
        bits=bits, group_size=gsize, sym=True, data_type="mx_fp",
        act_bits=act_bits, act_group_size=gsize, act_sym=True,
        act_data_type="mx_fp", act_dynamic=True,
    )


def _int_autoround(bits, gsize, act_bits=16):
    return dict(
        bits=bits, group_size=gsize, sym=True, data_type="int",
        act_bits=act_bits, act_group_size=gsize if act_bits <= 8 else 0,
        act_sym=True, act_data_type="int" if act_bits <= 8 else "float",
        act_dynamic=True,
    )


# NVFP4 / NVFP4A16  (NVIDIA, group_size=16, FP8 scales)
register_format(FormatSpec(
    name="NVFP4",
    weight_bits=4, group_size=16, scale_bits=8, scale_dtype_name="fp8_e4m3",
    weight_element_dtype="fp4_e2m1", act_bits=4, act_dtype_name="fp4_e2m1",
    act_group_size=16, family="nv", min_capability_sm=100,
    autoround_config=lambda: _nv_autoround(4, 16, 4),
    quantize_dequantize=_make_rtn("fp4_e2m1", 16),
))
register_format(FormatSpec(
    name="NVFP4A16",
    weight_bits=4, group_size=16, scale_bits=8, scale_dtype_name="fp8_e4m3",
    weight_element_dtype="fp4_e2m1", act_bits=None,
    family="nv", min_capability_sm=100,
    autoround_config=lambda: _nv_autoround(4, 16, 16),
    quantize_dequantize=_make_rtn("fp4_e2m1", 16),
))

# MXFP4 / MXFP8 / MXFP6 variants  (OCP MX, group_size=32, E8M0 scales)
register_format(FormatSpec(
    name="MXFP4",
    weight_bits=4, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp4_e2m1", act_bits=4, act_dtype_name="fp4_e2m1",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(4, 32, 4, "fp4_e2m1"),
    quantize_dequantize=_make_rtn("fp4_e2m1", 32),
))
register_format(FormatSpec(
    name="MXFP6_E3M2",
    weight_bits=6, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp6_e3m2", act_bits=6, act_dtype_name="fp6_e3m2",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(6, 32, 6, "fp6_e3m2"),
    quantize_dequantize=_make_rtn("fp6_e3m2", 32),
))
register_format(FormatSpec(
    name="MXFP6_E2M3",
    weight_bits=6, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp6_e2m3", act_bits=6, act_dtype_name="fp6_e2m3",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(6, 32, 6, "fp6_e2m3"),
    quantize_dequantize=_make_rtn("fp6_e2m3", 32),
))
register_format(FormatSpec(
    name="MXFP8",  # alias for MXFP8_E4M3 (OCP MX canonical default)
    weight_bits=8, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp8_e4m3", act_bits=8, act_dtype_name="fp8_e4m3",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(8, 32, 8, "fp8_e4m3"),
    quantize_dequantize=_make_rtn("fp8_e4m3", 32),
))
register_format(FormatSpec(
    name="MXFP8_E4M3",  # explicit name for the canonical variant
    weight_bits=8, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp8_e4m3", act_bits=8, act_dtype_name="fp8_e4m3",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(8, 32, 8, "fp8_e4m3"),
    quantize_dequantize=_make_rtn("fp8_e4m3", 32),
))
register_format(FormatSpec(
    name="MXFP8_E5M2",  # wider dynamic range, less mantissa precision
    weight_bits=8, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp8_e5m2", act_bits=8, act_dtype_name="fp8_e5m2",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(8, 32, 8, "fp8_e5m2"),
    quantize_dequantize=_make_rtn("fp8_e5m2", 32),
))
register_format(FormatSpec(
    name="MXFP8A16",
    weight_bits=8, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp8_e4m3", act_bits=None,
    family="mx", min_capability_sm=80,  # W8A16 works on Marlin
    autoround_config=lambda: _mx_autoround(8, 32, 16, "fp8_e4m3"),
    quantize_dequantize=_make_rtn("fp8_e4m3", 32),
))

# INT8 per-channel / INT4 per-group
register_format(FormatSpec(
    name="INT8_W8A16",
    weight_bits=8, group_size=0, scale_bits=16, scale_dtype_name="fp16",
    weight_element_dtype="int8",
    family="int", min_capability_sm=70,
    autoround_config=lambda: _int_autoround(8, -1, 16),
    quantize_dequantize=lambda w: _rtn_uniform_int(w, 8, 0),
))
register_format(FormatSpec(
    name="INT4_W4A16_g128",
    weight_bits=4, group_size=128, scale_bits=16, scale_dtype_name="fp16",
    weight_element_dtype="int4",
    family="int", min_capability_sm=70,
    autoround_config=lambda: _int_autoround(4, 128, 16),
    quantize_dequantize=lambda w: _rtn_uniform_int(w, 4, 128),
))

# Passthrough for highest-precision layer when budget is loose
register_format(FormatSpec(
    name="BF16",
    weight_bits=16, group_size=0, scale_bits=0, scale_dtype_name="none",
    weight_element_dtype="bfloat16",
    family="fp", min_capability_sm=75,
    autoround_config=lambda: dict(bits=16, group_size=0,
                                   data_type="float", act_bits=16,
                                   act_data_type="float"),
    quantize_dequantize=lambda w: w.clone(),
))


def list_formats(family: str | None = None) -> list[FormatSpec]:
    if family is None:
        return sorted(REGISTRY.values(), key=lambda s: s.effective_bits)
    return sorted((s for s in REGISTRY.values() if s.family == family),
                  key=lambda s: s.effective_bits)


def get_format(name: str) -> FormatSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown format '{name}'. Available: "
                       f"{sorted(REGISTRY.keys())}")
    return REGISTRY[name]
