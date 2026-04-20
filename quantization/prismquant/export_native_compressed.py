#!/usr/bin/env python3
"""export_native_compressed.py — materialize a PrismQuant recipe as a
standard `compressed-tensors` checkpoint that vLLM serves natively.

This is the unified export path. Decoder layers are streamed from
safetensors one at a time: the model skeleton is built on meta via
`init_empty_weights`, head + embed + norm + lm_head + rotary stay
resident, and each decoder layer flows disk → quantize → emit → unload.
Small models pay the no-op cost of a LayerCache large enough to keep
everything resident; big models (Qwen3.5-122B at 244 GB BF16) fit
through the same path on a 121 GB host.

Reads the per-tensor format assignment produced by `allocator.py`
(layer_config.json) and emits a directory containing:

  - `model-*.safetensors` (sharded), with each Linear / packed-MoE
    tensor written under the standard compressed-tensors schema:
        <name>.weight_packed         (uint8, 4-bit packed for NVFP4)
        <name>.weight_scale          (fp8_e4m3fn for NVFP4 / e8m0 for MXFP8)
        <name>.weight_global_scale   (fp32, NVFP4 only)
        <name>.input_global_scale    (fp32, A4/A8 formats only)
    OR `<name>.weight` (passthrough bf16) for layers in the BF16 bucket.

  - `model.safetensors.index.json` matching the safetensors layout

  - `config.json` carrying a `quantization_config` with
    `format = mixed-precision` and one config_group per nominated
    format. Targets are explicit per-Linear regex anchors so vLLM's
    compressed-tensors dispatcher routes every parameter to the right
    scheme without ambiguity.

  - `mixed_native_manifest.json` summarizing the export (format
    histogram, ignore list, source recipe path) for traceability.

  - tokenizer / config files copied verbatim from the source.

Why this exists separately from llmcompressor's oneshot:
  - llmcompressor's QuantizationModifier matches nn.Linear modules. It
    does not handle 3D packed-expert tensors (Qwen3.5/3.6's
    `gate_up_proj` / `down_proj`), which silently fall back to dense
    bf16 in the standard pipeline.
  - llmcompressor pins transformers <5; transformers v5 is required to
    load Qwen3.6 (`qwen3_5_moe`). The two cannot coexist.

This exporter pins to transformers v5 for model load, uses the
compressed-tensors lib's `pack_fp4_to_uint8` reference (inlined to
avoid the lib's transformers-coupled `__init__`), and writes the
on-disk layout directly. vLLM's existing `compressed_tensors` and
`compressed_tensors_moe_w4a4_nvfp4` schemes load the result without
patches.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors.torch import save_file

from .model_profiles.qwen3_5 import Qwen3_5Profile

# ---------------------------------------------------------------------------
# NVFP4 packing (inlined from compressed-tensors fp4_quantized.py to avoid
# importing the library's __init__ which pulls in transformers internals
# that are not stable across the 4.x → 5.x break).
# ---------------------------------------------------------------------------
FLOAT_TO_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
NVFP4_MAX = 6.0     # max(|FLOAT_TO_E2M1|)
FP8_E4M3_MAX = 448.0  # max representable in torch.float8_e4m3fn

# Back-compat exports for unit tests that validate the Qwen3.5 naming
# and per-expert catch-all contract via the historical helper symbols.
_COMPAT_QWEN_PROFILE = Qwen3_5Profile()
PER_EXPERT_MOE_REGEX = _COMPAT_QWEN_PROFILE.per_expert_moe_regex()


def _to_vllm_internal_name(checkpoint_name: str) -> str:
    """Compatibility helper kept for unit tests.

    The production path is profile-driven via `profile.to_vllm_internal_name`;
    this helper preserves the historical Qwen3.5/3.6 mapping semantics
    without depending on a local vLLM install.
    """
    name = checkpoint_name
    if name.startswith("mtp."):
        return name
    if name == "lm_head":
        return "language_model.lm_head"
    if name.startswith("model.visual."):
        return name[len("model."):]
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model."):]
    if (name.startswith("model.layers.")
            or name.startswith("model.embed_tokens")
            or name.startswith("model.norm")
            or name == "model"):
        return "language_model.model." + name[len("model."):]
    return name


def _nvfp4_codebook(device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(FLOAT_TO_E2M1, device=device, dtype=dtype)


def _round_to_codebook(values_in_grid: torch.Tensor) -> torch.Tensor:
    """Round per-element values (already scaled into the [-6, +6]
    NVFP4 grid) to the nearest codebook entry, using bucketize on the
    sorted absolute codebook. O(N log K) instead of O(N · K).

    Returns a Long tensor of 4-bit indices in [0, 15], where bit 3 is
    the sign bit and bits 0-2 are the abs-codebook index.
    """
    cb = _nvfp4_codebook(values_in_grid.device, dtype=torch.float32)
    abs_x = values_in_grid.abs().contiguous()
    idx = torch.bucketize(abs_x, cb)        # insertion: cb[idx-1] <= x < cb[idx]
    idx_lo = (idx - 1).clamp_min(0).clamp_max(cb.numel() - 1)
    idx_hi = idx.clamp_max(cb.numel() - 1)
    lo_v = cb[idx_lo]
    hi_v = cb[idx_hi]
    pick_hi = (hi_v - abs_x).abs() < (abs_x - lo_v).abs()
    abs_idx = torch.where(pick_hi, idx_hi, idx_lo).long()
    sign_bit = torch.signbit(values_in_grid).to(torch.long) << 3
    return abs_idx + sign_bit                # [..., shape]; values 0-15


def pack_fp4_indices(fp4_indices: torch.Tensor, last_dim: int) -> torch.Tensor:
    """Pack a tensor of 4-bit indices (final dim must be even) into
    uint8, two indices per byte. Preserves leading dimensions.
    """
    if last_dim % 2 != 0:
        raise ValueError("nvfp4 pack requires an even last dim")
    pairs = fp4_indices.reshape(*fp4_indices.shape[:-1], last_dim // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


DEFAULT_INPUT_GLOBAL_SCALE = 1.0  # placeholder; overridden by calibration


def compute_nvfp4_global_real(weight: torch.Tensor, group_size: int = 16
                              ) -> torch.Tensor:
    """Return the per-tensor `global_real` that NVFP4 packing would
    pick for `weight` alone. Useful for fused-sibling pre-pass: caller
    takes the max across siblings and passes the joint value back into
    `quantize_dequantize_nvfp4(global_real_override=...)`."""
    rows, cols = weight.shape
    grouped = weight.float().reshape(rows, cols // group_size, group_size)
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    s_g_real = max_abs / NVFP4_MAX
    return (s_g_real.amax() / FP8_E4M3_MAX).clamp_min(1e-12)


def quantize_dequantize_nvfp4(
    weight: torch.Tensor, group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply NVFP4 RTN to a 2D `[rows, cols]` weight and return the
    on-disk triple `(weight_packed, weight_scale, weight_global_scale)`
    in the **compressed-tensors NVFP4 convention**:

      - per-group dequant scale  s_g_real = max-abs(group) / NVFP4_MAX
      - per-tensor outer scale   global   = max(s_g_real) / FP8_E4M3_MAX
        (so the fp8-stored per-group scale stays inside [0, 448])
      - on-disk weight_scale (fp8) = s_g_real / global  ∈ [0, 448]
      - on-disk weight_global_scale = 1 / global  (DIVISOR)
        vLLM inverts on load: `layer.weight_global_scale = 1/loaded`
        → recovers `global` and applies it as the per-tensor multiplier
        in the NVFP4 GEMM.

    Dequant in the kernel: `weight ≈ codebook[index] · weight_scale_fp8 · global`

    `global_real_override` lets a caller force a particular per-tensor
    scale — used for fused siblings (q/k/v, gate/up) that vLLM expects
    to share one global_scale slot. Pass the max across the sibling
    group's natural global_real values.
    """
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {cols}")
    n_groups = cols // group_size
    grouped = weight.float().reshape(rows, n_groups, group_size)
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)               # [rows, n_groups]
    s_g_real = max_abs / NVFP4_MAX                                       # the actual per-group scale
    if global_real_override is not None:
        global_real = global_real_override.to(weight.device).clamp_min(1e-12)
    else:
        global_real = (s_g_real.amax() / FP8_E4M3_MAX).clamp_min(1e-12)  # scalar
    fp8_scale_real = (s_g_real / global_real).clamp(0, FP8_E4M3_MAX)     # [rows, n_groups], in [0, 448]
    # Per-element grid mapping: weight / (fp8_scale_real * global_real) = weight / s_g_real
    in_grid = grouped / s_g_real.unsqueeze(-1).clamp_min(1e-12)          # [rows, n_groups, group_size]
    fp4_idx = _round_to_codebook(in_grid).reshape(rows, cols)
    weight_packed = pack_fp4_indices(fp4_idx, cols)
    return (
        weight_packed,
        fp8_scale_real.to(torch.float8_e4m3fn),
        (1.0 / global_real).to(torch.float32).reshape(1),  # divisor convention
    )


def quantize_dequantize_nvfp4_packed(
    packed: torch.Tensor, group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-expert NVFP4 packing for a 3D `[E, M, N]` packed tensor.
    Each expert gets its own `global_real` (so the weight_global_scale
    output has shape `[E]`); the on-disk values are divisors (1/scale)
    matching the compressed-tensors convention.
    """
    E, M, N = packed.shape
    if N % group_size != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {N}")
    g = N // group_size
    grouped = packed.float().reshape(E, M, g, group_size)
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    s_g_real = max_abs / NVFP4_MAX                                          # [E, M, g]
    global_real = (s_g_real.reshape(E, -1).amax(dim=-1) / FP8_E4M3_MAX).clamp_min(1e-12)  # [E]
    fp8_scale_real = (s_g_real / global_real.view(E, 1, 1)).clamp(0, FP8_E4M3_MAX)
    in_grid = grouped / s_g_real.unsqueeze(-1).clamp_min(1e-12)
    fp4_idx = _round_to_codebook(in_grid).reshape(E, M, N)
    weight_packed = pack_fp4_indices(fp4_idx, N)
    return (
        weight_packed,
        fp8_scale_real.to(torch.float8_e4m3fn),
        (1.0 / global_real).to(torch.float32),
    )


# ---------------------------------------------------------------------------
# MXFP8 packing (E4M3 element format, E8M0 per-group scale).
# ---------------------------------------------------------------------------
MXFP8_E4M3_MAX = 448.0   # max representable in fp8_e4m3fn


def _mxfp8_quantize_grouped(grouped: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute MXFP8 quantized values + E8M0 scale for an arbitrary
    rank-N tensor whose LAST dim is the per-group axis (size group_size).

    Returns:
      - quant_fp8: same shape as `grouped`, dtype torch.float8_e4m3fn
      - e8m0_uint8: same shape minus the last dim, uint8 (E8M0)

    Care: with E8M0 round-to-nearest the per-group scale can be
    slightly smaller than max-abs/MXFP8_E4M3_MAX, which would push
    quant_grid past 448 (fp8_e4m3fn max) and produce NaN on cast.
    We use ceil() on log2 to guarantee s_g >= max-abs/MXFP8_E4M3_MAX,
    keeping all quant_grid values inside the representable range.
    """
    s_g_real = grouped.abs().amax(dim=-1).clamp_min(2.0 ** -127) / MXFP8_E4M3_MAX
    log2_s = torch.log2(s_g_real)
    e8m0 = torch.ceil(log2_s).clamp(-127, 127)
    s_g = torch.pow(2.0, e8m0)
    quant_grid = grouped / s_g.unsqueeze(-1).clamp_min(2.0 ** -127)
    # Defensive clamp against numerical edge cases at the saturation boundary.
    quant_grid = quant_grid.clamp(-MXFP8_E4M3_MAX, MXFP8_E4M3_MAX)
    quant_fp8 = quant_grid.to(torch.float8_e4m3fn)
    e8m0_uint8 = (e8m0 + 127).to(torch.int32).clamp(0, 255).to(torch.uint8)
    return quant_fp8, e8m0_uint8


def quantize_dequantize_mxfp8(weight: torch.Tensor, group_size: int = 32
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MXFP8 (E4M3) RTN with E8M0 per-group scale to a 2D weight.

    On-disk schema (compressed-tensors `mxfp8-quantized` format):
      - weight_packed: torch.float8_e4m3fn, same shape as weight
      - weight_scale:  uint8 E8M0, shape (rows, cols // group_size)
    """
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"MXFP8 group_size={group_size} ∤ {cols}")
    grouped = weight.float().reshape(rows, cols // group_size, group_size)
    quant_fp8, e8m0_uint8 = _mxfp8_quantize_grouped(grouped)
    return quant_fp8.reshape(rows, cols), e8m0_uint8


def quantize_dequantize_mxfp8_packed(packed: torch.Tensor, group_size: int = 32
                                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MXFP8 RTN to a 3D packed-experts tensor `[E, M, N]`.

    Returns:
      - weight_packed: float8_e4m3fn `[E, M, N]`
      - weight_scale:  uint8 E8M0   `[E, M, N//group_size]`
    """
    E, M, N = packed.shape
    if N % group_size != 0:
        raise ValueError(f"MXFP8 group_size={group_size} ∤ {N}")
    grouped = packed.float().reshape(E, M, N // group_size, group_size)
    quant_fp8, e8m0_uint8 = _mxfp8_quantize_grouped(grouped)
    return quant_fp8.reshape(E, M, N), e8m0_uint8


def quantize_dequantize_fp8_dynamic(weight: torch.Tensor
                                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 W8A8 dynamic per-channel weight quantization.

    Matches vLLM's CompressedTensorsW8A8Fp8 expectation:
      - weight: torch.float8_e4m3fn, shape `[out, in]`
      - weight_scale: torch.float32, shape `[out, 1]` (per-channel)

    Per-channel scale = max-abs(row) / fp8_max. Dynamic-token activation
    quantization is handled at runtime by vLLM (no on-disk activation
    scale needed).
    """
    rows, cols = weight.shape
    w_f = weight.float()
    s = w_f.abs().amax(dim=-1, keepdim=True).clamp_min(2.0 ** -127) / MXFP8_E4M3_MAX
    quant = (w_f / s).clamp(-MXFP8_E4M3_MAX, MXFP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return quant, s.to(torch.float32)


def quantize_dequantize_fp8_dynamic_packed(packed: torch.Tensor
                                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert FP8 W8A8 dynamic per-channel for `[E, M, N]` packed.

    Returns weight `[E, M, N]` fp8 and scale `[E, M, 1]` fp32.
    """
    E, M, N = packed.shape
    p_f = packed.float()
    s = p_f.abs().amax(dim=-1, keepdim=True).clamp_min(2.0 ** -127) / MXFP8_E4M3_MAX
    quant = (p_f / s).clamp(-MXFP8_E4M3_MAX, MXFP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return quant, s.to(torch.float32)


# ---------------------------------------------------------------------------
# Recipe parsing — mirrors export_mixed_native.canonicalize_format but
# accepts the allocator's exact AutoRound-shaped output.
# ---------------------------------------------------------------------------
def canonicalize_format(scheme_dict: dict | str | int) -> str:
    """Map a layer_config entry to one of {NVFP4, MXFP8, BF16}.

    Accepts the dicts emitted by allocator.py via FormatSpec.autoround_config()
    (data_type=nv_fp/mx_fp/float, bits=4/8/16) plus a few tolerant
    string aliases.
    """
    if isinstance(scheme_dict, dict):
        dt = scheme_dict.get("data_type")
        bits = int(scheme_dict.get("bits", 0))
        if dt == "nv_fp" and bits == 4:
            return "NVFP4"
        if dt == "mx_fp" and bits == 4:
            return "NVFP4"  # 4-bit floor — only NVFP4 has vLLM serving today
        if dt == "mx_fp" and bits == 8:
            return "MXFP8"
        if dt in ("float", "bfloat16") and bits in (16, 0):
            return "BF16"
        if dt == "fp8_e4m3" and bits == 8:
            return "MXFP8"  # collapse plain FP8 onto the MX bucket for now
        raise ValueError(f"unsupported scheme: {scheme_dict!r}")
    if isinstance(scheme_dict, str):
        s = scheme_dict.lower()
        if s in ("nvfp4", "fp4", "4"):
            return "NVFP4"
        if s in ("mxfp8", "fp8", "8"):
            return "MXFP8"
        if s in ("bf16", "bfloat16", "16"):
            return "BF16"
    if isinstance(scheme_dict, int):
        if scheme_dict <= 4:
            return "NVFP4"
        if scheme_dict <= 8:
            return "MXFP8"
        return "BF16"
    raise ValueError(f"unrecognized layer-config entry: {scheme_dict!r}")


def _strip_weight(name: str) -> str:
    return name[:-7] if name.endswith(".weight") else name


def _explicit_regex(name: str) -> str:
    """Anchor a Linear name as a compressed-tensors regex target."""
    return f"re:^{name.replace('.', '[.]')}$"


# ---------------------------------------------------------------------------
# Module / parameter discovery — mirrors what install_packed_expert_hooks
# detects, so the export sees the same units as the probe.
# ---------------------------------------------------------------------------
_PACKED_EXPERT_PARAM_NAMES = {
    "gate_up_proj", "down_proj", "w1", "w2", "w3",
    "gate_proj", "up_proj",
}


def _is_packed_experts_module(module: nn.Module) -> bool:
    cls_name = type(module).__name__.lower()
    if "expert" not in cls_name:
        return False
    for n, p in module.named_parameters(recurse=False):
        if (isinstance(p, nn.Parameter)
                and p.dim() == 3
                and n in _PACKED_EXPERT_PARAM_NAMES):
            return True
    return False


def _packed_experts_param_names(module: nn.Module) -> list[str]:
    return sorted(
        n for n, p in module.named_parameters(recurse=False)
        if (isinstance(p, nn.Parameter)
            and p.dim() == 3
            and n in _PACKED_EXPERT_PARAM_NAMES)
    )


# ---------------------------------------------------------------------------
# Fused-sibling joint global_scale (for dense Linears)
# ---------------------------------------------------------------------------
# vLLM's compressed_tensors_w4a4_nvfp4.process_weights_after_loading warns
# (and reduces accuracy) when q/k/v or gate/up have different
# weight_global_scale. We compute the max over each fused group's natural
# global_scale and force every sibling to use it.
#
# Patterns mirror vLLM's `packed_modules_mapping` for qwen3_5; if a new
# model family is added, mirror its packed_modules_mapping here.
_FUSED_DENSE_PATTERNS = [
    (re.compile(r"^(?P<pre>.+)\.self_attn\.(?P<sib>q_proj|k_proj|v_proj)$"),
     ("q_proj", "k_proj", "v_proj")),
    (re.compile(r"^(?P<pre>.+)\.mlp\.(?P<sib>gate_proj|up_proj)$"),
     ("gate_proj", "up_proj")),
    (re.compile(r"^(?P<pre>.+)\.mlp\.shared_expert\.(?P<sib>gate_proj|up_proj)$"),
     ("gate_proj", "up_proj")),
    (re.compile(r"^(?P<pre>.+)\.linear_attn\.(?P<sib>in_proj_qkv|in_proj_z)$"),
     ("in_proj_qkv", "in_proj_z")),
    (re.compile(r"^(?P<pre>.+)\.linear_attn\.(?P<sib>in_proj_a|in_proj_b)$"),
     ("in_proj_a", "in_proj_b")),
]


def _fused_dense_group(name: str) -> tuple[str, tuple[str, ...]] | None:
    """Return (group_key, sibling_member_names) if `name` is part of a
    known fused dense Linear group; else None. group_key is the parent
    prefix used to bucket siblings together."""
    for pat, members in _FUSED_DENSE_PATTERNS:
        m = pat.match(name)
        if m:
            return (m.group("pre"), members)
    return None


def _compute_nvfp4_joint_global(
    model: nn.Module, assignment: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Pre-pass over the model: for each fused-sibling group whose
    members are all assigned to NVFP4, compute the joint global_real
    (max across siblings). Return a dict mapping each sibling's qname
    to the shared global_real tensor."""
    # Bucket siblings by (parent_prefix, kind). Missing siblings are
    # OK — vLLM's loader handles partial fusion fine.
    groups: dict[tuple[str, tuple[str, ...]], list[tuple[str, nn.Linear]]] = {}
    for qname, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if assignment.get(qname) != "NVFP4":
            continue
        g = _fused_dense_group(qname)
        if g is None:
            continue
        groups.setdefault(g, []).append((qname, mod))

    out: dict[str, torch.Tensor] = {}
    for (_pre, _members), siblings in groups.items():
        # Need every sibling to also be NVFP4 — otherwise vLLM allocates
        # the fused tensor under a different scheme and our joint scale
        # wouldn't apply consistently. The allocator's promote_fused
        # already enforces this; here we just verify and skip on partial
        # consistency (defensive — a mixed-format fused group is a bug
        # upstream of the export and would fail the load anyway).
        candidates = [
            compute_nvfp4_global_real(mod.weight.detach().float())
            for _, mod in siblings
        ]
        joint = torch.stack(candidates).max()
        for qname, _ in siblings:
            out[qname] = joint
    return out


# ---------------------------------------------------------------------------
# Quantization pipeline
# ---------------------------------------------------------------------------
def _quantize_2d(
    weight: torch.Tensor, fmt: str,
    nvfp4_global_real_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compress a 2D Linear weight under format `fmt`.

    Returns the dict of on-disk tensors keyed by the suffix
    (`weight_packed`, `weight_scale`, `weight_global_scale`, ...).

    `nvfp4_global_real_override`: when this Linear is one shard of a
    fused parameter (q/k/v/o, gate/up), pass the joint per-tensor
    scale shared across all siblings. vLLM warns when sibling scales
    differ and reports degraded accuracy; sharing avoids both.

    `fmt = MXFP8` emits real MXFP8 tensors: fp8_e4m3fn weights plus
    E8M0 uint8 per-group scales (group_size=32).
    """
    if fmt == "NVFP4":
        wp, ws, wg = quantize_dequantize_nvfp4(
            weight, group_size=16,
            global_real_override=nvfp4_global_real_override,
        )
        return {
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg,
            # Required by vLLM's CompressedTensorsW4A4Nvfp4 process; see
            # compressed_tensors_w4a4_nvfp4.py:115. Without it vLLM
            # initializes input_global_scale to zeros and computes
            # 1/zero on activation quant → degenerate output. We supply
            # a sane default; calibrated values can be merged in later.
            "input_global_scale": torch.tensor(
                [DEFAULT_INPUT_GLOBAL_SCALE], dtype=torch.float32,
            ),
        }
    if fmt == "MXFP8":
        w, ws = quantize_dequantize_mxfp8(weight, group_size=32)
        return {"weight": w, "weight_scale": ws}
    if fmt == "BF16":
        return {"weight": weight.to(torch.bfloat16)}
    raise ValueError(f"unsupported format: {fmt}")


def _quantize_3d_packed(packed: torch.Tensor, fmt: str) -> dict[str, torch.Tensor]:
    """Compress a 3D packed-expert tensor `[E, M, N]` as a single
    batched op (per-expert independent scales).

    Returns tensors with leading expert dim preserved, matching what
    vLLM's `compressed_tensors_moe_w4a4_nvfp4` allocates internally
    (uint8 packed weights, fp8/uint8 per-group scales, per-expert
    global scales for NVFP4).
    """
    if fmt == "BF16":
        return {"weight": packed.to(torch.bfloat16)}
    if fmt == "NVFP4":
        wp, ws, wg = quantize_dequantize_nvfp4_packed(packed, group_size=16)
        return {
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg,
        }
    if fmt == "MXFP8":
        w, ws = quantize_dequantize_mxfp8_packed(packed, group_size=32)
        return {"weight": w, "weight_scale": ws}
    raise ValueError(f"unsupported format for packed-MoE: {fmt}")


# ---------------------------------------------------------------------------
# Fused-sibling joint NVFP4 scale (per-layer scope, used by the streaming
# materializer below). The whole-model variant `_compute_nvfp4_joint_global`
# lives above and is kept for the MTP path + unit tests.
# ---------------------------------------------------------------------------
_FUSED_SIBLINGS = {
    "q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv",
    "gate_proj": "gate_up", "up_proj": "gate_up",
    # Qwen3.5/3.6 DeltaNet linear-attention pairs. vLLM fuses
    # `in_proj_qkv + in_proj_z → in_proj_qkvz` and
    # `in_proj_b + in_proj_a → in_proj_ba` at load time; the fused
    # packed Linear needs ONE shared NVFP4 `weight_global_scale`.
    # Omitting these triggers vLLM's
    # `compressed_tensors_w4a4_nvfp4.py:97` warning about reduced
    # accuracy from mismatched parallel-layer scales.
    "in_proj_qkv": "qkvz", "in_proj_z": "qkvz",
    "in_proj_b": "ba", "in_proj_a": "ba",
}


def _compute_layer_joint_nvfp4(layer_mod: nn.Module,
                               layer_qname: str,
                               assignment: dict[str, str],
                               profile,
                               ) -> dict[str, torch.Tensor]:
    """Return {recipe_key -> joint global scale} for NVFP4 fused-sibling
    groups inside this decoder layer. Only keys assigned NVFP4 get an
    override entry; the rest compute per-Linear scales at quantize time.

    Semantically equivalent to a scoped `_compute_nvfp4_joint_global`
    across just this layer's modules."""
    groups: dict[tuple[str, str], list[tuple[str, nn.Linear]]] = defaultdict(list)
    for sub_name, mod in layer_mod.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        last = sub_name.rsplit(".", 1)[-1]
        fam = _FUSED_SIBLINGS.get(last)
        if fam is None:
            continue
        parent = sub_name.rsplit(".", 1)[0] if "." in sub_name else ""
        groups[(parent, fam)].append((sub_name, mod))

    out: dict[str, torch.Tensor] = {}
    for (_, _), members in groups.items():
        fqn_fmt = []
        for sub_name, mod in members:
            full = f"{layer_qname}.{sub_name}" if sub_name else layer_qname
            recipe_key = profile.live_to_recipe_name(full)
            fmt = assignment.get(recipe_key)
            fqn_fmt.append((full, recipe_key, fmt, mod))
        fmts = {f for _, _, f, _ in fqn_fmt}
        if fmts != {"NVFP4"}:
            continue
        candidates = []
        for _, _, _, mod in fqn_fmt:
            w = mod.weight.detach().float()
            candidates.append(compute_nvfp4_global_real(w, group_size=16))
        joint = torch.stack(candidates).max()
        for full, recipe_key, _, _ in fqn_fmt:
            out[recipe_key] = joint
    return out


def _init_rotary_inplace(base_model: nn.Module, device: torch.device,
                         dtype: torch.dtype) -> None:
    """After init_empty_weights, rotary modules exist but their
    `inv_freq` buffers are on meta. Re-run the module's own rope init
    (which is deterministic from config) so `inv_freq` lives on the
    exec device with correct values — matching what `from_pretrained`
    would have produced."""
    from .layer_streaming import _get_rotary
    rotary = _get_rotary(base_model)
    if rotary is None:
        return
    cfg = getattr(rotary, "config", None)
    if cfg is None:
        return
    try:
        rope_init_fn = rotary.compute_default_rope_parameters
    except AttributeError:
        return
    inv_freq, attention_scaling = rope_init_fn(cfg, device)
    rotary.register_buffer("inv_freq", inv_freq.to(dtype=torch.float32,
                                                   device=device),
                           persistent=False)
    if hasattr(rotary, "original_inv_freq"):
        rotary.register_buffer(
            "original_inv_freq",
            inv_freq.to(dtype=torch.float32, device=device).clone(),
            persistent=False)
    rotary.attention_scaling = attention_scaling


def materialize_tensors_streaming(
    model_path: str,
    assignment: dict[str, str],
    *,
    profile,
    bf16_passthrough: set[str],
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    offload_folder: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Stream decoder layers through quantize → emit → unload. Never
    holds the full model in memory. Small models still exercise this
    path — the LayerCache just keeps everything resident, so load/
    unload degenerates to a no-op.

    Output: `(out_tensors, hist)` matching the shape the monolithic
    materialize used to return, ready for `write_sharded_safetensors`."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from .layer_streaming import (
        _build_install_resolver,
        _build_weight_map,
        _fast_install,
        _get_layer_list,
        _head_prefixes,
        _materialize,
        _read_layer_to_device,
        _resolve_base_prefix,
        _unload,
    )
    from .sensitivity_probe import stage_text_only

    # ----- 1. Meta skeleton + manual head materialization -----
    # Pure `init_empty_weights` path — avoids accelerate's
    # `from_pretrained` which would write ~244 GB of offload files to
    # disk on Qwen3.5-122B before we ever read them. Instead we:
    #   (a) build the full skeleton on meta (0 bytes),
    #   (b) read head/embed/norm/lm_head tensors directly from the
    #       source safetensors and install on the exec device,
    #   (c) re-run rotary's init_fn to populate `inv_freq` (not in
    #       state_dict — computed from config),
    #   (d) leave decoder layers on meta until the per-layer loop
    #       streams them in.
    staged = stage_text_only(model_path)
    config = AutoConfig.from_pretrained(staged, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_model, layers = _get_layer_list(model)
    base_prefix = _resolve_base_prefix(model, base_model)
    num_layers = len(layers)
    layers_prefix = f"{base_prefix}.layers." if base_prefix else "layers."

    weight_shard, weight_ckpt = _build_weight_map(model_path)

    # Materialize head (embed + norm + lm_head). These are in the
    # safetensors and get populated via `set_module_tensor_to_device`.
    print(f"[export-stream] base_prefix={base_prefix!r}  layers={num_layers}",
          flush=True)
    t0 = time.time()
    head_pfxs = _head_prefixes(None, base_prefix)
    loaded_n = _materialize(model, head_pfxs, weight_shard, weight_ckpt,
                            device, dtype)

    # Rotary's `inv_freq` isn't in the state_dict — compute from config.
    _init_rotary_inplace(base_model, device, dtype)
    print(f"[export-stream] head materialized ({loaded_n} tensors, rotary "
          f"re-init) in {time.time()-t0:.1f}s", flush=True)

    out: dict[str, torch.Tensor] = {}
    hist: Counter = Counter()
    unmapped_keys: list[str] = []

    # ----- 2. Head / embed / norm / lm_head / rotary passthrough -----
    # These are resident on `device` already. Emit as BF16 passthrough
    # UNLESS `lm_head` (or similar) is explicitly in the assignment.
    t_head = time.time()

    def _emit_head_param(full_qname: str, param: nn.Parameter):
        recipe_key = profile.live_to_recipe_name(full_qname)
        fmt = assignment.get(recipe_key)
        if fmt is not None and fmt != "BF16":
            joint = None
            compressed = _quantize_2d(
                param.detach().float(), fmt,
                nvfp4_global_real_override=joint,
            )
            for suffix, t in compressed.items():
                base_name = (full_qname[:-len(".weight")]
                             if full_qname.endswith(".weight")
                             else full_qname)
                out_key = (base_name
                           if suffix == "weight"
                           else f"{base_name}.{suffix}")
                out[out_key] = t.cpu()
            hist[("head", fmt)] += 1
        else:
            out[full_qname] = param.detach().to(torch.bfloat16).cpu()
            hist[("head_passthrough", "BF16")] += 1

    for name, p in model.named_parameters():
        if p.is_meta:
            continue  # only head/embed/norm/lm_head resident here
        _emit_head_param(name, p)

    for mod_name, mod in model.named_modules():
        non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
        for buf_name, buf in mod.named_buffers(recurse=False):
            if buf_name in non_persistent:
                continue
            if buf.is_meta:
                continue
            full = f"{mod_name}.{buf_name}" if mod_name else buf_name
            if full in out:
                continue
            out[full] = buf.detach().to(torch.bfloat16).cpu()
            hist[("head_buffer", "BF16")] += 1
    print(f"[export-stream] head+embed+norm+lm_head passthrough: "
          f"{time.time()-t_head:.1f}s  keys={len(out)}", flush=True)

    # ----- 3. Per-layer streaming quantize loop -----
    t_layers = time.time()
    for L in range(num_layers):
        layer_t0 = time.time()
        layer_qname = f"{layers_prefix}{L}".rstrip(".")
        if layer_qname.endswith("."):
            layer_qname = layer_qname[:-1]

        # 3a. Load layer from safetensors (direct to device).
        load_t0 = time.time()
        tensors = _read_layer_to_device(
            f"{layers_prefix}{L}.", weight_shard, weight_ckpt, dtype, device)
        resolver = _build_install_resolver(model, layer_qname)
        _fast_install(resolver, tensors, device, model=model)
        load_s = time.time() - load_t0

        layer_mod = model.get_submodule(layer_qname)

        # 3b. Joint NVFP4 scales across fused siblings in this layer.
        joint_globals = _compute_layer_joint_nvfp4(
            layer_mod, layer_qname, assignment, profile)

        # 3c. Emit Linears.
        covered: set[str] = set()
        linear_count = 0
        for sub_name, mod in layer_mod.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            linear_count += 1
            full = f"{layer_qname}.{sub_name}"
            recipe_key = profile.live_to_recipe_name(full)
            fmt = assignment.get(recipe_key)
            if fmt is None:
                # No assignment → BF16 passthrough.
                if not mod.weight.is_meta:
                    out[f"{full}.weight"] = mod.weight.detach().to(torch.bfloat16).cpu()
                    if mod.bias is not None and not mod.bias.is_meta:
                        out[f"{full}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
                    hist[("linear", "BF16")] += 1
                    covered.add(full)
                continue

            if fmt == "BF16" or recipe_key in bf16_passthrough:
                out[f"{full}.weight"] = mod.weight.detach().to(torch.bfloat16).cpu()
                if mod.bias is not None:
                    out[f"{full}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
                hist[("linear", "BF16")] += 1
                covered.add(full)
                continue

            override = joint_globals.get(recipe_key) if fmt == "NVFP4" else None
            compressed = _quantize_2d(
                mod.weight.detach().float(), fmt,
                nvfp4_global_real_override=override,
            )
            for suffix, t in compressed.items():
                out[f"{full}.{suffix}"] = t.cpu()
            if mod.bias is not None:
                out[f"{full}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
            hist[("linear", fmt)] += 1
            covered.add(full)

        # 3d. Emit packed MoE experts, scoped to this layer.
        packed_count = 0
        for sub_name, mod in layer_mod.named_modules():
            if not _is_packed_experts_module(mod):
                continue
            packed_count += 1
            for pn in _packed_experts_param_names(mod):
                experts_qname = (f"{layer_qname}.{sub_name}"
                                 if sub_name else layer_qname)
                full = f"{experts_qname}.{pn}"
                recipe_key = profile.live_to_recipe_name(full)
                fmt = assignment.get(recipe_key)
                if fmt is None:
                    unmapped_keys.append(full)
                    continue
                packed_param = getattr(mod, pn).detach().float()
                E, M, N = packed_param.shape
                if pn == "gate_up_proj":
                    half = M // 2
                    proj_split = [
                        ("gate_proj", packed_param[:, :half, :]),
                        ("up_proj",   packed_param[:, half:, :]),
                    ]
                else:
                    proj_split = [(pn, packed_param)]

                is_bf16 = fmt == "BF16" or full in bf16_passthrough
                disk_qname = profile.on_disk_expert_qname(experts_qname)
                should_split = profile.split_packed_experts_for_format(fmt)

                if not should_split:
                    out[f"{disk_qname}.{pn}"] = packed_param.to(torch.bfloat16).cpu()
                    covered.add(full)
                    hist[("packed_moe", "BF16" if is_bf16 else fmt)] += 1
                    del packed_param
                    continue

                # Per-expert joint global scale when NVFP4 splits gate+up.
                per_expert_joint: list[torch.Tensor | None] = [None] * E
                if fmt == "NVFP4" and len(proj_split) > 1:
                    for e in range(E):
                        cands = [
                            compute_nvfp4_global_real(sp[e].float(),
                                                      group_size=16)
                            for _, sp in proj_split
                        ]
                        per_expert_joint[e] = torch.stack(cands).max()

                for proj_name, sub_packed in proj_split:
                    E_p, Mp, Np = sub_packed.shape
                    for e in range(E_p):
                        expert_2d = sub_packed[e]
                        base = f"{disk_qname}.{e}.{proj_name}"
                        if is_bf16:
                            out[f"{base}.weight"] = expert_2d.to(torch.bfloat16).cpu()
                        else:
                            compressed = _quantize_2d(
                                expert_2d, fmt,
                                nvfp4_global_real_override=per_expert_joint[e],
                            )
                            for suffix, t in compressed.items():
                                key = (base
                                       if suffix == "weight"
                                       else f"{base}.{suffix}")
                                out[key] = t.cpu()
                covered.add(full)
                hist[("packed_moe_per_expert", "BF16" if is_bf16 else fmt)] += 1
                del packed_param, proj_split

        # 3e. Remaining layer-scoped params (norms, conv1d, biases on
        # passthrough-only modules) and persistent buffers.
        for sub_name, param in layer_mod.named_parameters():
            full = f"{layer_qname}.{sub_name}"
            if full in out:
                continue
            if any(full.startswith(c + ".") or full == c for c in covered):
                continue
            if param.is_meta:
                continue
            out[full] = param.detach().to(torch.bfloat16).cpu()
            hist[("layer_passthrough", "BF16")] += 1
        for mod_name, mod in layer_mod.named_modules():
            non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
            for buf_name, buf in mod.named_buffers(recurse=False):
                if buf_name in non_persistent:
                    continue
                full_modpath = (f"{layer_qname}.{mod_name}"
                                if mod_name else layer_qname)
                full = f"{full_modpath}.{buf_name}"
                if full in out or buf.is_meta:
                    continue
                out[full] = buf.detach().to(torch.bfloat16).cpu()
                hist[("layer_buffer", "BF16")] += 1

        # 3f. Unload.
        _unload(model, [f"{layers_prefix}{L}."])
        del tensors, resolver, joint_globals
        # Aggressive GPU cleanup — we've already `.cpu()`'d every
        # quantized output into `out`, so the per-layer GPU working
        # set (fp32 weight copies, grouped/packed intermediates) can
        # be released immediately. Keeps per-layer peak bounded.
        if device.type == "cuda":
            torch.cuda.synchronize()  # ensure outputs are CPU-resident
            torch.cuda.empty_cache()
        if L % 4 == 0:
            gc.collect()
        if L % 4 == 0 or L == num_layers - 1:
            elapsed = time.time() - layer_t0
            print(f"[export-stream] layer {L:02d}  linears={linear_count} "
                  f"packed={packed_count}  load={load_s:.2f}s  "
                  f"total={elapsed:.2f}s  out_keys={len(out)}", flush=True)

    print(f"[export-stream] layer sweep: {time.time()-t_layers:.1f}s",
          flush=True)

    if unmapped_keys:
        print(f"[export-stream] WARN {len(unmapped_keys)} unmapped assignment "
              f"keys — first 5: {unmapped_keys[:5]}", flush=True)

    return out, dict(hist)


def _materialize_tensors_inmemory(
    model: nn.Module,
    assignment: dict[str, str],
    *,
    bf16_passthrough: set[str],
    profile: "ModelProfile | None" = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Whole-model quantizer used for small auxiliary modules (notably the
    MTP wrapper) that fit in RAM. The main decoder export path uses the
    streaming materializer above; this helper exists because MTP is
    built standalone from safetensors and its root module is orders of
    magnitude smaller than the decoder body."""
    from .model_profiles import DefaultProfile
    profile = profile or DefaultProfile()
    remap = profile.live_to_recipe_name

    out: dict[str, torch.Tensor] = {}
    hist = Counter()
    covered: set[str] = set()

    # Pre-pass: joint NVFP4 global_scale per fused-sibling group so
    # q/k/v (or gate/up, etc.) share one weight_global_scale slot.
    nvfp4_joint_global = _compute_nvfp4_joint_global(model, assignment)

    for qname, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        fmt_key = remap(qname)
        fmt = assignment.get(fmt_key)
        if fmt is None:
            continue
        if fmt == "BF16" or fmt_key in bf16_passthrough:
            out[f"{qname}.weight"] = mod.weight.detach().to(torch.bfloat16).cpu()
            if mod.bias is not None:
                out[f"{qname}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
            covered.add(qname)
            hist[("linear", "BF16")] += 1
            continue
        joint = nvfp4_joint_global.get(fmt_key) if fmt == "NVFP4" else None
        compressed = _quantize_2d(
            mod.weight.detach().float(), fmt,
            nvfp4_global_real_override=joint,
        )
        for suffix, tensor in compressed.items():
            out[f"{qname}.{suffix}"] = tensor.cpu()
        if mod.bias is not None:
            out[f"{qname}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
        covered.add(qname)
        hist[("linear", fmt)] += 1

    for qname, mod in model.named_modules():
        if not _is_packed_experts_module(mod):
            continue
        for pn in _packed_experts_param_names(mod):
            full_name = f"{qname}.{pn}" if qname else pn
            recipe_key = remap(full_name)
            fmt = assignment.get(recipe_key)
            if fmt is None:
                continue
            packed_param = getattr(mod, pn).detach().float()
            E, M, N = packed_param.shape
            if pn == "gate_up_proj":
                half = M // 2
                proj_split = [
                    ("gate_proj", packed_param[:, :half, :]),
                    ("up_proj",   packed_param[:, half:, :]),
                ]
            elif pn in ("down_proj", "w1", "w2", "w3", "gate_proj", "up_proj"):
                proj_split = [(pn, packed_param)]
            else:
                proj_split = [(pn, packed_param)]

            is_bf16 = fmt == "BF16" or full_name in bf16_passthrough
            disk_qname = profile.on_disk_expert_qname(qname)
            should_split = profile.split_packed_experts_for_format(fmt)

            if not should_split:
                out[f"{disk_qname}.{pn}"] = packed_param.to(torch.bfloat16).cpu()
                covered.add(full_name)
                hist[("packed_moe", "BF16" if is_bf16 else fmt)] += 1
                continue

            per_expert_joint: list[torch.Tensor | None] = [None] * E
            if fmt == "NVFP4" and len(proj_split) > 1:
                for e in range(E):
                    candidates = [
                        compute_nvfp4_global_real(sub_packed[e].float(),
                                                  group_size=16)
                        for _, sub_packed in proj_split
                    ]
                    per_expert_joint[e] = torch.stack(candidates).max()

            for proj_name, sub_packed in proj_split:
                E_p, Mp, Np = sub_packed.shape
                for e in range(E_p):
                    expert_2d = sub_packed[e]
                    base = f"{disk_qname}.{e}.{proj_name}"
                    if is_bf16:
                        out[f"{base}.weight"] = expert_2d.to(torch.bfloat16).cpu()
                    else:
                        compressed = _quantize_2d(
                            expert_2d, fmt,
                            nvfp4_global_real_override=per_expert_joint[e],
                        )
                        for suffix, tensor in compressed.items():
                            key = base if suffix == "weight" else f"{base}.{suffix}"
                            out[key] = tensor.cpu()
            covered.add(full_name)
            hist[("packed_moe_per_expert", "BF16" if is_bf16 else fmt)] += 1

    for name, p in model.named_parameters():
        if any(name.startswith(c + ".") or name == c for c in covered):
            continue
        if name in out:
            continue
        out[name] = p.detach().to(torch.bfloat16).cpu()
        hist[("passthrough", "BF16")] += 1

    for mod_name, mod in model.named_modules():
        non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
        for buf_name, buf in mod.named_buffers(recurse=False):
            if buf_name in non_persistent:
                continue
            full = f"{mod_name}.{buf_name}" if mod_name else buf_name
            if any(full.startswith(c + ".") or full == c for c in covered):
                continue
            if full in out:
                continue
            out[full] = buf.detach().to(torch.bfloat16).cpu()
            hist[("passthrough_buffer", "BF16")] += 1

    return out, dict(hist)


# ---------------------------------------------------------------------------
# Compressed-tensors quantization_config
# ---------------------------------------------------------------------------
NVFP4_SCHEME = {
    "format": "nvfp4-pack-quantized",
    "weights": {
        "num_bits": 4, "type": "float", "strategy": "tensor_group",
        "group_size": 16, "symmetric": True, "dynamic": False,
        "scale_dtype": "torch.float8_e4m3fn",
        "zp_dtype": "torch.float8_e4m3fn",
        "observer": "memoryless_minmax",
    },
    "input_activations": {
        "num_bits": 4, "type": "float", "strategy": "tensor_group",
        "group_size": 16, "symmetric": True,
        "dynamic": "local", "observer": "static_minmax",
        "scale_dtype": "torch.float8_e4m3fn",
        "zp_dtype": "torch.float8_e4m3fn",
    },
}
MXFP8_SCHEME = {
    "format": "mxfp8-quantized",
    "weights": {
        "num_bits": 8, "type": "float", "strategy": "group",
        "group_size": 32,
        "symmetric": True, "dynamic": False,
        "scale_dtype": "torch.uint8",
        "zp_dtype": "torch.uint8",
        "observer": "memoryless_minmax",
    },
    "input_activations": {
        "num_bits": 8, "type": "float", "strategy": "group",
        "group_size": 32,
        "symmetric": True, "dynamic": True,
        "scale_dtype": "torch.uint8",
        "zp_dtype": "torch.uint8",
    },
}
def _bf16_packed_expert_ignore_regex(
        recipe_key: str,
        profile,
) -> list[str]:
    """If `recipe_key` names a BF16 packed-MoE tensor
    (`...experts.gate_up_proj` or `...experts.down_proj`), return one or
    more regex strings that match the corresponding per-expert Linear
    qnames at scheme-dispatch time, so vLLM's `find_matched_target`
    routes them to `ignore` instead of a config_groups target.

    For `gate_up_proj` we emit two patterns (one for `gate_proj`, one
    for `up_proj`) because the packed tensor splits into both at
    materialize time. Returns `[]` if the recipe_key doesn't look
    like a packed-expert entry or the profile has no vLLM class to
    derive naming from."""
    import re as _re

    # Does this recipe key name a packed-expert tensor?
    m = _re.match(r"^(.*\.)(experts)\.(gate_up_proj|down_proj|w\d|gate_proj|up_proj)$",
                  recipe_key)
    if not m:
        return []
    parent = m.group(1)          # `model.layers.X.`  or `model.layers.X.moe.`
    pn = m.group(3)

    # Convert the recipe parent prefix to a live-model prefix by
    # asking the profile. `profile.live_to_recipe_name` is the
    # opposite direction, so we'd need its inverse — instead emit a
    # regex loose enough to match both live forms on both sides of
    # the remap (text-only-style `...layers.X.experts.Y.*` and
    # multimodal `language_model.model.layers.X.moe.experts.Y.*`).
    # The profile's `per_expert_moe_regex` already encodes the live
    # form; we narrow it to this specific layer by pinning the layer
    # index.
    layer_idx = None
    lm = _re.search(r"\.layers\.(\d+)\.", recipe_key)
    if lm:
        layer_idx = lm.group(1)
    # Build per-proj regex. `gate_up_proj` splits into `gate_proj`
    # and `up_proj` on disk; `down_proj` stays as `down_proj`.
    if pn == "gate_up_proj":
        proj_options = "gate_proj|up_proj"
    elif pn == "down_proj":
        proj_options = "down_proj"
    else:
        proj_options = _re.escape(pn)

    # Use the profile's own regex as the base; swap its `(gate|up|down)_proj`
    # group with the exact projections we emit, and constrain to this
    # layer.
    base = profile.per_expert_moe_regex() if profile else None
    if not base or not base.startswith("re:"):
        # No profile regex — emit a conservative default spanning
        # both common live-module conventions.
        patterns = []
        if layer_idx is None:
            return patterns
        # Try the multimodal (Gemma / Qwen3.6) layout first.
        patterns.append(
            rf"re:^language_model[.]model[.]layers[.]{layer_idx}[.]"
            rf"(?:moe[.])?experts[.][0-9]+[.]({proj_options})$"
        )
        # And the text-only / dense layout.
        patterns.append(
            rf"re:^model[.]layers[.]{layer_idx}[.]"
            rf"(?:moe[.])?experts[.][0-9]+[.]({proj_options})$"
        )
        return patterns

    # Profile-provided regex. Strip the `re:` prefix, pin to this
    # layer index, constrain to the emitted projections.
    body = base[len("re:"):]
    # Replace [0-9]+ between layers.X. and .experts. with the specific
    # layer index. Fall back to leaving as-is if the pattern doesn't
    # match our expectations.
    pinned = _re.sub(r"layers\[\.\]\[0-9\]\+", f"layers[.]{layer_idx}", body, count=1)
    # Replace `(gate|up|down)_proj` with only the split projections we
    # actually emitted (so we don't over-ignore).
    pinned = pinned.replace("(gate|up|down)_proj", f"({proj_options})")
    return [f"re:{pinned}"]


FORMAT_SCHEME = {
    "NVFP4": NVFP4_SCHEME,
    "MXFP8": MXFP8_SCHEME,
}


def build_quantization_config(
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    extra_ignore: Iterable[str] = (),
    *,
    profile: "ModelProfile | None" = None,
) -> dict:
    """Emit a `quantization_config` dict with explicit per-name targets
    grouped by format. Targets and ignore are remapped to vLLM's
    internal naming via the supplied `profile` so `find_matched_target`
    matches.

    `extra_ignore` is for module qnames that aren't in the recipe at
    all but should be excluded from any catch-all group (e.g. routers).
    The catch-all default group is the format with the most non-BF16
    members (typically NVFP4).

    `profile` controls the architecture-specific bits: name remap,
    per-expert MoE / MTP regexes. Defaults to `DefaultProfile()` (plain
    names, no catch-all regexes) when omitted.
    """
    from .model_profiles import DefaultProfile
    from .model_profiles.vllm_registry import (
        vllm_class_for_architecture, packed_modules_mapping_from_class,
    )
    profile = profile or DefaultProfile()

    by_fmt: dict[str, list[str]] = {}
    ignore: list[str] = []
    for n in bf16_passthrough:
        ignore.append(profile.to_vllm_internal_name(n))
    for n in extra_ignore:
        ignore.append(profile.to_vllm_internal_name(n))
    for name, fmt in sorted(assignment.items()):
        vllm_name = profile.to_vllm_internal_name(name)
        if fmt == "BF16":
            ignore.append(vllm_name)
            # Packed MoE tensors in BF16 are emitted as per-expert
            # per-projection splits (not as the 3D packed tensor). vLLM
            # scheme-dispatches against the per-expert Linear qnames
            # (e.g. `...experts.0.gate_proj`), not the packed parent —
            # so the `ignore` for a BF16 packed-expert recipe entry
            # must cover every per-expert per-projection for that layer.
            # We emit a narrow regex per layer rather than enumerating
            # hundreds of explicit names.
            regex_list = _bf16_packed_expert_ignore_regex(name, profile)
            for r in regex_list:
                ignore.append(r)
            continue
        by_fmt.setdefault(fmt, []).append(vllm_name)

    # Fill in fused-sibling members that exist in the live vLLM
    # model but weren't in the probe assignment — e.g. Gemma 4's
    # full_attention layers have no v_proj on disk, so the probe
    # never saw it, but vLLM's QKVParallelLinear still instantiates
    # a v_proj sub-module that gets k_proj's weights at load. Scheme
    # dispatch requires all fused siblings to have consistent
    # scheme. We infer missing siblings by walking the assignment for
    # fused groups that landed in `ignore` and filling in every
    # sibling from vLLM's `packed_modules_mapping` — including ones
    # we never saw weights for.
    vllm_cls = vllm_class_for_architecture(profile.vllm_architecture_class() or "")
    packed_mapping = packed_modules_mapping_from_class(vllm_cls)
    if packed_mapping:
        # Reverse map: sibling-leaf-name -> fused-name (e.g.
        # q_proj -> qkv_proj).
        leaf_to_fused: dict[str, str] = {}
        for fused_name, siblings in packed_mapping.items():
            for s in siblings:
                leaf_to_fused[s] = fused_name
        # Set of leaf suffixes we should have. We'll only fill in
        # siblings under names that match known fused patterns.
        bf16_name_set = set(ignore)
        for name, fmt in list(assignment.items()):
            if fmt != "BF16":
                continue
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in leaf_to_fused:
                continue
            fused = leaf_to_fused[leaf]
            expected_siblings = packed_mapping[fused]
            parent = name[: -(len(leaf))]
            for sib in expected_siblings:
                full = parent + sib
                vllm_name = profile.to_vllm_internal_name(full)
                if vllm_name not in bf16_name_set:
                    ignore.append(vllm_name)
                    bf16_name_set.add(vllm_name)

    # Fused-linear target emission. vLLM's model-loading time fuses
    # siblings from `packed_modules_mapping` into a single packed Linear
    # (e.g. Qwen3.5 DeltaNet's `in_proj_qkv + in_proj_z → in_proj_qkvz`,
    # standard `q_proj + k_proj + v_proj → qkv_proj`). Scheme dispatch
    # keys off the FUSED module's prefix, so our config must list that
    # fused name alongside the siblings. When all expected siblings
    # share one format, emit the fused name into that format's target
    # list; when all land in ignore, emit the fused name into ignore.
    # Mixed-format fused groups are blocked upstream by the allocator's
    # `fused_sibling_group` pre-pass — but we defensively skip emitting
    # a fused target in that case rather than guess.
    if packed_mapping:
        # Map leaf sibling → fused-name, using packed_mapping that vLLM
        # reads at load time.
        leaf_to_fused = {s: fused for fused, sibs in packed_mapping.items()
                         for s in sibs}

        # Build parent-path → {leaf: (fmt|IGNORE, vllm_name)} for every
        # live entry (assignment + extra_ignore + bf16_passthrough).
        def _parent_leaf(vname: str):
            parts = vname.rsplit(".", 1)
            if len(parts) != 2:
                return None, vname
            return parts[0], parts[1]

        # (parent, leaf) → (fmt or "IGNORE")
        leaf_state: dict[tuple[str, str], str] = {}
        for fmt, names in by_fmt.items():
            for vname in names:
                parent, leaf = _parent_leaf(vname)
                if parent is None:
                    continue
                leaf_state[(parent, leaf)] = fmt
        ignore_set = set(ignore)
        for vname in ignore_set:
            parent, leaf = _parent_leaf(vname)
            if parent is None:
                continue
            leaf_state.setdefault((parent, leaf), "IGNORE")

        # For each (parent, fused) pair where all siblings are present
        # and share a state, emit the fused-name target.
        fused_emitted: set[str] = set()
        parents = {p for (p, _) in leaf_state}
        for parent in parents:
            for fused_name, sibs in packed_mapping.items():
                # Skip degenerate fused definitions (single-sibling).
                if len(sibs) < 2:
                    continue
                states = [leaf_state.get((parent, s)) for s in sibs]
                if any(s is None for s in states):
                    continue  # not all siblings present → skip
                if len(set(states)) != 1:
                    continue  # mixed formats → caller's bug; don't emit
                state = states[0]
                fused_vllm_name = f"{parent}.{fused_name}"
                if fused_vllm_name in fused_emitted:
                    continue
                fused_emitted.add(fused_vllm_name)
                if state == "IGNORE":
                    ignore.append(fused_vllm_name)
                else:
                    by_fmt.setdefault(state, []).append(fused_vllm_name)

    if not by_fmt:
        return {}

    sizes = {k: len(v) for k, v in by_fmt.items()}
    catchall = max(sizes, key=sizes.get) if sizes else None
    config_groups = {}
    idx = 0
    for fmt, names in by_fmt.items():
        if fmt == catchall:
            continue
        scheme = deepcopy(FORMAT_SCHEME[fmt])
        scheme["targets"] = [_explicit_regex(n) for n in sorted(names)]
        config_groups[f"group_{idx}"] = scheme
        idx += 1
    if catchall is not None:
        scheme = deepcopy(FORMAT_SCHEME[catchall])
        # Explicit per-name targets, NOT a class-name catch-all
        # ("Linear"). The class-name catch-all matches via a substring
        # check against module class (e.g. MergedColumnParallelLinear)
        # and short-circuits vLLM's fused-layer regex resolution, which
        # is needed to route the explicit per-component MXFP8 targets
        # to vLLM's fused parameter (in_proj_qkvz, qkv_proj, etc.).
        # We additionally add architecture-specific per-expert regexes
        # from the profile so ~30k per-expert MoE entries don't need
        # explicit enumeration.
        explicit = sorted(by_fmt[catchall])
        expert_regexes = []
        if (r := profile.per_expert_moe_regex()) is not None:
            expert_regexes.append(r)
        if (r := profile.per_expert_mtp_regex()) is not None:
            expert_regexes.append(r)
        scheme["targets"] = [_explicit_regex(n) for n in explicit] + expert_regexes
        config_groups[f"group_{idx}"] = scheme

    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": config_groups,
        "ignore": sorted(set(ignore)),
        "quantization_status": "compressed",
    }


# ---------------------------------------------------------------------------
# Recipe canonicalization + Main
# ---------------------------------------------------------------------------
def _canonicalize_assignment(raw: dict) -> dict[str, str]:
    """Accept either AutoRound-style dicts (`{key: {bits: 4, data_type: nv_fp,
    ...}}`) or shorthand (`{key: "NVFP4"}`). Return `{key: fmt_str}` with
    fmt in {"NVFP4", "MXFP8", "BF16"}."""
    out: dict[str, str] = {}
    for k, v in raw.items():
        name = _strip_weight(k)
        out[name] = canonicalize_format(v)
    return out


# Per-expert siblings map to a fused packed parent at recipe level.
# If the parent IS quantized, the per-expert source keys are already
# covered and must NOT be added to `extra_ignore` — otherwise vLLM's
# compressed-tensors loader marks the FusedMoE layer as un-quantized
# and the NVFP4 scale params (w2_input_global_scale, ...) never get
# registered, crashing at weight-load.
_PER_EXPERT_RE = re.compile(
    r"^(?P<prefix>.+\.experts)\.\d+\.(?P<proj>gate|up|down)_proj$")


def _per_expert_parent(base: str) -> str | None:
    """Map a per-expert source tensor base like
    `model.layers.0.mlp.experts.3.gate_proj` to its packed parent
    `model.layers.0.mlp.experts.gate_up_proj` / `.down_proj`, or None
    if `base` is not a per-expert tensor."""
    m = _PER_EXPERT_RE.match(base)
    if not m:
        return None
    proj = m.group("proj")
    parent = "gate_up_proj" if proj in ("gate", "up") else "down_proj"
    return f"{m.group('prefix')}.{parent}"


def compute_extra_ignore(source_shape_iter, assignment: dict[str, str]
                         ) -> list[str]:
    """Return the list of 2D `.weight` basenames that must be added to
    the compressed-tensors `ignore` set because the recipe doesn't cover
    them.

    `source_shape_iter` yields `(ckpt_key, shape)` for every tensor in
    the source checkpoint (or None for shape when unknown — treated as
    non-2D and skipped). `assignment` maps recipe names to formats.

    Per-expert source keys (e.g. `...experts.3.gate_proj.weight`) are
    NOT added to `extra_ignore` when their packed parent is in the
    assignment — the parent's emitted compressed-tensors scheme already
    covers them at vLLM load time, and adding the per-expert name to
    `ignore` would mark the FusedMoE layer as un-quantized.
    """
    extra_ignore: list[str] = []
    seen_recipe = set(assignment)
    for ckpt_key, shape in source_shape_iter:
        if not ckpt_key.endswith(".weight"):
            continue
        base = ckpt_key[:-7]
        recipe_name = ("model." + base[len("model.language_model."):]
                       if base.startswith("model.language_model.")
                       else base)
        if recipe_name in seen_recipe:
            continue
        parent = _per_expert_parent(recipe_name)
        if parent is not None and parent in seen_recipe:
            continue
        if shape is None or len(shape) != 2:
            continue
        extra_ignore.append(base)
    return extra_ignore


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="HF model dir (source safetensors + config.json)")
    ap.add_argument("--layer-config", required=True,
                    help="layer_config.json from allocator.py")
    ap.add_argument("--output", required=True,
                    help="Output directory for the compressed checkpoint")
    ap.add_argument("--shard-bytes", type=int, default=5 * 1024**3,
                    help="Approx per-shard size in bytes (default 5 GiB)")
    ap.add_argument("--device", default="cuda",
                    help="Device for quantization arithmetic. Layer "
                         "weights are read into this device; "
                         "_quantize_2d / _quantize_3d_packed run here; "
                         "outputs are moved to CPU before storage.")
    ap.add_argument("--offload-folder", default=None,
                    help="Accelerate disk-offload folder (defaults to "
                         "sibling of output).")
    ap.add_argument("--ignore", nargs="*", default=["lm_head"],
                    help="Module qnames to keep at bf16 even if the "
                         "allocator assigned another format.")
    args = ap.parse_args()

    from .model_profiles import detect_profile
    profile = detect_profile(args.model)
    print(f"[export-stream] model profile: {profile.name}", flush=True)

    with open(args.layer_config) as f:
        raw_recipe = json.load(f)
    assignment = _canonicalize_assignment(raw_recipe)
    validate_mtp_assignment_coverage(args.model, assignment, profile)
    fmts = Counter(assignment.values())
    print(f"[export-stream] recipe: {len(assignment)} entries  mix={dict(fmts)}",
          flush=True)

    dtype = torch.bfloat16
    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_passthrough = set(args.ignore)
    if args.offload_folder is None:
        args.offload_folder = str(out_dir / "_streaming_offload")

    tensors, hist = materialize_tensors_streaming(
        args.model, assignment,
        profile=profile, bf16_passthrough=bf16_passthrough,
        dtype=dtype, device=device,
        offload_folder=args.offload_folder,
    )
    print(f"[export-stream] materialized {len(tensors)} tensors  hist={hist}",
          flush=True)

    # Rename body keys → `model.language_model.` on disk for multimodal-
    # umbrella arches (Qwen3.5/3.6 ConditionalGeneration, Gemma 4
    # ConditionalGeneration). Our streaming loop produces the text-only
    # `model.layers.X.*` form.
    body_infix = getattr(profile, "body_ondisk_infix", None)
    if callable(body_infix):
        infix = body_infix()
    else:
        # Default: Qwen3.5/3.6 pattern. Profiles for non-multimodal
        # archs can return "" and we'll skip the rename.
        infix = "language_model." if profile.name.startswith("qwen3_5") else ""
    if infix:
        renamed: dict[str, torch.Tensor] = {}
        for k, v in tensors.items():
            if (k.startswith("model.layers.")
                    or k.startswith("model.embed_tokens")
                    or k.startswith("model.norm")):
                renamed[f"model.{infix}{k[len('model.'):]}"] = v
            else:
                renamed[k] = v
        tensors = renamed
        print(f"[export-stream] renamed body → model.{infix}...",
              flush=True)

    # MTP materialization if the profile has heads. Uses the in-memory
    # helper — MTP heads are small enough that full-model residency
    # isn't a concern.
    mtp_tensors: dict[str, torch.Tensor] = {}
    if profile.has_mtp():
        print("[export-stream] materializing MTP tensors ...", flush=True)
        mtp_tensors = _materialize_mtp_tensors(
            args.model, assignment,
            bf16_passthrough=bf16_passthrough, hist=hist)
        print(f"[export-stream] MTP: {len(mtp_tensors)} tensors", flush=True)
    else:
        print(f"[export-stream] profile '{profile.name}' has no MTP — "
              "skipping", flush=True)

    # Merge source passthrough (visual/audio towers etc.) that aren't
    # part of our streaming pass. Drop entries that MTP materialize
    # already covered.
    passthrough_prefixes = tuple(profile.source_passthrough_prefixes())
    if passthrough_prefixes:
        src_extra = _load_source_passthrough(
            args.model, prefix_filters=passthrough_prefixes)
        materialized_bases: set[str] = set()
        for k in mtp_tensors:
            base = k
            for suf in (".weight_packed", ".weight_scale",
                        ".weight_global_scale", ".input_global_scale",
                        ".weight"):
                if k.endswith(suf):
                    base = k[:-len(suf)] + ".weight"
                    break
            materialized_bases.add(base)
            m = re.match(r"^(mtp\.layers\.\d+\.mlp\.experts)\.\d+\.(gate|up|down)_proj\.", k)
            if m:
                if m.group(2) in ("gate", "up"):
                    materialized_bases.add(f"{m.group(1)}.gate_up_proj")
                else:
                    materialized_bases.add(f"{m.group(1)}.down_proj")
        src_extra = {k: v for k, v in src_extra.items()
                     if k not in materialized_bases}
        for k in list(src_extra.keys()):
            if k in tensors or k in mtp_tensors:
                del src_extra[k]
        tensors.update(mtp_tensors)
        tensors.update(src_extra)
        print(f"[export-stream] merged {len(src_extra)} source-passthrough + "
              f"{len(mtp_tensors)} MTP tensors", flush=True)
    else:
        tensors.update(mtp_tensors)

    print("[export-stream] writing safetensors shards ...", flush=True)
    t_write = time.time()
    write_sharded_safetensors(tensors, out_dir, args.shard_bytes)
    print(f"[export-stream] sharded write: {time.time()-t_write:.1f}s",
          flush=True)

    # Scan source safetensors for 2D `.weight` keys not covered by the
    # recipe — these are visual encoder / unmapped Linears that vLLM
    # instantiates during model-construction time. Without an explicit
    # ignore entry, compressed-tensors' `find_matched_target` raises
    # `ValueError: Unable to find matching target for visual.merger.*`.
    src_dir = Path(args.model)

    def _source_shape_iter():
        if not src_dir.exists():
            return
        from safetensors import safe_open
        import os as _os
        for f in sorted(_os.listdir(src_dir)):
            if not f.endswith(".safetensors"):
                continue
            with safe_open(str(src_dir / f), framework="pt") as sf:
                for k in sf.keys():
                    try:
                        shape = list(sf.get_slice(k).get_shape())
                    except Exception:
                        shape = None
                    yield k, shape

    extra_ignore = compute_extra_ignore(_source_shape_iter(), assignment)
    print(f"[export-stream] extra ignore (unmapped Linears): "
          f"{len(extra_ignore)}", flush=True)

    write_config_with_quantization(
        args.model, out_dir, assignment, bf16_passthrough,
        extra_ignore=extra_ignore)
    _copy_tokenizer(args.model, out_dir)

    with open(out_dir / "mixed_native_manifest.json", "w") as f:
        json.dump({
            "source_model": args.model,
            "source_recipe": args.layer_config,
            "format_histogram": {f"{k[0]}/{k[1]}": v for k, v in hist.items()},
            "n_assignment_entries": len(assignment),
            "ignore": sorted(bf16_passthrough),
        }, f, indent=2)

    print(f"[export-stream] done. Serve with:\n"
          f"  vllm serve {out_dir.resolve()} --quantization compressed-tensors",
          flush=True)


# ---------------------------------------------------------------------------
# Sharded safetensors writer (mirrors HF transformers' shard layout so
# the index file is the same one transformers + vLLM expect).
# ---------------------------------------------------------------------------
def write_sharded_safetensors(
    tensors: dict[str, torch.Tensor],
    out_dir: Path,
    shard_bytes: int,
) -> None:
    # Detach + clone any tensors that share underlying storage so
    # safetensors' dedup check doesn't raise. This covers tied
    # embeddings (Gemma 4: `lm_head.weight` ≡ `embed_tokens.weight`)
    # and any other view-ties produced by HF's
    # `_tied_weights_keys`. Cost: one extra copy of the embed matrix;
    # correctness: identical bytes on disk, no runtime semantic change.
    seen_storage: dict[int, str] = {}
    for k, t in list(tensors.items()):
        try:
            sid = t.untyped_storage().data_ptr()
        except Exception:
            continue
        if sid in seen_storage:
            # This tensor shares storage with an earlier one.
            # Deep-copy so safetensors treats them independently.
            tensors[k] = t.detach().clone().contiguous()
        else:
            seen_storage[sid] = k

    keys = sorted(tensors.keys())
    sizes = {k: tensors[k].numel() * tensors[k].element_size() for k in keys}
    total = sum(sizes.values())
    n_shards = max(1, math.ceil(total / shard_bytes))
    target = math.ceil(total / n_shards)

    shards: list[list[str]] = [[]]
    cur = 0
    for k in keys:
        if cur + sizes[k] > target and shards[-1]:
            shards.append([])
            cur = 0
        shards[-1].append(k)
        cur += sizes[k]

    if len(shards) == 1:
        path = out_dir / "model.safetensors"
        save_file(
            {k: tensors[k].contiguous() for k in shards[0]},
            str(path),
            metadata={"format": "pt"},
        )
        return

    weight_map: dict[str, str] = {}
    n = len(shards)
    for i, shard_keys in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        save_file(
            {k: tensors[k].contiguous() for k in shard_keys},
            str(out_dir / shard_name),
            metadata={"format": "pt"},
        )
        for k in shard_keys:
            weight_map[k] = shard_name

    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump({
            "metadata": {"total_size": total},
            "weight_map": weight_map,
        }, f, indent=2)


def write_config_with_quantization(
    src_model: str, out_dir: Path,
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    extra_ignore: Iterable[str] = (),
) -> None:
    from .model_profiles import detect_profile
    profile = detect_profile(src_model)
    src_cfg_path = Path(src_model) / "config.json"
    cfg = json.load(open(src_cfg_path))
    qc = build_quantization_config(assignment, bf16_passthrough,
                                   extra_ignore, profile=profile)
    if qc:
        cfg["quantization_config"] = qc
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def _materialize_mtp_tensors(src_model: str,
                             assignment: dict[str, str],
                             *,
                             bf16_passthrough: set[str],
                             hist: dict) -> dict[str, torch.Tensor]:
    """Quantize MTP weights per the allocator recipe.

    Transformers v5 does not instantiate MTP modules when loading
    Qwen3.5/3.6 MoE checkpoints (see `_keys_to_ignore_on_load_unexpected`),
    so the streaming decoder-layer sweep never sees any `mtp.*` entry in
    `assignment`. We build a standalone MTP module, load the source
    `mtp.*` weights into it, wrap it in a parent module named `mtp` (so
    qualified names come out as `mtp.fc`, `mtp.layers.0.self_attn.q_proj`,
    ...), and run the in-memory materialize helper.

    Output tensor names match the checkpoint convention (`mtp.fc.*`,
    `mtp.layers.0.<rest>`). vLLM's `qwen3_5_mtp.load_weights` remaps
    `mtp.→model.` at load time.
    """
    from .mtp_module import MtpModule, _load_into_mtp, _load_mtp_state_dict
    from transformers import AutoConfig

    # Build an MTP wrapper with source weights.
    cfg = AutoConfig.from_pretrained(src_model, trust_remote_code=True)
    text_config = getattr(cfg, "text_config", cfg)
    inner = MtpModule(text_config)
    wrapper = nn.Module()
    wrapper.add_module("mtp", inner)
    wrapper.to(dtype=torch.bfloat16)
    raw = _load_mtp_state_dict(src_model)
    _load_into_mtp(inner, raw)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # Filter assignment to just `mtp.*` entries.
    mtp_assignment = {k: v for k, v in assignment.items() if k.startswith("mtp.")}
    if not mtp_assignment:
        return {}

    out, sub_hist = _materialize_tensors_inmemory(
        wrapper, mtp_assignment, bf16_passthrough=bf16_passthrough,
    )
    # Merge MTP histogram into caller's.
    for k, v in sub_hist.items():
        hist[("mtp_" + k[0], k[1])] = hist.get(("mtp_" + k[0], k[1]), 0) + v
    return out


def _load_source_passthrough(src_model: str,
                             prefix_filters: tuple[str, ...]
                             ) -> dict[str, torch.Tensor]:
    """Pull tensors from the source safetensors whose key begins with
    any of `prefix_filters`. Returns the loaded tensors so they can be
    written back verbatim into the export. Used for visual encoder +
    MTP head weights that the recipe doesn't touch but vLLM expects to
    find at load time.
    """
    import os
    from safetensors.torch import safe_open
    src_dir = Path(src_model)
    out: dict[str, torch.Tensor] = {}
    for f in sorted(os.listdir(src_dir)):
        if not f.endswith(".safetensors"):
            continue
        with safe_open(str(src_dir / f), framework="pt") as sf:
            for k in sf.keys():
                if any(k.startswith(p) for p in prefix_filters):
                    out[k] = sf.get_tensor(k)
    return out


def _copy_tokenizer(src_model: str, out_dir: Path) -> None:
    src = Path(src_model)
    for name in (
        "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
        "special_tokens_map.json", "merges.txt", "vocab.json",
        "added_tokens.json", "generation_config.json", "configuration.json",
        # Multimodal preprocessor configs — vLLM's loader for
        # qwen3_vl_moe constructs the multimodal processor even for
        # text-only requests, so the preprocessor files must travel
        # with the checkpoint.
        "preprocessor_config.json", "video_preprocessor_config.json",
        "processor_config.json",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, out_dir / name)


def _source_has_prefixed_weights(src_model: str, prefix: str) -> bool:
    """Return True when the source safetensors index contains any key
    beginning with `prefix`.

    Export-time validation should use the index rather than a loaded HF
    model because transformers intentionally drops `mtp.*` on load for
    Qwen3.5/3.6, which would otherwise make missing recipe coverage look
    benign.
    """
    idx_path = Path(src_model) / "model.safetensors.index.json"
    if not idx_path.exists():
        return False
    with open(idx_path) as f:
        weight_map = json.load(f).get("weight_map", {})
    return any(k.startswith(prefix) for k in weight_map)


def validate_mtp_assignment_coverage(src_model: str,
                                     assignment: dict[str, str],
                                     profile) -> None:
    """Fail fast when an architecture with MTP source weights is being
    exported without any allocator coverage for `mtp.*`.

    Passing raw MTP weights through silently produces a checkpoint that
    looks complete but violates PrismQuant's intended contract: MTP must
    participate in the same probe/cost/allocation loop as the body. This
    exact state was observed on Qwen3.5-122B where the body artifacts on
    disk were generated without merged MTP probe/cost results.
    """
    if not profile.has_mtp():
        return
    if not _source_has_prefixed_weights(src_model, "mtp."):
        return
    if any(k.startswith("mtp.") for k in assignment):
        return
    raise RuntimeError(
        "source checkpoint contains mtp.* weights but the allocator recipe "
        "contains no mtp.* entries. Re-run the incremental probe + cost "
        "with --include-mtp (the default) so mtp.* tensors are measured, "
        "then rerun allocator/export."
    )


if __name__ == "__main__":
    main()
