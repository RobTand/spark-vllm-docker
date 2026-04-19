#!/usr/bin/env python3
"""
measure_bit_utility.py — measure per-Linear bit-utility curves.

For each Linear L in the model, and for each bit count b in {1..16}, quantize
just L to b bits (per-group symmetric INT-b with fp32 scale), leaving the rest
of the model at BF16. Measure KL vs the BF16 reference on a calibration slice.
Record KL(L, b). Restore L. Next Linear.

The curve's derivative dKL/db is the marginal value of bit b in Linear L.
Ranking all marginal values across all Linears gives a global bit-importance
order, which feeds the water-filling allocator (allocate_bits.py).

Output: curves.json with per-Linear KL curves and metadata.

Memory discipline: at any time, only one Linear has a cloned original weight
stored alongside its quantized form. Peak extra memory ≈ sizeof(largest Linear).
The rest of the model sits at bf16 in unified memory.

Usage:
    python3 measure_bit_utility.py \\
        --model /models/Qwen2.5-0.5B \\
        --output /tmp/curves/qwen25-0.5b.json \\
        --bit-levels 1 2 3 4 5 6 7 8 10 12 14 16 \\
        --n-calib-samples 8 --calib-seqlen 256
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse staging + KL utilities from build_rtn_cache
sys.path.insert(0, str(Path(__file__).parent))
from build_rtn_cache import (
    stage_multimodal,
    load_wikitext_calibration,
    cache_reference_log_probs,
    measure_kl,
    iter_quantizable_tensors,
    should_always_skip,
)


# ---------------------------------------------------------------------------
# Multi-bit integer quantizer (per-group, symmetric)
# ---------------------------------------------------------------------------

def int_quantize_per_group(
    weight: torch.Tensor, bits: int, group_size: Optional[int] = None,
) -> torch.Tensor:
    """Symmetric per-group INT-b quantize-dequantize. Returns a tensor of the
    same shape and dtype as ``weight``, representing the round-tripped values.

    Parameters:
        bits: 1..16. At 1 bit, uses ±(per-group mean absolute value) which is
            the standard 1-bit quantization scheme. At 16 bits, returns weight
            unchanged (no-op). For 2..15, uses symmetric signed integer with
            per-group fp32 scale and qmax = 2^(b-1) - 1.
        group_size: defaults to 128 for bits >= 5, 32 for bits < 5.
            Smaller groups at low bits compensate for reduced dynamic range.

    Handles both 2D Linear weights (out, in) and 3D fused MoE expert tensors
    (E, out, in) — the latter is reshaped to 2D, quantized, and reshaped back.
    """
    if bits >= 16:
        return weight

    if group_size is None:
        group_size = 128 if bits >= 5 else 32

    orig_dtype = weight.dtype
    orig_shape = weight.shape
    ndim = weight.dim()
    w = weight.float()

    if ndim == 3:
        E, out_f, in_f = w.shape
        w = w.reshape(E * out_f, in_f)
    elif ndim == 2:
        out_f, in_f = w.shape
    else:
        raise ValueError(f"unsupported weight rank: {ndim}")

    n_groups = (in_f + group_size - 1) // group_size
    pad = n_groups * group_size - in_f
    if pad > 0:
        w = F.pad(w, (0, pad))
    grouped = w.view(w.shape[0], n_groups, group_size)

    if bits == 1:
        # 1-bit: sign * per-group mean absolute value (BitNet-style)
        scale = grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        dequant = grouped.sign() * scale
    else:
        max_abs = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        qmax = (2 ** (bits - 1)) - 1
        scale = max_abs / qmax
        q = (grouped / scale).round().clamp(-qmax - 1, qmax)
        dequant = q * scale

    dequant = dequant.view(w.shape[0], n_groups * group_size)
    if pad > 0:
        dequant = dequant[:, :in_f]

    if ndim == 3:
        dequant = dequant.view(*orig_shape)

    return dequant.to(orig_dtype)


# ---------------------------------------------------------------------------
# Per-Linear bit-utility measurement loop
# ---------------------------------------------------------------------------

def measure_linear_curve(
    model, mod, attr: str, bits_to_try: List[int],
    calib_ids, ref_log_probs, device,
) -> Dict[int, float]:
    """Measure KL at each bit count for one Linear.

    Saves the original weight, loops over bit counts quantizing + measuring +
    restoring. Peak extra memory = sizeof(this Linear's weight).
    """
    param = getattr(mod, attr)
    original = param.data.clone()
    curve: Dict[int, float] = {}
    try:
        for b in bits_to_try:
            if b >= 16:
                # bf16 reference = 0 by definition
                curve[b] = 0.0
                continue
            quantized = int_quantize_per_group(original, b)
            param.data.copy_(quantized)
            curve[b] = measure_kl(model, calib_ids, ref_log_probs, device)
    finally:
        param.data.copy_(original)
        del original
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True,
                        help="Output JSON path (curves.json)")
    parser.add_argument("--bit-levels", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
                        help="Bit counts to probe. Default covers the full "
                             "dynamic range without being exhaustive.")
    parser.add_argument("--n-calib-samples", type=int, default=8)
    parser.add_argument("--calib-seqlen", type=int, default=256)
    parser.add_argument("--max-linears", type=int, default=None,
                        help="Limit total Linears measured (for smoke tests)")
    parser.add_argument("--skip-small", type=int, default=1000,
                        help="Skip Linears with fewer than this many elements")
    args = parser.parse_args()

    t_start = time.time()

    # Stage model if multimodal
    staged, cleanup = stage_multimodal(args.model)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[bit-util] loading {staged}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            staged, torch_dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
        device = next(model.parameters()).device
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[bit-util]   {n_params:,} params", flush=True)

        # Reference log probs from BF16 model
        calib_ids = load_wikitext_calibration(
            tokenizer, args.n_calib_samples, args.calib_seqlen)
        print(f"[bit-util] computing BF16 reference log_probs", flush=True)
        ref_log_probs = cache_reference_log_probs(model, calib_ids, device)

        # Enumerate quantizable tensors
        targets = []
        for full_name, mod, attr in iter_quantizable_tensors(model):
            param = getattr(mod, attr)
            if param.numel() < args.skip_small:
                continue
            targets.append((full_name, mod, attr, tuple(param.shape), param.numel()))

        if args.max_linears:
            targets = targets[:args.max_linears]

        print(f"[bit-util] {len(targets)} Linears/experts to measure "
              f"× {len(args.bit_levels)} bit levels "
              f"= {len(targets) * len(args.bit_levels)} forward passes", flush=True)

        curves: Dict[str, dict] = {}
        t_measure = time.time()
        for i, (name, mod, attr, shape, numel) in enumerate(targets):
            t0 = time.time()
            curve = measure_linear_curve(
                model, mod, attr, args.bit_levels,
                calib_ids, ref_log_probs, device,
            )
            curves[name] = {
                "shape": list(shape),
                "numel": numel,
                "kl_per_bits": curve,
            }
            dt = time.time() - t0
            if (i + 1) % 5 == 0 or i == len(targets) - 1:
                elapsed = time.time() - t_measure
                remaining = elapsed / (i + 1) * (len(targets) - (i + 1))
                print(f"[bit-util]   {i+1}/{len(targets)} {name} "
                      f"({dt:.1f}s, ETA {remaining/60:.1f}m)", flush=True)
                # Quick peek at the curve shape
                kl4 = curve.get(4, None)
                kl8 = curve.get(8, None)
                if kl4 is not None and kl8 is not None:
                    print(f"[bit-util]     KL@4={kl4:.5f}  KL@8={kl8:.5f}",
                          flush=True)

        # Save
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output = {
            "source_model": args.model,
            "bit_levels": args.bit_levels,
            "n_calib_samples": args.n_calib_samples,
            "calib_seqlen": args.calib_seqlen,
            "n_params": n_params,
            "n_linears_measured": len(targets),
            "elapsed_sec": time.time() - t_start,
            "curves": curves,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[bit-util] done in {time.time() - t_start:.0f}s, saved to {args.output}",
              flush=True)
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
