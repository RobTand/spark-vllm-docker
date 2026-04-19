#!/usr/bin/env python3
"""
learn_nvfp4_rotation.py — learn per-group orthogonal rotations that minimize
NVFP4 reconstruction error.

For each quantizable Linear, and for each group of 16 weights along the input
dimension, learns a 16×16 orthogonal matrix R that minimizes:

    ||W_group @ R - Q_nvfp4(W_group @ R)||²_F

where Q_nvfp4 is the NVFP4 E2M1 round-trip. The rotation R is parameterized
via the Cayley transform of a skew-symmetric matrix A:

    R = (I - A) @ inv(I + A),   A = -A^T

which ensures R stays orthogonal throughout optimization. 120 free parameters
per 16×16 rotation (n(n-1)/2 for skew-symmetric).

At inference, R^T is applied to each group of 16 input features before the
matmul with the quantized weight. Fusible into dequantization with a custom
kernel (e.g., simd_shuffle_xor on Blackwell).

Usage:
    python3 learn_nvfp4_rotation.py \\
        --model /models/Qwen3.5-4B-bf16 \\
        --output /tmp/rotations/qwen35-4b.json \\
        --max-linears 20 --n-steps 200
"""
import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from build_rtn_cache import (
    stage_multimodal,
    iter_quantizable_tensors,
)


# ---------------------------------------------------------------------------
# NVFP4 E2M1 quantizer with straight-through estimator (STE)
# ---------------------------------------------------------------------------

# The 8 non-negative E2M1 magnitudes (the 16 representable values are ±these)
E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def nvfp4_ste(weight: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """NVFP4 E2M1 quantize-dequantize with STE for backward pass.

    Forward: standard NVFP4 round-to-nearest with per-group FP8 scale.
    Backward: straight-through (gradient passes through as identity).

    Input shape: (..., group_size) where the last dim is the group dimension.
    """
    # Per-group scale: max_abs / 6.0 (6.0 is the max E2M1 magnitude)
    max_abs = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 6.0

    # Normalize to [-6, 6] range
    normalized = weight / scale

    # Round to nearest E2M1 level
    sign = normalized.sign()
    abs_n = normalized.abs()

    # Vectorized round-to-nearest using the E2M1 level midpoints
    levels = E2M1_LEVELS.to(weight.device, dtype=weight.dtype)
    # Thresholds between adjacent levels (midpoints)
    thresholds = (levels[:-1] + levels[1:]) / 2  # [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]

    # Quantize using searchsorted (finds the nearest level)
    q_idx = torch.searchsorted(thresholds, abs_n.contiguous())
    q_abs = levels[q_idx]

    # Dequantize
    dequant = sign * q_abs * scale

    # STE: forward uses quantized, backward uses identity
    return weight + (dequant - weight).detach()


# ---------------------------------------------------------------------------
# Cayley parametrization of orthogonal matrices
# ---------------------------------------------------------------------------

def skew_to_orthogonal(A: torch.Tensor) -> torch.Tensor:
    """Cayley transform: skew-symmetric A → orthogonal R.

    R = (I - A) @ inv(I + A)

    A has shape (..., n, n) and must satisfy A = -A^T.
    Returns R with shape (..., n, n).
    """
    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    return torch.linalg.solve(I + A, I - A)


def make_skew_params(n_groups: int, group_size: int, device, dtype) -> torch.Tensor:
    """Create learnable skew-symmetric parameters.

    Returns a flat tensor of shape (n_groups, group_size*(group_size-1)//2)
    initialized to zero (→ identity rotation).
    """
    n_free = group_size * (group_size - 1) // 2
    return torch.zeros(n_groups, n_free, device=device, dtype=dtype)


def params_to_skew(params: torch.Tensor, group_size: int) -> torch.Tensor:
    """Convert flat free params to skew-symmetric matrices.

    params: (n_groups, n_free) where n_free = group_size*(group_size-1)//2
    Returns: (n_groups, group_size, group_size) skew-symmetric matrices.
    """
    n_groups = params.shape[0]
    A = torch.zeros(n_groups, group_size, group_size,
                    device=params.device, dtype=params.dtype)
    idx = 0
    for i in range(group_size):
        for j in range(i + 1, group_size):
            A[:, i, j] = params[:, idx]
            A[:, j, i] = -params[:, idx]
            idx += 1
    return A


# ---------------------------------------------------------------------------
# Per-Linear rotation learning
# ---------------------------------------------------------------------------

def learn_rotation_for_linear(
    weight: torch.Tensor,
    group_size: int = 16,
    n_steps: int = 200,
    lr: float = 0.01,
) -> Tuple[torch.Tensor, float, float]:
    """Learn per-group rotations for one Linear's weight tensor.

    Args:
        weight: (out_features, in_features) or (E, out, in) for fused MoE
        group_size: NVFP4 group size (16)
        n_steps: optimization steps
        lr: learning rate

    Returns:
        (rotations, mse_before, mse_after)
        rotations: (n_groups, group_size, group_size) learned orthogonal matrices
        mse_before: reconstruction MSE without rotation
        mse_after: reconstruction MSE with learned rotation
    """
    orig_shape = weight.shape
    ndim = weight.dim()
    if ndim == 3:
        E, out_f, in_f = weight.shape
        w = weight.reshape(E * out_f, in_f)
    elif ndim == 2:
        w = weight
    else:
        raise ValueError(f"unexpected ndim={ndim}")

    out_f, in_f = w.shape
    n_groups = in_f // group_size
    assert in_f % group_size == 0, f"in_f={in_f} not divisible by group_size={group_size}"

    # Reshape to (out_f, n_groups, group_size)
    w_grouped = w.view(out_f, n_groups, group_size).float()

    # Measure baseline MSE (no rotation)
    with torch.no_grad():
        w_q_baseline = nvfp4_ste(w_grouped)
        mse_before = (w_grouped - w_q_baseline).pow(2).mean().item()

    # Initialize learnable skew params (→ identity rotation)
    skew_params = make_skew_params(n_groups, group_size, w.device, torch.float32)
    skew_params = nn.Parameter(skew_params)
    optimizer = torch.optim.Adam([skew_params], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        A = params_to_skew(skew_params, group_size)         # (n_groups, 16, 16)
        R = skew_to_orthogonal(A)                            # (n_groups, 16, 16)
        # Rotate: w_rot[o, g, :] = w_grouped[o, g, :] @ R[g, :, :]
        w_rot = torch.einsum('ogk, gkh -> ogh', w_grouped, R)
        w_rot_q = nvfp4_ste(w_rot)
        loss = (w_rot - w_rot_q).pow(2).mean()
        loss.backward()
        optimizer.step()

    # Final MSE with learned rotation
    with torch.no_grad():
        A = params_to_skew(skew_params.data, group_size)
        R = skew_to_orthogonal(A)
        w_rot = torch.einsum('ogk, gkh -> ogh', w_grouped, R)
        w_rot_q = nvfp4_ste(w_rot)
        mse_after = (w_rot - w_rot_q).pow(2).mean().item()

    return R.detach(), mse_before, mse_after


def fixed_hadamard_mse(weight: torch.Tensor, group_size: int = 16) -> float:
    """Measure NVFP4 MSE with a fixed per-group Hadamard rotation."""
    ndim = weight.dim()
    if ndim == 3:
        E, out_f, in_f = weight.shape
        w = weight.reshape(E * out_f, in_f)
    elif ndim == 2:
        w = weight
    else:
        raise ValueError(f"unexpected ndim={ndim}")

    out_f, in_f = w.shape
    n_groups = in_f // group_size
    w_grouped = w.view(out_f, n_groups, group_size).float()

    # Build normalized Hadamard matrix (Sylvester construction)
    H = torch.tensor([[1.0]], device=w.device)
    while H.shape[0] < group_size:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    H = H / math.sqrt(group_size)  # normalize to orthogonal

    # Apply fixed Hadamard rotation to all groups
    w_rot = w_grouped @ H.T  # broadcast over (out_f, n_groups)
    with torch.no_grad():
        w_rot_q = nvfp4_ste(w_rot)
        mse = (w_rot - w_rot_q).pow(2).mean().item()
    return mse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-linears", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--group-size", type=int, default=16)
    args = parser.parse_args()

    t_start = time.time()
    staged, cleanup = stage_multimodal(args.model)
    try:
        from transformers import AutoModelForCausalLM
        print(f"[rot] loading {staged}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            staged, torch_dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True,
        )

        targets = []
        for full_name, mod, attr in iter_quantizable_tensors(model):
            param = getattr(mod, attr)
            if param.numel() < 1000:
                continue
            # Check input dim is divisible by group_size
            in_dim = param.shape[-1]
            if in_dim % args.group_size != 0:
                continue
            targets.append((full_name, mod, attr))
        if args.max_linears:
            targets = targets[:args.max_linears]

        print(f"[rot] {len(targets)} linears to optimize, "
              f"{args.n_steps} steps each", flush=True)

        results = []
        for i, (name, mod, attr) in enumerate(targets):
            t0 = time.time()
            param = getattr(mod, attr)
            R, mse_before, mse_after = learn_rotation_for_linear(
                param.data, args.group_size, args.n_steps, args.lr,
            )
            # Also measure fixed Hadamard for comparison
            mse_hadamard = fixed_hadamard_mse(param.data, args.group_size)

            improvement_learned = (1 - mse_after / mse_before) * 100
            improvement_hadamard = (1 - mse_hadamard / mse_before) * 100
            dt = time.time() - t0

            results.append({
                "name": name,
                "shape": list(param.shape),
                "mse_no_rotation": mse_before,
                "mse_hadamard": mse_hadamard,
                "mse_learned": mse_after,
                "improvement_hadamard_pct": improvement_hadamard,
                "improvement_learned_pct": improvement_learned,
            })

            if (i + 1) % 5 == 0 or i == len(targets) - 1:
                print(f"[rot] {i+1}/{len(targets)} {name}: "
                      f"none={mse_before:.6f} "
                      f"had={mse_hadamard:.6f} ({improvement_hadamard:+.1f}%) "
                      f"learned={mse_after:.6f} ({improvement_learned:+.1f}%) "
                      f"({dt:.1f}s)", flush=True)

        # Summary
        avg_imp_had = sum(r["improvement_hadamard_pct"] for r in results) / len(results)
        avg_imp_learned = sum(r["improvement_learned_pct"] for r in results) / len(results)
        print(f"\n{'='*60}")
        print(f"ROTATION COMPARISON ({len(results)} linears)")
        print(f"{'='*60}")
        print(f"  avg improvement (fixed Hadamard):   {avg_imp_had:+.1f}%")
        print(f"  avg improvement (learned per-group): {avg_imp_learned:+.1f}%")
        print(f"  learned vs Hadamard advantage:       {avg_imp_learned - avg_imp_had:+.1f}%")
        print(f"  total time: {time.time() - t_start:.0f}s")

        # Save
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output = {
            "source_model": args.model,
            "n_steps": args.n_steps,
            "lr": args.lr,
            "group_size": args.group_size,
            "n_linears": len(results),
            "avg_improvement_hadamard_pct": avg_imp_had,
            "avg_improvement_learned_pct": avg_imp_learned,
            "elapsed_sec": time.time() - t_start,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[rot] saved to {args.output}")
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
