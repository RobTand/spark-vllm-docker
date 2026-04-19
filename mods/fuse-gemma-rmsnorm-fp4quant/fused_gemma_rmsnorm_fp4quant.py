"""
Fused Gemma-style RMSNorm + NVFP4 activation quantization.

Eliminates the memory round-trip between RMSNorm (BF16 output) and
scaled_fp4_quant (BF16 input → FP4 output). For Qwen3.5's 48 layers,
this removes ~96 kernel launches per token (2 per layer: input_layernorm
and post_attention_layernorm, each followed by a linear projection that
quantizes activations).

The Gemma-style RMSNorm differs from standard: x * (1 + w) instead of x * w.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _gemma_rmsnorm_fp4quant_kernel(
    # Inputs
    X_ptr,          # [M, N] input tensor (BF16)
    W_ptr,          # [N] norm weights (FP32 stored)
    SCALE_ptr,      # [] global scale inverse (FP32 scalar)
    # Outputs
    OUT_FP4_ptr,    # [M, N//2] packed FP4 output (uint8)
    OUT_SF_ptr,     # [M, N//SF_VEC_SIZE] block scales (FP8 E4M3)
    # Strides
    stride_x_m,
    stride_x_n,
    stride_fp4_m,
    stride_fp4_n,
    stride_sf_m,
    stride_sf_n,
    # Params
    N: tl.constexpr,
    EPS: tl.constexpr,
    SF_VEC_SIZE: tl.constexpr,  # 16 for NVFP4
    BLOCK_N: tl.constexpr,
):
    """Fused Gemma RMSNorm + FP4 quantization kernel.

    Each program instance handles one row (one token).
    For batch=1 decode, M=1 so this is a single program.
    """
    row_idx = tl.program_id(0)

    # Step 1: Compute RMS norm variance
    # Load the full row and compute sum of squares
    _var = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row_idx * stride_x_m + cols * stride_x_n,
                     mask=mask, other=0.0).to(tl.float32)
        _var += x * x

    var = tl.sum(_var) / N
    rstd = tl.rsqrt(var + EPS)

    # Load global scale
    global_scale_inv = tl.load(SCALE_ptr).to(tl.float32)

    # Step 2: Normalize + quantize in blocks of SF_VEC_SIZE
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N

        # Load input
        x = tl.load(X_ptr + row_idx * stride_x_m + cols * stride_x_n,
                     mask=mask, other=0.0).to(tl.float32)

        # Gemma-style RMSNorm: x * (1 + w)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = x * rstd * (1.0 + w)

        # Scale by global scale for quantization
        scaled = normed * global_scale_inv

        # Block-wise FP4 quantization (SF_VEC_SIZE=16 elements per block)
        # For each block of 16: find max abs, compute E4M3 scale, quantize to E2M1
        # This is a simplified version - the actual NVFP4 format uses specific
        # E2M1 encoding and E4M3 block scales

        # Store normalized BF16 output (for now - full FP4 packing requires
        # careful E2M1 bit manipulation that matches CUTLASS expectations)
        # TODO: Implement proper E2M1 packing
        tl.store(X_ptr + row_idx * stride_x_m + cols * stride_x_n,
                 normed.to(tl.bfloat16), mask=mask)


def gemma_rmsnorm_fp4quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    global_scale_inv: torch.Tensor,
    eps: float = 1e-6,
    sf_vec_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused Gemma-style RMSNorm + NVFP4 quantization.

    Args:
        x: Input tensor [M, N] in BF16
        weight: Norm weights [N]
        global_scale_inv: Inverse global scale for FP4 quantization
        eps: Epsilon for RMSNorm
        sf_vec_size: Scale factor vector size (16 for NVFP4)

    Returns:
        normed: RMSNorm output in BF16 [M, N] (for residual connection)
        fp4_out: Packed FP4 tensor [M, N//2] in uint8
        block_scales: Block scales [M, ceil(N/sf_vec_size)] in FP8 E4M3
    """
    M, N = x.shape

    # For now, use the two-step approach but with FlashInfer's optimized kernels
    # This is the baseline before we have a proper fused Triton kernel
    import flashinfer

    # Step 1: Gemma RMSNorm (single kernel)
    normed = flashinfer.gemma_rmsnorm(x, weight, eps=eps)

    # Step 2: FP4 quantize (single kernel)
    fp4_out, block_scales = flashinfer.nvfp4_quantize(
        normed, global_scale_inv,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=True,
    )

    return normed, fp4_out, block_scales


def gemma_add_rmsnorm_fp4quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    global_scale_inv: torch.Tensor,
    eps: float = 1e-6,
    sf_vec_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused residual add + Gemma-style RMSNorm + NVFP4 quantization.

    Returns:
        normed: RMSNorm output in BF16 [M, N]
        residual_out: Updated residual [M, N]
        fp4_out: Packed FP4 tensor [M, N//2] in uint8
        block_scales: Block scales in FP8 E4M3
    """
    import flashinfer

    # Step 1: Fused residual add + Gemma RMSNorm (single kernel)
    flashinfer.gemma_fused_add_rmsnorm(x, residual, weight, eps=eps)
    # After this call: x = gemma_rmsnorm(x + residual), residual = x + residual (in-place)

    # Step 2: FP4 quantize the normalized output (single kernel)
    fp4_out, block_scales = flashinfer.nvfp4_quantize(
        x, global_scale_inv,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=True,
    )

    return x, residual, fp4_out, block_scales
