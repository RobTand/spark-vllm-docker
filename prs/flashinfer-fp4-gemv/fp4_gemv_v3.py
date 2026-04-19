"""
FP4 GEMV v3: Properly vectorized with blocked K iteration.

Key change: each thread block processes ONE output column (or a few),
iterating over K in large vectorized blocks. This matches the memory
access pattern to how Triton actually works — 2D loads across threads.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_gemv_v3_kernel(
    x_ptr,        # (K,) BF16 input
    w_ptr,        # (N, K//2) uint8 packed FP4
    w_sf_ptr,     # (N, K//SF) uint8 (FP8 E4M3 viewed)
    alpha_ptr,    # scalar f32
    out_ptr,      # (N,) BF16 output
    N, K,
    stride_w: tl.constexpr,    # K//2
    stride_sf: tl.constexpr,   # K//SF
    SF_SIZE: tl.constexpr,     # 16
    BLOCK_K_BYTES: tl.constexpr,  # bytes per K iteration (must be multiple of SF_SIZE//2)
):
    """
    Each program handles one output column.
    Grid: (N,)
    """
    col = tl.program_id(0)
    if col >= N:
        return

    acc = tl.zeros([], dtype=tl.float32)
    BLOCK_K = BLOCK_K_BYTES * 2  # 2 FP4 values per byte

    w_row_ptr = w_ptr + col * stride_w
    sf_row_ptr = w_sf_ptr + col * stride_sf

    for byte_start in range(0, K // 2, BLOCK_K_BYTES):
        byte_offs = byte_start + tl.arange(0, BLOCK_K_BYTES)
        byte_mask = byte_offs < (K // 2)

        # Load BLOCK_K_BYTES weight bytes — COALESCED within warp
        w_bytes = tl.load(w_row_ptr + byte_offs, mask=byte_mask, other=0).to(tl.int32)

        # Unpack to 2*BLOCK_K_BYTES values
        lo = w_bytes & 0xF
        hi = (w_bytes >> 4) & 0xF

        # Dequant both nibbles
        lo_f = _e2m1_dequant_vec(lo)
        hi_f = _e2m1_dequant_vec(hi)

        # Load corresponding input values
        k_lo = byte_start * 2 + tl.arange(0, BLOCK_K_BYTES) * 2
        k_hi = k_lo + 1
        x_lo = tl.load(x_ptr + k_lo, mask=byte_mask, other=0.0).to(tl.float32)
        x_hi = tl.load(x_ptr + k_hi, mask=byte_mask, other=0.0).to(tl.float32)

        # Load block scales for this K range
        # Each scale covers SF_SIZE=16 FP4 elements = 8 bytes
        # Scale index for each byte pair: (byte_start*2 + arange*2) // SF_SIZE
        sf_offs = (byte_start * 2 + tl.arange(0, BLOCK_K_BYTES) * 2) // SF_SIZE
        sf_raw = tl.load(sf_row_ptr + sf_offs, mask=byte_mask, other=0).to(tl.int32)

        # Decode FP8 E4M3: sign(1) exp(4) man(3), bias=7
        # value = 2^(exp-7) * (1 + man/8) for normal (exp > 0)
        sf_exp = (sf_raw >> 3) & 0xF
        sf_man = sf_raw & 0x7
        sf_vals = tl.exp2((sf_exp - 7).to(tl.float32)) * (1.0 + sf_man.to(tl.float32) / 8.0)
        # Zero out subnormals (exp==0) — rare in practice for scale factors
        sf_vals = tl.where(sf_exp > 0, sf_vals, sf_man.to(tl.float32) / 8.0 * tl.exp2(tl.full(sf_exp.shape, -6.0, dtype=tl.float32)))

        # Accumulate
        acc += tl.sum(lo_f * x_lo * sf_vals + hi_f * x_hi * sf_vals)

    alpha = tl.load(alpha_ptr)
    acc *= alpha
    tl.store(out_ptr + col, acc.to(tl.bfloat16))


@triton.jit
def _e2m1_dequant_vec(nibbles):
    """Vectorized E2M1 dequant."""
    sign_bit = (nibbles >> 3) & 1
    exp_bits = (nibbles >> 1) & 0x3
    man_bit = nibbles & 1

    is_normal = exp_bits > 0
    normal_val = tl.exp2((exp_bits - 1).to(tl.float32)) * (1.0 + 0.5 * man_bit.to(tl.float32))
    subnormal_val = 0.5 * man_bit.to(tl.float32)
    abs_val = tl.where(is_normal, normal_val, subnormal_val)

    return tl.where(sign_bit > 0, -abs_val, abs_val)


def fp4_gemv_v3(
    x: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor = None,
    block_sf: int = 16,
) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    assert M == 1

    if out is None:
        out = torch.empty(N, dtype=torch.bfloat16, device=x.device)

    BLOCK_K_BYTES = min(K // 2, 512)  # Process 512 bytes (1024 FP4 values) per iteration

    grid = (N,)
    _fp4_gemv_v3_kernel[grid](
        x.view(-1), w, w_sf.view(torch.uint8), alpha, out,
        N=N, K=K,
        stride_w=K // 2,
        stride_sf=K // block_sf,
        SF_SIZE=block_sf,
        BLOCK_K_BYTES=BLOCK_K_BYTES,
        num_warps=4,
    )
    return out.unsqueeze(0)
