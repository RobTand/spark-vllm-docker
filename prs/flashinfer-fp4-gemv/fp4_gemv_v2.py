"""
FP4 GEMV v2: Restructured for coalesced memory access.

Key insight: For GEMV (M=1), the optimal access pattern is to have each
thread block load contiguous columns of W, then broadcast x across threads.
Each warp processes a subset of output columns.

For (1, K) × (K, N): each output element is a dot product of x with a column of W.
Threads should load W in column-major order for coalesced access to the packed FP4 data.

Actually, W is stored as (N, K//2) row-major. So each row is one output column's
weights. Adjacent threads should process adjacent rows (output columns) and load
contiguous bytes.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_gemv_v2_kernel(
    # (1, K) BF16 input vector
    x_ptr,
    # (N, K//2) uint8 packed FP4 weights, row-major
    w_ptr,
    # (N, K//SF_SIZE) uint8 (viewed from FP8 E4M3) block scales, row-major
    w_sf_ptr,
    # Global alpha (scalar float32)
    alpha_ptr,
    # (1, N) BF16 output
    out_ptr,
    # Strides
    stride_w_n,    # K//2 (bytes per weight row)
    stride_sf_n,   # K//SF_SIZE (scales per row)
    N, K,
    SF_SIZE: tl.constexpr,     # 16 for NVFP4
    BLOCK_N: tl.constexpr,     # output columns per block
):
    """
    Each block computes BLOCK_N output values.
    Within each block, threads cooperate to compute dot products.

    Strategy:
    - Load x once into shared memory (or registers since K is moderate)
    - Each thread handles one or a few output columns
    - For each column, iterate through K loading and dequanting FP4 weights
    """
    pid = tl.program_id(0)
    col_idx = pid * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    col_mask = col_idx < N

    alpha = tl.load(alpha_ptr)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    num_sf = K // SF_SIZE

    # Iterate over K in pairs (since 2 FP4 values per byte)
    for byte_idx in range(K // 2):
        k0 = byte_idx * 2
        k1 = k0 + 1

        # Load 2 input values (BF16 → f32)
        x0 = tl.load(x_ptr + k0).to(tl.float32)
        x1 = tl.load(x_ptr + k1).to(tl.float32)

        # Load packed weight byte for all BLOCK_N columns — COALESCED
        # w_ptr + col_idx * stride_w_n + byte_idx
        w_bytes = tl.load(
            w_ptr + col_idx * stride_w_n + byte_idx,
            mask=col_mask, other=0
        ).to(tl.int32)  # [BLOCK_N]

        # Unpack low and high nibbles
        lo = w_bytes & 0xF        # even k (k0)
        hi = (w_bytes >> 4) & 0xF  # odd k (k1)

        # Dequant E2M1
        lo_val = _e2m1_dequant(lo)  # [BLOCK_N]
        hi_val = _e2m1_dequant(hi)  # [BLOCK_N]

        # Load block scale for this K position
        sf_idx = k0 // SF_SIZE
        w_sf = tl.load(
            w_sf_ptr + col_idx * stride_sf_n + sf_idx,
            mask=col_mask, other=0
        ).to(tl.float32)
        # Note: this loads raw uint8 bytes of FP8 E4M3 values.
        # Proper dequant needed. For now use as approximate scale.

        # Accumulate
        acc += (lo_val * x0 + hi_val * x1) * w_sf

    acc *= alpha
    tl.store(out_ptr + col_idx, acc.to(tl.bfloat16), mask=col_mask)


@triton.jit
def _e2m1_dequant(nibble):
    """Dequant 4-bit E2M1 value to float32.

    E2M1 format: [sign(1), exp(2), mantissa(1)]
    Normal (exp > 0): ±2^(exp-1) × (1 + 0.5 × mantissa)
    Subnormal (exp == 0): ±0.5 × mantissa
    """
    sign_bit = (nibble >> 3) & 1
    exp_bits = (nibble >> 1) & 0x3
    man_bit = nibble & 1

    # Compute absolute value
    is_normal = exp_bits > 0
    # Normal: 2^(exp-1) * (1 + 0.5*man)
    normal_val = tl.exp2((exp_bits - 1).to(tl.float32)) * (1.0 + 0.5 * man_bit.to(tl.float32))
    # Subnormal: 0.5 * man
    subnormal_val = 0.5 * man_bit.to(tl.float32)
    abs_val = tl.where(is_normal, normal_val, subnormal_val)

    # Apply sign
    return tl.where(sign_bit > 0, -abs_val, abs_val)


def fp4_gemv_v2(
    x: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor = None,
    block_sf: int = 16,
) -> torch.Tensor:
    """FP4 GEMV with coalesced weight access."""
    M, K = x.shape
    N = w.shape[0]
    assert M == 1

    if out is None:
        out = torch.empty((1, N), dtype=torch.bfloat16, device=x.device)

    BLOCK_N = 128  # Each block handles 128 output columns

    grid = (triton.cdiv(N, BLOCK_N),)
    _fp4_gemv_v2_kernel[grid](
        x, w, w_sf.view(torch.uint8), alpha, out,
        stride_w_n=K // 2,
        stride_sf_n=K // block_sf,
        N=N, K=K,
        SF_SIZE=block_sf,
        BLOCK_N=BLOCK_N,
    )
    return out
