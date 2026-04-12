#!/usr/bin/env python3
"""
fused_dequant_gemv.py — Fused N-bit dequantize + GEMV Triton kernel.

For decode (batch=1, memory-bandwidth-bound), this is the critical kernel:
    y = W @ x
where W is stored as packed N-bit integers with per-group FP32 scales.

The kernel reads packed weights from VRAM, dequantizes in registers, and
accumulates the dot product without ever materializing the full BF16 weight
matrix. This means memory traffic = packed weight bytes + scales, which is
N/16 of the bf16 traffic for N-bit storage.

Supports bit widths 1-16.  Values spanning >8 bits use 3-byte extraction.
1-bit uses sign-only dequantization; 2-16 bit uses symmetric integer codes.

Architecture:
    Each thread block handles one output row (one element of y).
    Within the block, threads cooperatively load packed bytes, extract
    N-bit codes, dequantize via per-group scales, multiply by x, and
    reduce to a scalar.

Usage:
    python3 fused_dequant_gemv.py  # runs self-test + benchmark
"""
import torch
import triton
import triton.language as tl
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pack_utils import pack_Nbit_tensor, unpack_Nbit_tensor


@triton.jit
def _fused_dequant_gemv_kernel(
    # Pointers
    y_ptr,          # output: (M,) or (M, N) for batched
    x_ptr,          # input: (K,) or (N, K) for batched
    packed_ptr,     # packed weights: flat uint8 array, row-major
    scales_ptr,     # scales: (M, K // group_size) fp32
    # Dimensions
    M,              # output features (rows of W)
    K,              # input features (cols of W)
    N,              # batch size
    # Strides
    stride_x_n,     # stride between batch elements in x
    stride_y_n,     # stride between batch elements in y
    # Quantization params
    n_bits: tl.constexpr,
    group_size: tl.constexpr,
    # Tile sizes
    BLOCK_K: tl.constexpr,   # how many K elements per iteration
):
    """Fused dequant + GEMV: y[m] = sum_k( dequant(W[m,k]) * x[k] )

    Each program handles one row m and one batch element n.
    Loops over K in chunks of BLOCK_K, dequanting and accumulating.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m >= M:
        return

    n_groups = K // group_size
    qmax_plus_1 = 1 << (n_bits - 1)

    # Accumulator (scalar)
    acc = tl.zeros([], dtype=tl.float32)

    # Bit offset for the start of this row in the packed data
    # Use int64 to avoid overflow for large matrices (M*K*n_bits > 2^31)
    row_bit_start = pid_m.to(tl.int64) * K * n_bits

    # Loop over K in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask = k_offsets < K

        # Load x values for this chunk
        x_vals = tl.load(x_ptr + pid_n * stride_x_n + k_offsets, mask=mask, other=0.0)

        # Load scales for groups that overlap this chunk
        group_indices = k_offsets // group_size
        s = tl.load(scales_ptr + pid_m * n_groups + group_indices, mask=mask, other=1.0)

        # Extract N-bit codes from packed bytes (int64 for large matrices)
        bit_positions = row_bit_start + k_offsets.to(tl.int64) * n_bits
        byte_indices = bit_positions // 8
        bit_in_byte = (bit_positions % 8).to(tl.int32)

        # Load bytes — up to 3 bytes per value for 9-16 bit extraction
        packed_bytes_total = (M * K * n_bits + 7) // 8
        b0 = tl.load(packed_ptr + byte_indices, mask=mask, other=0).to(tl.int32)
        b1_mask = mask & ((byte_indices + 1) < packed_bytes_total)
        b1 = tl.load(packed_ptr + byte_indices + 1, mask=b1_mask, other=0).to(tl.int32)

        # Extract: shift b0 right, OR with b1 shifted left, mask to n_bits
        raw = (b0 >> bit_in_byte) | (b1 << (8 - bit_in_byte))

        # For n_bits > 8: a value can span 3 bytes (bit_in_byte + n_bits > 16)
        if n_bits > 8:
            b2_mask = mask & ((byte_indices + 2) < packed_bytes_total)
            b2 = tl.load(packed_ptr + byte_indices + 2, mask=b2_mask, other=0).to(tl.int32)
            raw = raw | (b2 << (16 - bit_in_byte))

        codes = raw & ((1 << n_bits) - 1)

        # Dequantize: 1-bit is sign-only, 2+ bit is symmetric integer
        if n_bits == 1:
            w_vals = (codes.to(tl.float32) * 2.0 - 1.0) * s
        else:
            q_signed = (codes - qmax_plus_1).to(tl.float32)
            w_vals = q_signed * s
        prod = w_vals * x_vals.to(tl.float32)

        # Accumulate
        acc = acc + tl.sum(prod, axis=0)

    # Store output
    tl.store(y_ptr + pid_n * stride_y_n + pid_m, acc)


def fused_dequant_gemv(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    n_bits: int,
    out_features: int,
    in_features: int,
    group_size: int = 16,
) -> torch.Tensor:
    """Fused N-bit dequant + matrix-vector multiply.

    Args:
        x: input tensor, shape (K,) or (N, K)
        packed: packed uint8 weights for all rows, flat
        scales: (out_features, in_features // group_size) fp32
        n_bits: bits per weight value
        out_features: M (rows of weight matrix)
        in_features: K (cols of weight matrix)
        group_size: quantization group size

    Returns:
        y: (M,) or (N, M) float32
    """
    batched = x.dim() == 2
    if not batched:
        x = x.unsqueeze(0)  # (1, K)

    N, K = x.shape
    M = out_features
    assert K == in_features

    y = torch.empty(N, M, dtype=torch.float32, device=x.device)

    BLOCK_K = min(256, triton.next_power_of_2(K))

    # Pad packed buffer to avoid OOB reads at byte boundaries
    needed = (M * K * n_bits + 7) // 8
    if packed.numel() < needed + 4:
        packed = torch.nn.functional.pad(packed, (0, 4))

    grid = (M, N)
    _fused_dequant_gemv_kernel[grid](
        y, x, packed, scales,
        M, K, N,
        x.stride(0), y.stride(0),
        n_bits, group_size,
        BLOCK_K,
    )

    if not batched:
        y = y.squeeze(0)
    return y


# ---------------------------------------------------------------------------
# Also provide a fused dequant+GEMM for small batch (prefill)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_dequant_gemm_kernel(
    # Pointers
    c_ptr,          # output: (N, M)
    a_ptr,          # input: (N, K) — the activations
    packed_ptr,     # packed weights: W is (M, K), stored as packed N-bit
    scales_ptr,     # scales: (M, K // group_size) fp32
    # Dimensions
    N, M, K,
    # Strides
    stride_an, stride_ak,
    stride_cn, stride_cm,
    # Quantization params
    n_bits: tl.constexpr,
    group_size: tl.constexpr,
    # Tile sizes
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused dequant + GEMM: C[n,m] = sum_k( A[n,k] * dequant(W[m,k]) )

    Each program handles a BLOCK_N × BLOCK_M tile of the output.
    Loops over K in chunks of BLOCK_K.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    m_start = pid_m * BLOCK_M

    n_groups = K // group_size
    qmax_plus_1 = 1 << (n_bits - 1)

    # Accumulator tile: (BLOCK_N, BLOCK_M)
    acc = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)

    n_offsets = n_start + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    m_offsets = m_start + tl.arange(0, BLOCK_M)  # (BLOCK_M,)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)

        # Load A tile: (BLOCK_N, BLOCK_K)
        a_mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        a_tile = tl.load(
            a_ptr + n_offsets[:, None] * stride_an + k_offsets[None, :] * stride_ak,
            mask=a_mask, other=0.0
        ).to(tl.float32)

        # Dequantize W tile: (BLOCK_M, BLOCK_K) → need to extract per m, k
        # Use int64 for bit positions to avoid overflow for large matrices
        bit_positions = m_offsets[:, None].to(tl.int64) * (K * n_bits) + k_offsets[None, :].to(tl.int64) * n_bits
        byte_indices = bit_positions // 8
        bit_in_byte = (bit_positions % 8).to(tl.int32)

        w_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)

        packed_bytes_total = (M * K * n_bits + 7) // 8
        b0 = tl.load(packed_ptr + byte_indices, mask=w_mask, other=0).to(tl.int32)
        b1_mask = w_mask & ((byte_indices + 1) < packed_bytes_total)
        b1 = tl.load(packed_ptr + byte_indices + 1, mask=b1_mask, other=0).to(tl.int32)

        raw = (b0 >> bit_in_byte) | (b1 << (8 - bit_in_byte))

        if n_bits > 8:
            b2_mask = w_mask & ((byte_indices + 2) < packed_bytes_total)
            b2 = tl.load(packed_ptr + byte_indices + 2, mask=b2_mask, other=0).to(tl.int32)
            raw = raw | (b2 << (16 - bit_in_byte))

        codes = raw & ((1 << n_bits) - 1)

        # Load scales for these groups
        group_indices = k_offsets[None, :] // group_size  # (1, BLOCK_K) broadcast
        s = tl.load(
            scales_ptr + m_offsets[:, None] * n_groups + group_indices,
            mask=w_mask, other=1.0
        )

        if n_bits == 1:
            w_tile = (codes.to(tl.float32) * 2.0 - 1.0) * s
        else:
            q_signed = (codes - qmax_plus_1).to(tl.float32)
            w_tile = q_signed * s

        # Matmul accumulate: C += A @ W^T
        # A is (BLOCK_N, BLOCK_K), W is (BLOCK_M, BLOCK_K)
        # C += A @ W^T = (BLOCK_N, BLOCK_K) @ (BLOCK_K, BLOCK_M)
        acc += tl.dot(a_tile, tl.trans(w_tile))

    # Store output tile
    c_mask = (n_offsets[:, None] < N) & (m_offsets[None, :] < M)
    tl.store(
        c_ptr + n_offsets[:, None] * stride_cn + m_offsets[None, :] * stride_cm,
        acc,
        mask=c_mask,
    )


def fused_dequant_gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    n_bits: int,
    out_features: int,
    in_features: int,
    group_size: int = 16,
) -> torch.Tensor:
    """Fused N-bit dequant + GEMM: y = x @ W^T

    Args:
        x: (N, K) input activations
        packed: packed uint8 weights, W is (M, K)
        scales: (M, K // group_size) fp32
        n_bits: bits per weight value
        out_features: M
        in_features: K

    Returns:
        y: (N, M) float32
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    N, K = x.shape
    M = out_features
    assert K == in_features

    y = torch.empty(N, M, dtype=torch.float32, device=x.device)

    # Pad packed buffer by 2 bytes to avoid OOB reads at boundaries
    if packed.numel() < (M * K * n_bits + 7) // 8 + 2:
        packed = torch.nn.functional.pad(packed, (0, 2))

    BLOCK_N = min(32, triton.next_power_of_2(N))
    BLOCK_M = 32
    BLOCK_K = min(64, triton.next_power_of_2(K))

    # BLOCK_K must be multiple of 16 for tl.dot
    if BLOCK_K < 16:
        BLOCK_K = 16

    grid = (
        (N + BLOCK_N - 1) // BLOCK_N,
        (M + BLOCK_M - 1) // BLOCK_M,
    )

    _fused_dequant_gemm_kernel[grid](
        y, x, packed, scales,
        N, M, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        n_bits, group_size,
        BLOCK_N, BLOCK_M, BLOCK_K,
    )

    return y


# ---------------------------------------------------------------------------
# Unified entry point: picks GEMV for batch=1, GEMM for batch>1
# ---------------------------------------------------------------------------

def dynaquant_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    n_bits: int,
    out_features: int,
    in_features: int,
    bias: torch.Tensor = None,
    group_size: int = 16,
) -> torch.Tensor:
    """DynaQuant fused linear: y = dequant(W) @ x + bias

    Automatically selects GEMV (batch=1) or GEMM (batch>1).
    """
    if x.dim() == 1 or (x.dim() == 2 and x.shape[0] == 1):
        y = fused_dequant_gemv(x, packed, scales, n_bits, out_features, in_features, group_size)
    else:
        y = fused_dequant_gemm(x, packed, scales, n_bits, out_features, in_features, group_size)

    if bias is not None:
        y = y + bias

    return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Self-test + benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required")
        exit(1)

    torch.manual_seed(42)
    device = "cuda"

    print("=== Correctness tests (fused dequant+GEMV) ===")
    for n_bits in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]:
        M, K = 256, 512
        w = torch.randn(M, K)
        x = torch.randn(K, device=device)

        packed, scales = pack_Nbit_tensor(w, n_bits, group_size=16)
        packed_gpu = packed.to(device)
        scales_gpu = scales.to(device)

        # Reference: dequant then matmul
        w_deq = unpack_Nbit_tensor(packed, scales, n_bits, M, K, group_size=16)
        ref = (w_deq.to(device).float() @ x.float())

        # Fused
        y = fused_dequant_gemv(x, packed_gpu, scales_gpu, n_bits, M, K)

        err = (ref - y).abs().max().item()
        rel_err = err / ref.abs().max().item()
        print(f"  {n_bits}-bit: max_err={err:.4f}, rel_err={rel_err:.4f} "
              f"{'PASS' if rel_err < 0.02 else 'FAIL'}")

    print("\n=== Correctness tests (fused dequant+GEMM, batch=4) ===")
    for n_bits in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]:
        M, K, N = 256, 512, 4
        w = torch.randn(M, K)
        x = torch.randn(N, K, device=device)

        packed, scales = pack_Nbit_tensor(w, n_bits, group_size=16)
        packed_gpu = packed.to(device)
        scales_gpu = scales.to(device)

        w_deq = unpack_Nbit_tensor(packed, scales, n_bits, M, K, group_size=16)
        ref = x.float() @ w_deq.to(device).float().T

        y = fused_dequant_gemm(x, packed_gpu, scales_gpu, n_bits, M, K)

        err = (ref - y).abs().max().item()
        rel_err = err / ref.abs().max().item()
        print(f"  {n_bits}-bit: max_err={err:.4f}, rel_err={rel_err:.4f} "
              f"{'PASS' if rel_err < 0.02 else 'FAIL'}")

    print("\n=== Benchmark: GEMV (batch=1, 4096×4096) ===")
    M, K = 4096, 4096
    x = torch.randn(K, device=device, dtype=torch.bfloat16)

    # BF16 baseline
    w_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    for _ in range(5):
        _ = w_bf16 @ x
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = w_bf16 @ x
    torch.cuda.synchronize()
    bf16_ms = (time.time() - t0) * 10
    bf16_bw = M * K * 2 * 100 / (time.time() - t0) / 1e9
    print(f"  BF16 baseline: {bf16_ms:.2f} ms, {bf16_bw:.0f} GB/s effective")

    for n_bits in [1, 2, 3, 5, 6, 7, 8, 10, 12, 16]:
        w = torch.randn(M, K)
        packed, scales = pack_Nbit_tensor(w, n_bits, group_size=16)
        packed_gpu = packed.to(device)
        scales_gpu = scales.to(device)

        # Warmup
        for _ in range(5):
            _ = fused_dequant_gemv(x.float(), packed_gpu, scales_gpu, n_bits, M, K)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(100):
            _ = fused_dequant_gemv(x.float(), packed_gpu, scales_gpu, n_bits, M, K)
        torch.cuda.synchronize()
        dt = time.time() - t0

        packed_bytes = packed_gpu.numel()
        bw = packed_bytes * 100 / dt / 1e9
        ms = dt * 10
        speedup = bf16_ms / ms
        print(f"  {n_bits}-bit: {ms:.2f} ms ({speedup:.2f}x vs bf16), "
              f"packed {packed_bytes/1e6:.1f} MB, {bw:.0f} GB/s")
