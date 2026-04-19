#!/usr/bin/env python3
"""
dequant_triton.py — Triton kernel for N-bit weight dequantization.

Loads packed N-bit integer values from global memory, dequantizes using
per-group scales, and writes FP8/BF16 output. This is the "expand" step
in the sub-byte → tensor-core pipeline:

    VRAM (packed N-bit) → registers (dequant) → output (FP8 or BF16)

For a fused dequant+matmul kernel, the output step would feed the tensor
core directly instead of writing back to memory. This unfused version
validates correctness and measures dequant overhead separately.

Usage:
    python3 dequant_triton.py  # runs self-test + benchmark
"""
import torch
import triton
import triton.language as tl
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pack_utils import pack_Nbit_tensor, unpack_Nbit_tensor


# ---------------------------------------------------------------------------
# Triton kernel: unpack N-bit values and dequantize
# ---------------------------------------------------------------------------

@triton.jit
def dequant_Nbit_kernel(
    packed_ptr,       # pointer to packed uint8 data
    scales_ptr,       # pointer to fp32 scales (n_rows, n_groups)
    output_ptr,       # pointer to output bf16 tensor (n_rows, n_cols)
    n_rows,           # number of rows
    n_cols,           # number of columns
    n_bits: tl.constexpr,          # bits per value
    group_size: tl.constexpr,      # values per group (16)
    BLOCK_ROW: tl.constexpr,       # tile rows
    BLOCK_COL: tl.constexpr,       # tile cols (must be multiple of group_size)
):
    """Dequantize packed N-bit weights to bf16.

    Each program handles a BLOCK_ROW × BLOCK_COL tile of the output.
    Packed data is loaded as bytes, unpacked via shifts+masks, and
    scaled by per-group fp32 scales.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_start = pid_row * BLOCK_ROW
    col_start = pid_col * BLOCK_COL

    n_groups_total = n_cols // group_size

    for row_off in range(BLOCK_ROW):
        row = row_start + row_off
        if row >= n_rows:
            continue

        for col_off in range(0, BLOCK_COL, group_size):
            col = col_start + col_off
            if col >= n_cols:
                continue

            # Which group is this?
            group_idx = col // group_size
            scale = tl.load(scales_ptr + row * n_groups_total + group_idx)

            # Compute bit offset into the packed data for this row
            # Each row has n_cols values, each n_bits bits
            row_bit_offset = row * n_cols * n_bits

            # Dequantize each value in the group
            qmax_plus_1 = 1 << (n_bits - 1)
            qmax = qmax_plus_1 - 1

            for g in range(group_size):
                val_idx = col + g
                if val_idx >= n_cols:
                    continue
                bit_pos = row_bit_offset + val_idx * n_bits

                # Extract n_bits from the packed byte stream
                byte_idx = bit_pos // 8
                bit_in_byte = bit_pos % 8

                # Read up to 2 bytes to cover the value
                b0 = tl.load(packed_ptr + byte_idx).to(tl.int32)
                v = (b0 >> bit_in_byte)

                if bit_in_byte + n_bits > 8:
                    b1 = tl.load(packed_ptr + byte_idx + 1).to(tl.int32)
                    v = v | (b1 << (8 - bit_in_byte))

                if bit_in_byte + n_bits > 16:
                    b2 = tl.load(packed_ptr + byte_idx + 2).to(tl.int32)
                    v = v | (b2 << (16 - bit_in_byte))

                v = v & ((1 << n_bits) - 1)

                # Unsigned → signed: subtract offset
                q_signed = v - qmax_plus_1

                # Dequantize
                dequant = q_signed.to(tl.float32) * scale

                # Store
                out_idx = row * n_cols + val_idx
                tl.store(output_ptr + out_idx, dequant.to(tl.bfloat16))


def triton_dequant(packed: torch.Tensor, scales: torch.Tensor,
                   n_bits: int, out_features: int, in_features: int,
                   group_size: int = 16) -> torch.Tensor:
    """Python wrapper for the Triton dequant kernel."""
    output = torch.empty(out_features, in_features, dtype=torch.bfloat16,
                         device=packed.device)

    BLOCK_ROW = 4
    BLOCK_COL = min(128, in_features)
    grid = (
        (out_features + BLOCK_ROW - 1) // BLOCK_ROW,
        (in_features + BLOCK_COL - 1) // BLOCK_COL,
    )

    dequant_Nbit_kernel[grid](
        packed, scales, output,
        out_features, in_features,
        n_bits, group_size,
        BLOCK_ROW, BLOCK_COL,
    )
    return output


# ---------------------------------------------------------------------------
# PyTorch reference implementation (for correctness check)
# ---------------------------------------------------------------------------

def torch_dequant(packed: torch.Tensor, scales: torch.Tensor,
                  n_bits: int, out_features: int, in_features: int,
                  group_size: int = 16) -> torch.Tensor:
    """Reference dequant using CPU pack_utils."""
    return unpack_Nbit_tensor(
        packed.cpu(), scales.cpu(), n_bits,
        out_features, in_features, group_size
    ).to(torch.bfloat16).to(packed.device)


# ---------------------------------------------------------------------------
# Self-test + benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required for Triton kernels")
        exit(1)

    torch.manual_seed(42)
    device = "cuda"

    print("Correctness tests:")
    for n_bits in [5, 6, 7, 10, 12]:
        w = torch.randn(256, 512, device=device)
        packed, scales = pack_Nbit_tensor(w.cpu(), n_bits, group_size=16)
        packed_gpu = packed.to(device)
        scales_gpu = scales.to(device)

        # Reference
        ref = torch_dequant(packed_gpu, scales_gpu, n_bits, 256, 512)
        # Triton
        out = triton_dequant(packed_gpu, scales_gpu, n_bits, 256, 512)

        max_err = (ref.float() - out.float()).abs().max().item()
        print(f"  {n_bits:>2}-bit: max error = {max_err:.6f} "
              f"{'✓' if max_err < 0.01 else '✗ FAILED'}")

    print("\nBenchmark (1024×4096, 100 iterations):")
    w = torch.randn(1024, 4096, device=device)
    for n_bits in [5, 6, 7, 8, 10, 12]:
        packed, scales = pack_Nbit_tensor(w.cpu(), n_bits, group_size=16)
        packed_gpu = packed.to(device)
        scales_gpu = scales.to(device)

        # Warmup
        for _ in range(5):
            if n_bits in (4, 8, 16):
                out = w.to(torch.bfloat16)
            else:
                out = triton_dequant(packed_gpu, scales_gpu, n_bits, 1024, 4096)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(100):
            out = triton_dequant(packed_gpu, scales_gpu, n_bits, 1024, 4096)
        torch.cuda.synchronize()
        dt = time.time() - t0

        bw = packed_gpu.numel() * 100 / dt / 1e9
        print(f"  {n_bits:>2}-bit: {dt*10:.2f} ms/iter, "
              f"packed size {packed_gpu.numel()/1e3:.0f}KB, "
              f"effective BW {bw:.1f} GB/s")
