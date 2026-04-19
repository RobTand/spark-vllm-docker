#!/usr/bin/env python3
"""
dequant_gpu.py — GPU-accelerated N-bit weight dequantization using PyTorch ops.

Fast vectorized implementation using bitwise operations on GPU tensors.
Supports all bit widths 1-15. This is the reference implementation;
a fused Triton/CUDA kernel would be the production target.

The dequant pipeline:
    packed uint8 bytes → extract N-bit codes → apply per-group scale → bf16/fp8

Usage:
    python3 dequant_gpu.py  # runs self-test + benchmark
"""
import torch
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pack_utils import pack_Nbit_tensor


def dequant_Nbit_gpu(
    packed: torch.Tensor,
    scales: torch.Tensor,
    n_bits: int,
    out_features: int,
    in_features: int,
    group_size: int = 16,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize packed N-bit weights on GPU.

    Vectorized bit extraction using torch bitwise ops. No loops over
    individual values — everything is batched tensor operations.

    Args:
        packed: (n_bytes,) uint8 tensor on GPU
        scales: (out_features, n_groups) fp32 tensor on GPU
        n_bits: bits per value (1-15)
        out_features, in_features: weight dimensions
        group_size: values per scale group
        output_dtype: bf16 or fp8

    Returns:
        weight: (out_features, in_features) tensor in output_dtype
    """
    device = packed.device
    n_values = out_features * in_features

    # Step 1: Expand packed bytes to a flat bit stream
    # Each byte → 8 bits, stored as int32 for easy manipulation
    packed_int = packed.to(torch.int32)
    # Create bit positions 0..7 for each byte
    bit_shifts = torch.arange(8, device=device, dtype=torch.int32)
    # (n_bytes, 8) → each row is the 8 bits of that byte
    all_bits = ((packed_int.unsqueeze(1) >> bit_shifts) & 1)
    # Flatten to a 1D bit stream
    bit_stream = all_bits.reshape(-1)[:n_values * n_bits]

    # Step 2: Group consecutive n_bits into integer codes
    # Reshape to (n_values, n_bits), multiply by powers of 2, sum
    bit_groups = bit_stream.view(n_values, n_bits)
    powers = (1 << torch.arange(n_bits, device=device, dtype=torch.int32))
    codes = (bit_groups * powers).sum(dim=-1)  # (n_values,) unsigned codes

    # Step 3: Convert unsigned → signed and dequantize
    if n_bits == 1:
        # 1-bit: code 0 = -1, code 1 = +1
        signed = codes.float() * 2 - 1
    else:
        qmax_plus_1 = 1 << (n_bits - 1)
        signed = (codes - qmax_plus_1).float()

    # Step 4: Reshape and apply per-group scales
    n_groups = in_features // group_size
    signed = signed.view(out_features, n_groups, group_size)
    # scales: (out_features, n_groups) → (out_features, n_groups, 1)
    dequant = signed * scales.unsqueeze(-1)

    return dequant.reshape(out_features, in_features).to(output_dtype)


# ---------------------------------------------------------------------------
# Self-test + benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required")
        exit(1)

    torch.manual_seed(42)
    device = "cuda"

    print("Correctness tests (all supported bit widths):")
    for n_bits in range(1, 16):
        w = torch.randn(256, 512, device=device)
        packed, scales = pack_Nbit_tensor(w.cpu(), n_bits, group_size=16)
        packed_gpu = packed.to(device)
        scales_gpu = scales.to(device)

        out = dequant_Nbit_gpu(packed_gpu, scales_gpu, n_bits, 256, 512)

        # Check vs original (round-trip error)
        mse = (w.float() - out.float()).pow(2).mean().item()
        # Check the codes are correct by comparing to CPU reference
        from pack_utils import unpack_Nbit_tensor
        ref = unpack_Nbit_tensor(packed, scales.cpu(), n_bits, 256, 512).to(device)
        max_err = (ref.float() - out.float()).abs().max().item()
        status = "✓" if max_err < 0.01 else "✗"
        print(f"  {n_bits:>2}-bit: MSE={mse:.6f}, max_ref_err={max_err:.6f} {status}")

    print(f"\nBenchmark (2048×4096, 100 iterations):")
    w = torch.randn(2048, 4096, device=device)

    # BF16 baseline (just a memcpy)
    w_bf16 = w.to(torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = w_bf16.clone()
    torch.cuda.synchronize()
    dt_bf16 = time.time() - t0
    print(f"  bf16 clone: {dt_bf16*10:.2f} ms/iter ({w_bf16.numel()*2/1e6:.0f} MB)")

    for n_bits in [5, 6, 7, 10, 12, 13, 14, 15]:
        packed, scales = pack_Nbit_tensor(w.cpu(), n_bits, group_size=16)
        packed_gpu = packed.to(device)
        scales_gpu = scales.to(device)

        # Warmup
        for _ in range(3):
            out = dequant_Nbit_gpu(packed_gpu, scales_gpu, n_bits, 2048, 4096)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(100):
            out = dequant_Nbit_gpu(packed_gpu, scales_gpu, n_bits, 2048, 4096)
        torch.cuda.synchronize()
        dt = time.time() - t0

        packed_mb = packed_gpu.numel() / 1e6
        output_mb = out.numel() * 2 / 1e6
        print(f"  {n_bits:>2}-bit: {dt*10:.2f} ms/iter, "
              f"packed {packed_mb:.1f} MB → {output_mb:.0f} MB bf16, "
              f"expansion {output_mb/packed_mb:.1f}×")
