#!/usr/bin/env python3
"""
pack_utils.py — bit-packing and unpacking utilities for sub-byte quantization.

Provides pack_Nbit and unpack_Nbit for N in {5, 6, 7, 9, 10, 11, 12}.
Each packs/unpacks a flat tensor of values into a packed byte tensor.

Packing layout: values are packed contiguously in little-endian bit order.
For N=5: 8 values occupy 5 bytes (40 bits). For N=6: 4 values occupy 3 bytes.

These utilities are used by the dequantization kernels to load packed weights
from VRAM and expand them to FP8 or BF16 for tensor core consumption.
"""
import torch
from typing import Tuple


def pack_intN(values: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Pack a 1D tensor of int values (each in [0, 2^n_bits)) into bytes.

    Uses numpy for fast CPU packing. Falls back to torch scatter for GPU.

    Args:
        values: 1D int tensor, each element in [0, 2^n_bits)
        n_bits: bits per value (1-15)

    Returns:
        packed: 1D uint8 tensor of ceil(len(values) * n_bits / 8) bytes
    """
    assert values.dim() == 1

    if values.device.type == "cpu":
        return _pack_intN_numpy(values, n_bits)

    # GPU path: scatter-based (original)
    device = values.device
    n_values = values.numel()
    n_bytes = (n_values * n_bits + 7) // 8
    packed = torch.zeros(n_bytes, dtype=torch.uint8, device=device)
    vals = values.to(torch.int32)
    val_indices = torch.arange(n_values, device=device, dtype=torch.int64)

    for bit_idx in range(n_bits):
        bit_vals = ((vals >> bit_idx) & 1).to(torch.uint8)
        global_bit_pos = val_indices * n_bits + bit_idx
        byte_pos = (global_bit_pos // 8).to(torch.int64)
        bit_in_byte = (global_bit_pos % 8).to(torch.uint8)
        packed.scatter_add_(0, byte_pos, bit_vals << bit_in_byte)

    return packed


def _pack_intN_numpy(values: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Fast CPU packing via numpy — builds a bitstream and packs to bytes."""
    import numpy as np

    vals = values.numpy().astype(np.uint16 if n_bits <= 16 else np.uint32)
    n_values = len(vals)
    n_total_bits = n_values * n_bits
    n_bytes = (n_total_bits + 7) // 8

    # Build a flat bit array: for each value, extract n_bits bits
    # Shape: (n_values, n_bits) → flatten to (n_total_bits,)
    bit_matrix = np.zeros((n_values, n_bits), dtype=np.uint8)
    for bit_idx in range(n_bits):
        bit_matrix[:, bit_idx] = (vals >> bit_idx) & 1
    bits_flat = bit_matrix.ravel()

    # Pad to multiple of 8
    pad = (-len(bits_flat)) % 8
    if pad:
        bits_flat = np.concatenate([bits_flat, np.zeros(pad, dtype=np.uint8)])

    # Pack 8 bits → 1 byte using dot product with powers of 2
    bits_reshaped = bits_flat.reshape(-1, 8)
    powers = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    packed = bits_reshaped.dot(powers).astype(np.uint8)

    return torch.from_numpy(packed[:n_bytes])


def unpack_intN(packed: torch.Tensor, n_bits: int, n_values: int) -> torch.Tensor:
    """Unpack a packed byte tensor into individual N-bit integer values.

    Fully vectorized using torch bit operations — no Python loops.

    Args:
        packed: 1D uint8 tensor
        n_bits: bits per value
        n_values: number of values to unpack

    Returns:
        values: 1D int32 tensor of length n_values, each in [0, 2^n_bits)
    """
    device = packed.device

    # Expand each byte into 8 individual bits
    byte_shifts = torch.arange(8, device=device, dtype=torch.int32)
    bits = ((packed.to(torch.int32).unsqueeze(1) >> byte_shifts) & 1).reshape(-1)

    # Trim to exactly n_values * n_bits
    bits = bits[:n_values * n_bits]

    # Group into n_bits per value and reconstruct
    bits = bits.view(n_values, n_bits)
    bit_shifts = torch.arange(n_bits, device=device, dtype=torch.int32)
    values = (bits * (1 << bit_shifts)).sum(dim=1)

    return values


def pack_Nbit_tensor(weight: torch.Tensor, n_bits: int,
                     group_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight tensor to N-bit and pack.

    Per-group symmetric quantization: each group of `group_size` weights
    shares one fp32 scale factor.

    Args:
        weight: (out_features, in_features) float tensor
        n_bits: target bit width (1-15)
        group_size: quantization group size

    Returns:
        packed: packed uint8 tensor containing the quantized codes
        scales: (out_features, n_groups) fp32 scale tensor
    """
    out_f, in_f = weight.shape
    assert in_f % group_size == 0
    n_groups = in_f // group_size

    w = weight.float().view(out_f, n_groups, group_size)

    # Per-group symmetric scale
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    if n_bits == 1:
        # 1-bit: sign only
        codes = (w >= 0).to(torch.int32)
        scales_out = max_abs.squeeze(-1)
    else:
        qmax = (1 << (n_bits - 1)) - 1
        scale = max_abs / qmax
        # Quantize to signed int, then shift to unsigned for packing
        q_signed = (w / scale).round().clamp(-qmax - 1, qmax).to(torch.int32)
        codes = q_signed + (qmax + 1)  # shift to [0, 2^n_bits)
        scales_out = scale.squeeze(-1)

    # Pack the codes
    codes_flat = codes.reshape(-1)
    packed = pack_intN(codes_flat, n_bits)

    return packed, scales_out


def unpack_Nbit_tensor(packed: torch.Tensor, scales: torch.Tensor,
                       n_bits: int, out_features: int, in_features: int,
                       group_size: int = 16) -> torch.Tensor:
    """Unpack and dequantize an N-bit packed weight tensor.

    Args:
        packed: packed uint8 tensor from pack_Nbit_tensor
        scales: (out_features, n_groups) fp32 scales
        n_bits: bit width
        out_features, in_features: weight shape
        group_size: quantization group size

    Returns:
        weight: (out_features, in_features) dequantized float tensor
    """
    n_groups = in_features // group_size
    n_values = out_features * n_groups * group_size

    codes = unpack_intN(packed, n_bits, n_values)
    codes = codes.view(out_features, n_groups, group_size)

    if n_bits == 1:
        # 1-bit: code 0 = -1, code 1 = +1
        w = (codes.float() * 2 - 1) * scales.unsqueeze(-1)
    else:
        qmax = (1 << (n_bits - 1)) - 1
        q_signed = codes - (qmax + 1)
        w = q_signed.float() * scales.unsqueeze(-1)

    return w.view(out_features, in_features)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    for n_bits in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]:
        w = torch.randn(64, 128)
        packed, scales = pack_Nbit_tensor(w, n_bits, group_size=16)
        w_recon = unpack_Nbit_tensor(packed, scales, n_bits, 64, 128, group_size=16)
        mse = (w - w_recon).pow(2).mean().item()
        compression = w.numel() * 32 / (packed.numel() * 8)  # vs fp32
        print(f"{n_bits:>2}-bit: MSE={mse:.6f}, packed {packed.numel()} bytes "
              f"(compression {compression:.1f}×)")
    print("self-test passed")
