#!/usr/bin/env python3
"""
dynaquant_linear.py — Drop-in DynaQuantLinear for vLLM integration.

Replaces nn.Linear with a module that stores weights as N-bit packed integers
and dequantizes on the fly using the fused Triton kernels.

This module is designed to be injected into a vLLM model after weight loading,
or used as a standalone inference module.

Each DynaQuantLinear stores:
    - packed_weight: (ceil(out*in*n_bits/8),) uint8 — packed N-bit codes
    - weight_scales: (out, in//group_size) float32 — per-group scales
    - n_bits: int — bit width for this layer
    - bias: optional (out,) float — original bias

Forward pass:
    y = fused_dequant_gemv(x, packed_weight, weight_scales, n_bits, ...) + bias
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "kernels"))
from pack_utils import pack_Nbit_tensor
from fused_dequant_gemv import dynaquant_linear as _fused_linear


class DynaQuantLinear(nn.Module):
    """Linear layer with N-bit packed weights and fused dequant kernels."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_bits: int,
        group_size: int = 16,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.group_size = group_size

        # Packed weight buffer
        packed_size = (out_features * in_features * n_bits + 7) // 8 + 2  # +2 for OOB guard
        self.register_buffer("packed_weight", torch.zeros(packed_size, dtype=torch.uint8))

        # Per-group scales
        n_groups = in_features // group_size
        self.register_buffer("weight_scales", torch.zeros(out_features, n_groups, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Cache for dequantized weights (optional, for debugging)
        self._weight_cache = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, n_bits: int, group_size: int = 16) -> "DynaQuantLinear":
        """Quantize an existing nn.Linear to N-bit packed format."""
        has_bias = linear.bias is not None
        out_f, in_f = linear.weight.shape

        # Pad in_features to multiple of group_size if needed
        if in_f % group_size != 0:
            pad = group_size - (in_f % group_size)
            w = F.pad(linear.weight.data, (0, pad))
            in_f = in_f + pad
        else:
            w = linear.weight.data

        layer = cls(in_f, out_f, n_bits, group_size, bias=has_bias)

        # Quantize and pack
        packed, scales = pack_Nbit_tensor(w.cpu().float(), n_bits, group_size)
        packed_padded = torch.zeros(layer.packed_weight.shape, dtype=torch.uint8)
        packed_padded[:packed.numel()] = packed
        layer.packed_weight.copy_(packed_padded)
        layer.weight_scales.copy_(scales)

        if has_bias:
            layer.bias.data.copy_(linear.bias.data)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using fused dequant + GEMV/GEMM."""
        input_dtype = x.dtype
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        # Pad input if needed
        if x.shape[-1] < self.in_features:
            x = F.pad(x, (0, self.in_features - x.shape[-1]))

        y = _fused_linear(
            x.float(),
            self.packed_weight,
            self.weight_scales,
            self.n_bits,
            self.out_features,
            self.in_features,
            bias=self.bias,
            group_size=self.group_size,
        )

        # Restore original dtype and shape
        y = y.to(input_dtype)
        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape[:-1], self.out_features)
        return y

    def dequantize(self) -> torch.Tensor:
        """Dequantize weights back to float for debugging."""
        from pack_utils import unpack_Nbit_tensor
        packed = self.packed_weight[:((self.out_features * self.in_features * self.n_bits + 7) // 8)]
        return unpack_Nbit_tensor(
            packed.cpu(), self.weight_scales.cpu(),
            self.n_bits, self.out_features, self.in_features, self.group_size
        )

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bits={self.n_bits}, group={self.group_size}, "
                f"packed={self.packed_weight.numel()}B")


def quantize_model_dynaquant(
    model: nn.Module,
    recipe: dict,
    group_size: int = 16,
) -> nn.Module:
    """Replace Linear layers in a model with DynaQuantLinear per recipe.

    Args:
        model: the model to quantize (modified in-place)
        recipe: {param_name: n_bits} mapping, e.g.
                {"model.layers.0.mlp.gate_proj.weight": 6}
        group_size: quantization group size

    Returns:
        model with Linear layers replaced
    """
    # Build module name → n_bits mapping
    mod_bits = {}
    for param_name, bits in recipe.items():
        mod_name = param_name.replace(".weight", "")
        if bits < 16:
            mod_bits[mod_name] = bits

    replaced = 0
    for name, module in model.named_modules():
        if name in mod_bits and isinstance(module, nn.Linear):
            bits = mod_bits[name]
            dq = DynaQuantLinear.from_linear(module, bits, group_size)

            # Replace in parent
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], dq.to(module.weight.device))
            else:
                setattr(model, name, dq.to(module.weight.device))
            replaced += 1

    print(f"[dynaquant] replaced {replaced} linears with DynaQuantLinear")
    return model


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test from_linear
    for n_bits in [3, 5, 6, 7, 8]:
        linear = nn.Linear(512, 256, bias=True).to(device)
        dq = DynaQuantLinear.from_linear(linear, n_bits).to(device)

        x = torch.randn(4, 512, device=device)
        ref = linear(x)
        out = dq(x)

        # The outputs won't match exactly due to quantization error,
        # but they should be correlated
        cos_sim = F.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0)
        print(f"  {n_bits}-bit: cos_sim={cos_sim:.4f}, "
              f"packed={dq.packed_weight.numel()}B vs {256*512*2}B bf16 "
              f"({dq.packed_weight.numel()/(256*512*2)*100:.0f}%)")
