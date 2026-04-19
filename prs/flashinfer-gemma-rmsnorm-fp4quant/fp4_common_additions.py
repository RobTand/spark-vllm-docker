"""
Additions to flashinfer/cute_dsl/fp4_common.py for Gemma-style RMSNorm support.

Add these functions alongside the existing bfloat2_mul_8 and half2_mul_8.
"""


# BFloat16x2 representation of (1.0, 1.0): 0x3F803F80
# BFloat16 1.0 = 0x3F80, packed pair = 0x3F803F80
BFLOAT2_ONE = 0x3F803F80

# Float16x2 representation of (1.0, 1.0): 0x3C003C00
# Float16 1.0 = 0x3C00, packed pair = 0x3C003C00
HALF2_ONE = 0x3C003C00


@cute.jit
def bfloat2_add_one_8(w_h2: cute.Tensor) -> cute.Tensor:
    """Add 1.0 to each element of 8 bfloat2 pairs.

    Transforms w to (1 + w) for Gemma-style RMSNorm: x * (1 + w).
    Uses PTX add.bf16x2 for each pair.
    """
    one = cutlass.Uint32(BFLOAT2_ONE)
    result = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        result[i] = bfloat2_add(w_h2[i], one)
    return result


@cute.jit
def half2_add_one_8(w_h2: cute.Tensor) -> cute.Tensor:
    """Add 1.0 to each element of 8 half2 pairs.

    Transforms w to (1 + w) for Gemma-style RMSNorm: x * (1 + w).
    Uses PTX add.f16x2 for each pair.
    """
    one = cutlass.Uint32(HALF2_ONE)
    result = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        result[i] = half2_add(w_h2[i], one)
    return result
