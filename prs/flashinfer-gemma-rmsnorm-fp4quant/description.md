# FlashInfer PR: Add Gemma-style RMSNorm support to rmsnorm_fp4quant and add_rmsnorm_fp4quant

## Summary

Add a `gemma_style` parameter to `rmsnorm_fp4quant` and `add_rmsnorm_fp4quant` CuTe DSL
kernels to support Gemma-style RMSNorm (`x * (1 + w)` instead of `x * w`).

This enables Qwen3.5 (which uses Gemma-style RMSNorm) to use the fused
RMSNorm+FP4Quant kernel path, eliminating ~96 kernel launches per token
on NVFP4 quantized models.

## Motivation

Qwen3.5-122B-A10B-NVFP4 on DGX Spark (SM121/GB10) achieves ~26 tok/s. Profiling
shows that each of the 48 decoder layers calls `scaled_fp4_quant()` as a separate
CUDA kernel before every NVFP4 GEMM. The RMSNorm preceding these GEMMs is also a
separate kernel. Fusing RMSNorm + FP4 quantization into a single kernel eliminates
one kernel launch per RMSNorm→Linear pair.

The standard `rmsnorm_fp4quant` kernel exists but assumes `x * w` normalization.
Qwen3.5 uses `GemmaRMSNorm` which computes `x * (1 + w)`. Without Gemma-style
support, the fusion pass cannot match these operations.

## Changes

### flashinfer/cute_dsl/rmsnorm_fp4quant.py
- Add `gemma_style: bool` parameter to `RMSNormFP4QuantKernel.__init__`
- In the kernel Phase 3, before the `x * w` multiply, conditionally add 1.0 to
  the weight when `gemma_style=True`:
  - For BFloat16: add 1.0 to each bfloat2 pair before multiply
  - For Float16: add 1.0 to each half2 pair before multiply
- Add `gemma_style` to `_get_compiled_kernel` cache key
- Add `gemma_style` parameter to `rmsnorm_fp4quant()` Python API

### flashinfer/cute_dsl/add_rmsnorm_fp4quant.py
- Same changes for the residual-add variant

### flashinfer/norm/__init__.py
- Add `gemma_rmsnorm_fp4quant` and `gemma_add_rmsnorm_fp4quant` convenience wrappers
  that call the base functions with `gemma_style=True`

### Tests
- Add test cases for Gemma-style in existing rmsnorm_fp4quant tests
- Verify numerical equivalence: `gemma_rmsnorm_fp4quant(x, w)` matches
  `rmsnorm_fp4quant(x, w + 1)` (the latter being a semantic equivalence check)

## Kernel change detail

In `RMSNormFP4QuantKernel.kernel`, Phase 3, the weight multiply section:

```python
# Current (line ~504):
xw_h2 = bfloat2_mul_8(x_h2, w_h2)

# With gemma_style:
if cutlass.const_expr(self.gemma_style):
    w_h2 = bfloat2_add_one_8(w_h2)  # w_h2[i] = w_h2[i] + bf16x2(1.0, 1.0)
xw_h2 = bfloat2_mul_8(x_h2, w_h2)
```

The `bfloat2_add_one_8` helper adds `1.0` to each element of 8 bfloat16x2 pairs
using PTX `add.rn.bf16x2`. This is a compile-time branch so there is zero overhead
when `gemma_style=False`.

## Performance impact

- Eliminates ~96 kernel launches per token on Qwen3.5-122B (2 per layer × 48 layers)
- Each eliminated launch saves ~5-10µs of kernel launch overhead
- Estimated improvement: 0.5-1.0 ms per token at batch=1 (~1-2 tok/s)
- Additionally reduces memory traffic by avoiding the BF16 intermediate between
  RMSNorm output and FP4 quant input
