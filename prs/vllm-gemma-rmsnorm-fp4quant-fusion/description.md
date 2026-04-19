# vLLM PR: Enable RMSNorm+FP4Quant fusion for Gemma-style models (Qwen3.5)

## Summary

Wire up FlashInfer's `gemma_rmsnorm_fp4quant` / `gemma_add_rmsnorm_fp4quant` 
for Qwen3.5 and other models using Gemma-style RMSNorm (`x * (1 + w)`).

This eliminates ~96 separate `scaled_fp4_quant` kernel launches per token on
NVFP4-quantized models by fusing the RMSNorm and FP4 activation quantization
into a single kernel.

## Problem

On Qwen3.5-122B-A10B-NVFP4 (DGX Spark), the decode path for each layer is:
1. RMSNorm kernel → BF16 output
2. `scaled_fp4_quant` kernel → FP4 + block scales
3. CUTLASS FP4 GEMM kernel

Steps 1 and 2 can be fused into a single kernel, but the existing
`fuse_norm_quant` compiler pass only matches the standard `RMSNorm` custom op.
Qwen3.5 uses `GemmaRMSNorm` which is not matched.

## Changes

### Option A: Compiler pass approach (preferred)

1. **Register `GemmaRMSNorm` as a fusion-eligible custom op**
   - In `vllm/model_executor/layers/layernorm.py`, add a `forward_cuda` 
     implementation to `GemmaRMSNorm` that uses FlashInfer's `gemma_rmsnorm` 
     (or `gemma_fused_add_rmsnorm`) instead of torch.compile'd native code
   - Register as custom op `gemma_rms_norm` in the op registry

2. **Add fusion pattern for `gemma_rms_norm` + `scaled_fp4_quant`**
   - In `vllm/compilation/passes/fusion/rms_quant_fusion.py`, add a new pattern
     class `GemmaRMSNormStaticQuantPattern` that matches `gemma_rms_norm` followed
     by `scaled_fp4_quant` and replaces with `gemma_rmsnorm_fp4quant`
   - Similarly for the residual-add variant

3. **Wire up in pass manager**
   - Extend `fuse_norm_quant` to also enable the Gemma variant

### Option B: Direct model patch (simpler, less general)

1. **Modify `GemmaRMSNorm.forward_cuda`** to return both BF16 and FP4 outputs
   when the downstream consumer is an NVFP4 linear layer

2. **Modify `CompressedTensorsW4A4Fp4.apply_weights`** to accept pre-quantized
   FP4 input and skip `scaled_fp4_quant`

## Depends on

- FlashInfer PR adding `gemma_style` parameter to `rmsnorm_fp4quant` and
  `add_rmsnorm_fp4quant`, plus `gemma_rmsnorm_fp4quant` and
  `gemma_add_rmsnorm_fp4quant` convenience APIs.

## Performance impact

- Eliminates ~96 kernel launches per token (2 per layer × 48 layers)
- Reduces memory traffic: no BF16 intermediate between norm and quant
- Estimated: +1-3 tok/s on DGX Spark (SM121) for Qwen3.5-122B NVFP4
- No impact on non-NVFP4 models or non-Gemma-style models
