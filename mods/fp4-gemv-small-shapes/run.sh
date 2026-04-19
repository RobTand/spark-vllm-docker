#!/bin/bash
set -e

# Copy the Triton GEMV kernel to a location importable by vLLM
cp fp4_gemv_kernel.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp4_gemv_kernel.py

NVFP4_FILE="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py"

# 1. Store unswizzled weight scales during prepare_weights_for_nvfp4_cutlass
# Add a line after swizzling to save the original
python3 -c "
import re

with open('$NVFP4_FILE', 'r') as f:
    content = f.read()

# In prepare_weights_for_nvfp4_cutlass, save original scales before swizzling
old = 'swizzled_weight_scale = swizzle_blockscale(weight_scale)'
new = '''swizzled_weight_scale = swizzle_blockscale(weight_scale)
    # Store linear-layout scales for GEMV fast path
    _linear_weight_scale = weight_scale.clone()'''
content = content.replace(old, new)

# Return the linear scales along with the rest
old = 'return padded_weight, swizzled_weight_scale, weights_padding_cols'
new = 'return padded_weight, swizzled_weight_scale, weights_padding_cols, _linear_weight_scale'
content = content.replace(old, new)

# Update the caller to store linear scales on the layer
old = '''        weight, weight_scale, weights_padding_cols = prepare_weights_for_nvfp4_cutlass(
            layer.weight.data, layer.weight_scale.data
        )
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols'''
new = '''        weight, weight_scale, weights_padding_cols, linear_weight_scale = prepare_weights_for_nvfp4_cutlass(
            layer.weight.data, layer.weight_scale.data
        )
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols
        # Store linear-layout scales for GEMV fast path (small overhead: N * K/16 bytes)
        layer.weight_scale_linear = linear_weight_scale'''
content = content.replace(old, new)

# 2. Add GEMV dispatch to apply_nvfp4_linear
# Insert check after the Marlin/Emulation early returns, before the main CUTLASS path
old = '''    output_dtype = x.dtype
    output_shape = [*x.shape[:-1], output_size]

    # Quantize BF16 or FP16 to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = scaled_fp4_quant('''
new = '''    output_dtype = x.dtype
    output_shape = [*x.shape[:-1], output_size]

    # GEMV fast path: for M=1 and small weight matrices, use Triton GEMV
    # which avoids CUTLASS launch overhead and skips activation quantization
    M = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1]
    weight_elements = output_size * input_size
    GEMV_THRESHOLD = 10_000_000  # ~10M elements
    if M <= 1 and weight_elements < GEMV_THRESHOLD and hasattr(layer, \"weight_scale_linear\"):
        from vllm.model_executor.layers.quantization.utils.fp4_gemv_kernel import fp4_gemv
        # W4A16: skip activation quantization, use BF16 input directly
        # alpha = 1/weight_global_scale (no input_global_scale since input is BF16)
        gemv_alpha = torch.reciprocal(weight_global_scale)
        # Use original unpadded weight (trim padding if present)
        w = weight[:output_size]
        w_sf = layer.weight_scale_linear[:output_size].view(torch.uint8)
        out = fp4_gemv(x.view(1, -1), w, w_sf, gemv_alpha)
        if bias is not None:
            out = out + bias
        return out.view(output_shape)

    # Quantize BF16 or FP16 to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = scaled_fp4_quant('''
content = content.replace(old, new)

with open('$NVFP4_FILE', 'w') as f:
    f.write(content)

print('Patched nvfp4_utils.py with GEMV fast path')
"

echo "Applied FP4 GEMV small-shapes optimization"
