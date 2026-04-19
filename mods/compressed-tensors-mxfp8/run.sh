#!/bin/bash
set -e

# Patch vLLM + compressed-tensors to support MXFP8 in compressed-tensors checkpoints.
# This adds:
#   1. MXFP8/MXFP8A16 preset schemes to compressed-tensors (if not already present)
#   2. mxfp8-quantized compression format enum
#   3. CompressedTensorsMxfp8 scheme class in vLLM
#   4. _is_mxfp8() detection + dispatch in _get_scheme_from_parts()

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CT_PKG=$(python3 -c "import compressed_tensors; import os; print(os.path.dirname(compressed_tensors.__file__))")
VLLM_CT="$(python3 -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")/model_executor/layers/quantization/compressed_tensors"

echo "compressed-tensors package: $CT_PKG"
echo "vLLM compressed-tensors: $VLLM_CT"

# ── 1. Add MXFP8 schemes to compressed-tensors (idempotent) ──
if ! python3 -c "from compressed_tensors.quantization.quant_scheme import MXFP8" 2>/dev/null; then
    echo "Adding MXFP8/MXFP8A16 schemes to compressed-tensors..."
    python3 << 'PYEOF'
import re, sys

path = sys.argv[1] if len(sys.argv) > 1 else ""
# Find quant_scheme.py
import compressed_tensors.quantization.quant_scheme as qs
path = qs.__file__

with open(path) as f:
    content = f.read()

# Add MXFP8A16 and MXFP8 after MXFP4
mxfp8_defs = '''
MXFP8A16 = dict(
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    )
)

MXFP8 = dict(
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        dynamic=True,
        symmetric=True,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    ),
)
'''

if "MXFP8A16" not in content:
    # Insert after MXFP4 definition block
    # Find the end of MXFP4 dict
    idx = content.find("MXFP4 = dict(")
    if idx >= 0:
        # Find the closing paren of the outer dict
        depth = 0
        i = idx
        while i < len(content):
            if content[i] == '(':
                depth += 1
            elif content[i] == ')':
                depth -= 1
                if depth == 0:
                    # Find end of line
                    eol = content.index('\n', i)
                    content = content[:eol+1] + mxfp8_defs + content[eol+1:]
                    break
            i += 1

# Add to PRESET_SCHEMES
if '"MXFP8"' not in content:
    content = content.replace(
        '"MXFP4": MXFP4,\n}',
        '"MXFP4": MXFP4,\n    "MXFP8A16": MXFP8A16,\n    "MXFP8": MXFP8,\n}'
    )

with open(path, 'w') as f:
    f.write(content)
print(f"Patched {path}")
PYEOF
else
    echo "MXFP8 schemes already present in compressed-tensors"
fi

# ── 2. Add mxfp8-quantized format enum (idempotent) ──
if ! python3 -c "from compressed_tensors.config.base import CompressionFormat; CompressionFormat.mxfp8_quantized" 2>/dev/null; then
    echo "Adding mxfp8-quantized format..."
    python3 << 'PYEOF'
import compressed_tensors.config.base as base_mod
path = base_mod.__file__

with open(path) as f:
    content = f.read()

if "mxfp8_quantized" not in content:
    content = content.replace(
        'mxfp4_pack_quantized = "mxfp4-pack-quantized"',
        'mxfp4_pack_quantized = "mxfp4-pack-quantized"\n    mxfp8_quantized = "mxfp8-quantized"'
    )
    with open(path, 'w') as f:
        f.write(content)
    print(f"Patched {path}")
PYEOF
else
    echo "mxfp8-quantized format already present"
fi

# ── 3. Install CompressedTensorsMxfp8 scheme ──
echo "Installing CompressedTensorsMxfp8 scheme..."
cp "$SCRIPT_DIR/compressed_tensors_mxfp8.py" "$VLLM_CT/schemes/"

# Update __init__.py
INIT="$VLLM_CT/schemes/__init__.py"
if ! grep -q "CompressedTensorsMxfp8" "$INIT" 2>/dev/null; then
    # Add import
    sed -i '/from .compressed_tensors_w4a16_mxfp4/i from .compressed_tensors_mxfp8 import CompressedTensorsMxfp8' "$INIT"
    # Add to __all__
    sed -i 's/"CompressedTensorsW4A8Fp8",/"CompressedTensorsW4A8Fp8",\n    "CompressedTensorsMxfp8",/' "$INIT"
    echo "Patched $INIT"
fi

# ── 4. Patch _get_scheme_from_parts() dispatch ──
CT_MAIN="$VLLM_CT/compressed_tensors.py"
if ! grep -q "_is_mxfp8" "$CT_MAIN" 2>/dev/null; then
    echo "Patching compressed_tensors.py dispatch..."
    python3 << PYEOF
path = "$CT_MAIN"
with open(path) as f:
    content = f.read()

# Add import
content = content.replace(
    "CompressedTensorsW4A16Mxfp4,",
    "CompressedTensorsMxfp8,\n    CompressedTensorsW4A16Mxfp4,"
)

# Add _is_mxfp8 static method after _is_mxfp4
mxfp8_detector = '''
    @staticmethod
    def _is_mxfp8(quant_args: QuantizationArgs) -> bool:
        if quant_args is None:
            return False

        is_group_quant = quant_args.strategy == QuantizationStrategy.GROUP.value
        is_symmetric = quant_args.symmetric
        is_group_size_32 = quant_args.group_size == 32
        is_float_type = quant_args.type == QuantizationType.FLOAT
        is_8_bits = quant_args.num_bits == 8

        return (
            is_group_quant
            and is_float_type
            and is_8_bits
            and is_group_size_32
            and is_symmetric
        )

'''

if "_is_mxfp8" not in content:
    # Insert after _is_mxfp4 method
    idx = content.find("def _is_mxfp4(")
    if idx >= 0:
        # Find the return statement's closing paren
        ret_idx = content.find("return (", idx)
        if ret_idx >= 0:
            # Find end of that return block
            end_idx = content.find(")", ret_idx + 50)  # skip past first few chars
            eol = content.index('\n', end_idx)
            content = content[:eol+1] + mxfp8_detector + content[eol+1:]

# Add dispatch after mxfp4 dispatch
mxfp8_dispatch = '''
        # MXFP8: 8-bit float, group_size=32, uint8 scales (W8A8 or W8A16)
        if self._is_mxfp8(weight_quant):
            return CompressedTensorsMxfp8()

'''

if "CompressedTensorsMxfp8()" not in content:
    target = "return CompressedTensorsW4A16Mxfp4()"
    idx = content.find(target)
    if idx >= 0:
        eol = content.index('\n', idx)
        content = content[:eol+1] + mxfp8_dispatch + content[eol+1:]

with open(path, 'w') as f:
    f.write(content)
print(f"Patched {path}")
PYEOF
else
    echo "MXFP8 dispatch already present in compressed_tensors.py"
fi

# ── 5. Add mxfp8-quantized to activation formats list ──
UTILS="$VLLM_CT/utils.py"
if ! grep -q "mxfp8_quantized" "$UTILS" 2>/dev/null; then
    echo "Adding mxfp8-quantized to activation formats..."
    sed -i 's/CompressionFormat.nvfp4_pack_quantized.value,/CompressionFormat.nvfp4_pack_quantized.value,\n        CompressionFormat.mxfp8_quantized.value,/' "$UTILS"
    echo "Patched $UTILS"
else
    echo "mxfp8-quantized already in activation formats"
fi

echo "compressed-tensors-mxfp8 mod installed"
