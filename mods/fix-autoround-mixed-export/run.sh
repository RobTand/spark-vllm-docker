#!/bin/bash
set -euo pipefail

# AutoRound mixed-precision export pipeline fixes.
#
# Upstream AutoRound has three issues that break direct export of
# AutoScheme's mixed NVFP4 + MXFP8 picks to a vLLM-compatible
# compressed-tensors checkpoint:
#
#   1. auto_round/export/export_to_llmcompressor/export_to_fp.py
#      Hardcodes a single `group_0` and globally overrides parameters
#      when the DEFAULT data_type is mx_fp. Per-layer overrides in
#      layer_config are ignored, so AutoScheme's mixed output gets
#      flattened to one format.
#
#   2. vLLM's fused-projection loader (q_proj/k_proj/v_proj -> qkv_proj,
#      gate_proj/up_proj -> gate_up_proj) requires siblings to share the
#      same quantization scheme, otherwise the fused tensor layout is
#      invalid. AutoScheme can pick different formats per sibling.
#
#   3. Scheme dicts contain torch.dtype objects that HuggingFace
#      Transformers' config.save_pretrained() cannot JSON-serialize.
#
# The patched files in this mod:
#   - compressors/base.py: promotes fused-projection siblings to the
#     highest-precision scheme in layer_config BEFORE quantization, so
#     AutoRound packs consistent weights.
#   - export/export_to_llmcompressor/export_to_fp.py: emits one
#     config_group per unique format with explicit per-layer regex
#     targets (no "Linear" catch-all, matches llm_compressor output),
#     sets `format: mixed-precision` at the top level, and sanitizes
#     dtype objects before JSON serialization.
#
# Validated end-to-end on Qwen3-0.6B with NVFP4+MXFP8 mixed checkpoint
# loading in vLLM and generating coherent output with both FLASHINFER_CUTLASS
# kernel backends dispatched.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AR_BASE=$(python3 -c "import auto_round, os; print(os.path.dirname(auto_round.__file__))")
echo "auto_round install: $AR_BASE"

cp -v "$SCRIPT_DIR/patches/base.py"          "$AR_BASE/compressors/base.py"
cp -v "$SCRIPT_DIR/patches/export_to_fp.py"  "$AR_BASE/export/export_to_llmcompressor/export_to_fp.py"

# Sanity check: both marker strings should be present
for f in compressors/base.py export/export_to_llmcompressor/export_to_fp.py; do
    if ! grep -q "AUTO_ROUND_MIXED_EXPORT_PATCH_V1\|AUTO_ROUND_FUSED_PROMOTE_V1" "$AR_BASE/$f"; then
        echo "WARNING: expected marker missing from $f"
    fi
done

echo "AutoRound mixed-export patch installed"
