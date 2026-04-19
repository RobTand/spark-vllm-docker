#!/usr/bin/env python3
"""Patch vLLM to support DynaQuant arbitrary bit-width quantization.

Patches:
1. compressed_tensors/schemes/__init__.py — export CompressedTensorsDynaQuant
2. compressed_tensors/compressed_tensors.py — add dynaquant format detection + routing
3. compressed_tensors/utils.py — add dynaquant to supported data_types
"""
import os
import sys

import glob as _glob

# Find the compressed_tensors directory — path may vary across vLLM versions
_candidates = _glob.glob("/usr/local/lib/python3.*/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors")
if not _candidates:
    # Try site-packages too
    _candidates = _glob.glob("/usr/lib/python3.*/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors")
CT_DIR = _candidates[0] if _candidates else "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors"


def patch_schemes_init():
    """No changes needed — DynaQuant is imported lazily in compressed_tensors.py."""
    print("schemes/__init__.py — no changes needed (lazy import)")


def patch_compressed_tensors():
    path = f"{CT_DIR}/compressed_tensors.py"
    if not os.path.exists(path):
        print(f"compressed_tensors.py not found at {path}, skipping patch")
        return
    with open(path) as f:
        content = f.read()
    if "DynaQuant" in content:
        print("compressed_tensors.py already patched")
        return

    # 1. No top-level import — use lazy import in the routing code to avoid circular import

    # 2. Add detection method for dynaquant format
    # Insert before _get_scheme_from_parts
    detection_code = '''
    @staticmethod
    def _is_dynaquant_format(weight_quant) -> bool:
        """Check if this is a DynaQuant arbitrary bit-width format."""
        if weight_quant is None:
            return False
        # DynaQuant supports bits 1-16 with integer type, group quantization
        return (
            weight_quant.num_bits is not None
            and 1 <= weight_quant.num_bits <= 16
            and weight_quant.type == "int"
            and weight_quant.strategy in ("group", "tensor_group")
        )

'''
    content = content.replace(
        '    def _get_scheme_from_parts(',
        detection_code + '    def _get_scheme_from_parts(',
    )

    # 3. Add dynaquant routing in _get_scheme_from_parts
    # Insert early in the chain — before the pack_quantized / WNA16 check
    # Find the WNA16 check and insert before it
    wna16_check = "self._is_wNa16_group_channel(weight_quant, input_quant)"
    if wna16_check in content:
        dynaquant_block = (
            "\n"
            "        # DynaQuant: arbitrary bit-width (3-15) with fused Triton kernels\n"
            "        if (\n"
            "            self._is_dynaquant_format(weight_quant)\n"
            '            and (format == "dynaquant-pack-quantized"\n'
            '                 or (format == "pack-quantized"\n'
            "                     and weight_quant.num_bits not in (4, 8)))\n"
            "        ):\n"
            "            from dynaquant.compressed_tensors_dynaquant import CompressedTensorsDynaQuant\n"
            "            return CompressedTensorsDynaQuant(\n"
            "                num_bits=weight_quant.num_bits,\n"
            "                group_size=weight_quant.group_size or 16,\n"
            "            )\n"
            "\n"
        )
        target_line = f"        if (\n            {wna16_check}"
        content = content.replace(target_line, dynaquant_block + target_line)
    else:
        print("WARNING: Could not find WNA16 check to insert before")

    with open(path, 'w') as f:
        f.write(content)
    print("Patched compressed_tensors.py with DynaQuant scheme routing")


def patch_moe_dispatch():
    """Add DynaQuant routing to the compressed-tensors MoE dispatcher."""
    moe_dir = f"{CT_DIR}/compressed_tensors_moe"
    path = f"{moe_dir}/compressed_tensors_moe.py"
    if not os.path.exists(path):
        print(f"compressed_tensors_moe.py not found at {path}, skipping MoE patch")
        return
    with open(path) as f:
        content = f.read()
    if "DynaQuant" in content:
        print("compressed_tensors_moe.py already patched")
        return

    # Insert DynaQuant check before the WNA16 check
    # The WNA16 check is: "if quant_config._is_wNa16_group_channel"
    wna16_check = "if quant_config._is_wNa16_group_channel(weight_quant, input_quant):"
    if wna16_check in content:
        dynaquant_block = (
            "\n"
            "        # DynaQuant MoE: arbitrary bit-width (1-16) per expert\n"
            '        if format == "dynaquant-pack-quantized":\n'
            "            from dynaquant.dynaquant_moe import DynaQuantFusedMoEMethod\n"
            "            # max_bits from config controls allocation size\n"
            "            _mb = min(16, max(weight_quant.num_bits * 2, 8))\n"
            "            # Check for per-row MoE scales\n"
            "            _per_row = getattr(quant_config, 'moe_per_row_scales', False)\n"
            "            if hasattr(quant_config, 'config') and quant_config.config:\n"
            "                _per_row = quant_config.config.get('moe_per_row_scales', False)\n"
            "            return DynaQuantFusedMoEMethod(\n"
            "                moe=layer.moe_config,\n"
            "                group_size=weight_quant.group_size or 16,\n"
            "                max_bits=_mb,\n"
            "                per_row_scales=_per_row,\n"
            "            )\n"
            "\n"
            "        "
        )
        content = content.replace(
            "        " + wna16_check,
            dynaquant_block + wna16_check,
        )

        with open(path, 'w') as f:
            f.write(content)
        print("Patched compressed_tensors_moe.py with DynaQuant MoE routing")
    else:
        print("WARNING: Could not find WNA16 check in MoE dispatcher")


def patch_utils_data_types():
    """Add 'int' to supported data_types if it's restricted."""
    path = f"{CT_DIR}/utils.py"
    if not os.path.exists(path):
        print(f"utils.py not found at {path}, skipping")
        return
    with open(path) as f:
        content = f.read()
    # The error was "Unsupported data_type: nv_fp, currently only support {'int'}"
    # For dynaquant we use type="int" which should already be supported
    # But let's make sure the format string is accepted
    if "dynaquant" in content:
        print("utils.py already patched")
        return
    # No changes needed for utils.py — 'int' type is already supported
    print("utils.py — no changes needed")


def patch_model_runner():
    """Remove the old DynaQuant injection hook if present (no longer needed)."""
    _runner_candidates = _glob.glob("/usr/local/lib/python3.*/dist-packages/vllm/v1/worker/gpu_model_runner.py")
    if not _runner_candidates:
        print("gpu_model_runner.py — not found (skip)")
        return
    path = _runner_candidates[0]
    with open(path) as f:
        content = f.read()
    if "dynaquant" not in content:
        print("gpu_model_runner.py — no injection hook present (clean)")
        return
    # Remove the old injection hook
    lines = content.split('\n')
    new_lines = []
    skip = False
    for line in lines:
        if "DynaQuant: inject packed" in line:
            skip = True
        if skip and line.strip() == "":
            skip = False
            continue
        if not skip:
            new_lines.append(line)
    with open(path, 'w') as f:
        f.write('\n'.join(new_lines))
    print("gpu_model_runner.py — removed old injection hook")


if __name__ == "__main__":
    patch_schemes_init()
    patch_compressed_tensors()
    patch_moe_dispatch()
    patch_utils_data_types()
    patch_model_runner()
    print("All patches applied successfully")
