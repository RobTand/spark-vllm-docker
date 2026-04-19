"""
dynaquant_inject.py — Post-load hook for vLLM to replace Linear layers
with DynaQuantLinear using packed N-bit weights.

Called automatically from the patched gpu_model_runner after model loading.
Looks for /tmp/dynaquant_packed/dynaquant_config.json — if present, replaces
matching Linear layers with fused DynaQuant kernels.
"""
import json
import os
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger("dynaquant")

DYNAQUANT_DIR = os.environ.get("DYNAQUANT_DIR", "/tmp/dynaquant_packed")


def maybe_inject_dynaquant(model: nn.Module):
    """Check for DynaQuant packed weights and inject if present."""
    config_path = Path(DYNAQUANT_DIR) / "dynaquant_config.json"
    if not config_path.exists():
        return  # No DynaQuant model — skip silently

    logger.warning(f"DynaQuant: found config at {config_path}, injecting...")

    with open(config_path) as f:
        dq_config = json.load(f)

    layer_configs = dq_config["layer_configs"]
    packed_dir = Path(DYNAQUANT_DIR) / "packed_weights"

    from .dynaquant_linear import DynaQuantLinear

    # Build module lookup
    all_modules = dict(model.named_modules())

    # Debug: dump first 20 module names to find the right prefix
    module_names = list(all_modules.keys())
    linear_names = [n for n in module_names if isinstance(all_modules[n], nn.Linear)]
    logger.warning(f"DynaQuant: model has {len(module_names)} modules, {len(linear_names)} Linear")
    for n in linear_names[:10]:
        logger.warning(f"  vLLM module: {n}")

    replaced = 0
    skipped = 0

    for mod_name, cfg in layer_configs.items():
        # Resolve name: try many prefix combinations since vLLM's internal
        # module tree differs from safetensors naming
        candidates = [mod_name]
        for prefix in ["model.language_model.", "model.", "language_model."]:
            if mod_name.startswith(prefix):
                candidates.append(mod_name[len(prefix):])
        for prefix in ["language_model.", "language_model.model.", "model."]:
            candidates.append(prefix + mod_name)
            for strip in ["model.language_model.", "model.", "language_model."]:
                if mod_name.startswith(strip):
                    candidates.append(prefix + mod_name[len(strip):])

        resolve_name = None
        for c in candidates:
            if c in all_modules:
                resolve_name = c
                break

        if resolve_name is None:
            skipped += 1
            if skipped <= 5:
                logger.warning(f"  skip: {mod_name} (tried {len(candidates)} variants)")
            continue

        existing = all_modules[resolve_name]
        if not isinstance(existing, nn.Linear):
            # Could be already quantized (e.g. NVFP4 layer) — skip
            skipped += 1
            continue

        # Load packed data
        packed_path = packed_dir / f"{mod_name}.packed"
        scales_path = packed_dir / f"{mod_name}.scales"
        if not packed_path.exists():
            skipped += 1
            continue

        packed = torch.load(packed_path, weights_only=True)
        scales = torch.load(scales_path, weights_only=True)

        # Create DynaQuantLinear
        has_bias = existing.bias is not None
        device = next(existing.parameters()).device

        dq = DynaQuantLinear(
            in_features=cfg["in_features"],
            out_features=cfg["out_features"],
            n_bits=cfg["n_bits"],
            group_size=cfg["group_size"],
            bias=has_bias,
        )

        dq.packed_weight[:packed.numel()] = packed
        dq.weight_scales.copy_(scales)
        if has_bias:
            dq.bias.data.copy_(existing.bias.data)
        dq = dq.to(device)

        # Replace in model
        parts = resolve_name.rsplit(".", 1)
        if len(parts) == 2:
            parent = all_modules[parts[0]]
            setattr(parent, parts[1], dq)
        else:
            setattr(model, parts[0], dq)
        replaced += 1

        del packed, scales

    if skipped > 5:
        logger.warning(f"  ... and {skipped - 5} more skipped")
    logger.warning(f"DynaQuant: replaced {replaced} layers, skipped {skipped}")
