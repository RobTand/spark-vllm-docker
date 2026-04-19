#!/usr/bin/env python3
"""
apply_bit_recipe.py — materialize a quantized model from a water-filling allocator recipe.

Takes a Pareto JSON from allocate_bits.py and applies the per-Linear bit
assignments to a fresh copy of the source BF16 model. For each Linear:

  - If the allocated bit count matches a hardware-native format (4, 8, 16),
    use the native quantizer (NVFP4, FP8, BF16).
  - Otherwise, use software INT-b per-group symmetric quantization.

Writes the resulting model via save_pretrained so it can be loaded and
evaluated like a normal HF model.

Usage:
    python3 apply_bit_recipe.py \\
        --model /models/Qwen3.5-4B-bf16 \\
        --pareto /tmp/pareto/qwen35-4b-hw.json \\
        --step knee \\
        --output /tmp/qwen35-4b-bit-recipe-knee
"""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_rtn_cache import (
    stage_multimodal,
    iter_quantizable_tensors,
    rtn_fp4_any_shape,
    rtn_fp8_any_shape,
)
from measure_bit_utility import int_quantize_per_group


def pick_quantizer(bits: int):
    """Return a function tensor → quantized tensor for the given bit count.
    Handles 2D (nn.Linear) and 3D (fused MoE experts) tensors."""
    if bits >= 16:
        return lambda w: w
    if bits == 4:
        return rtn_fp4_any_shape
    if bits == 8:
        return rtn_fp8_any_shape
    # All other bit widths (1-15 except 4,8): software INT-b
    return lambda w: int_quantize_per_group(w, bits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pareto", required=True,
                        help="Pareto JSON from allocate_bits.py")
    parser.add_argument("--step", default="knee",
                        help="'knee' or an integer step index into the pareto frontier")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.time()

    # Load Pareto recipe
    with open(args.pareto) as f:
        pareto_data = json.load(f)
    pareto = pareto_data["pareto"]

    # Each Pareto entry now carries its own `recipe` field
    if args.step == "knee":
        knee_step = pareto_data["knee_step"]
        # Find the entry closest to knee_step (the allocator samples, so the
        # exact step may not be in the recorded entries)
        target_entry = min(pareto, key=lambda p: abs(p["step"] - knee_step))
        print(f"[recipe] knee_step={knee_step}, using recorded entry at step "
              f"{target_entry['step']}")
    elif args.step == "final":
        target_entry = pareto[-1]
    else:
        step_idx = int(args.step)
        target_entry = min(pareto, key=lambda p: abs(p["step"] - step_idx))

    recipe = target_entry.get("recipe")
    if recipe is None:
        raise ValueError("selected Pareto entry has no recipe field — re-run allocator")

    print(f"[recipe] loaded recipe for step {target_entry['step']}: "
          f"{len(recipe)} entries, "
          f"cost {target_entry['cost_bytes']/1e9:.3f} GB, "
          f"predicted KL {target_entry['predicted_kl']:.4e}")

    # Histogram of bit assignments
    from collections import Counter
    hist = Counter(recipe.values())
    print(f"[recipe] bit histogram: {dict(sorted(hist.items()))}")

    # Stage + load model
    staged, cleanup = stage_multimodal(args.model)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[recipe] loading {staged}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            staged, torch_dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
        device = next(model.parameters()).device
        print(f"[recipe]   {sum(p.numel() for p in model.parameters()):,} params", flush=True)

        # Apply the recipe: for each quantizable tensor, apply the correct
        # quantizer in place
        n_applied = 0
        n_missing = 0
        n_bf16 = 0
        for full_name, mod, attr in iter_quantizable_tensors(model):
            if full_name not in recipe:
                n_missing += 1
                continue
            bits = recipe[full_name]
            if bits >= 16:
                n_bf16 += 1
                continue
            quantizer = pick_quantizer(bits)
            param = getattr(mod, attr)
            param.data.copy_(quantizer(param.data))
            n_applied += 1

        print(f"[recipe] quantized {n_applied} linears, {n_bf16} kept at bf16, "
              f"{n_missing} not in recipe (kept at bf16)", flush=True)

        # Save
        Path(args.output).mkdir(parents=True, exist_ok=True)
        print(f"[recipe] saving to {args.output}", flush=True)
        model.save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)

        # Write a manifest for downstream comparison
        manifest = {
            "source_model": args.model,
            "pareto_source": args.pareto,
            "step": target_entry["step"],
            "cost_bytes": target_entry["cost_bytes"],
            "predicted_kl": target_entry["predicted_kl"],
            "bit_histogram": dict(hist),
            "n_quantized": n_applied,
            "n_bf16": n_bf16,
            "elapsed_sec": time.time() - t_start,
        }
        with open(Path(args.output) / "bit_recipe_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[recipe] done in {time.time() - t_start:.0f}s", flush=True)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
