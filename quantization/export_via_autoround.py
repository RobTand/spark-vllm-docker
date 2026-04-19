#!/usr/bin/env python3
"""
export_via_autoround.py — export a DynaQuant {4,8,16} recipe as a
compressed-tensors model using AutoRound with iters=0 (pure RTN).

AutoRound at iters=0 does round-to-nearest (no gradient optimization)
and saves in the compressed-tensors format that vLLM loads natively.
Our DynaQuant recipe controls which layers get FP4/FP8/BF16.

The output model can be served directly:
    vllm serve /path/to/output --trust-remote-code

Usage:
    python3 export_via_autoround.py \\
        --model /models/Qwen3.5-27B-bf16 \\
        --pareto /tmp/pareto/qwen35-27b-hw.json \\
        --step knee \\
        --output /tmp/dynaquant-27b-servable
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/tmp/auto-round")
from build_rtn_cache import stage_multimodal, should_always_skip


def build_layer_config(recipe: dict, model) -> dict:
    """Convert a DynaQuant recipe to AutoRound's layer_config format.

    FP4 layers → NVFP4 quantization (packed by AutoRound)
    FP8/BF16 layers → skip (bits=16). These get RTN-FP8 applied manually
    before AutoRound runs, so they hold quantized values in bf16 storage.
    """
    import torch.nn as nn
    from build_rtn_cache import rtn_fp8_any_shape, should_always_skip

    layer_config = {}
    n_fp8_applied = 0
    for name, bits in recipe.items():
        mod_name = name.replace(".weight", "")

        if bits <= 4:
            layer_config[mod_name] = {
                "bits": 4,
                "group_size": 16,
                "data_type": "nv_fp4",
            }
        else:
            # FP8 and BF16 layers: skip AutoRound quantization.
            # For FP8 layers, we pre-apply RTN-FP8 to the model weights
            # so the bf16 storage holds FP8-quality values.
            layer_config[mod_name] = {
                "bits": 16,
                "group_size": -1,
                "data_type": "int",
            }
            if bits <= 8:
                # Pre-apply RTN-FP8 to this layer's weights
                parts = mod_name.split(".")
                mod = model
                for p in parts:
                    mod = getattr(mod, p, None)
                    if mod is None:
                        break
                if mod is not None and hasattr(mod, "weight"):
                    mod.weight.data.copy_(rtn_fp8_any_shape(mod.weight.data))
                    n_fp8_applied += 1

    print(f"[export] pre-applied RTN-FP8 to {n_fp8_applied} layers", flush=True)
    return layer_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pareto", required=True)
    parser.add_argument("--step", default="knee")
    parser.add_argument("--output", required=True)
    parser.add_argument("--nsamples", type=int, default=8,
                        help="Calibration samples (used for activation scales only)")
    parser.add_argument("--seqlen", type=int, default=512)
    args = parser.parse_args()

    t_start = time.time()

    # Load recipe
    with open(args.pareto) as f:
        pareto_data = json.load(f)
    pareto = pareto_data["pareto"]
    if args.step == "knee":
        entry = min(pareto, key=lambda p: abs(p["step"] - pareto_data["knee_step"]))
    else:
        entry = min(pareto, key=lambda p: abs(p["step"] - int(args.step)))
    recipe = entry["recipe"]

    # Snap to hardware buckets and count
    snapped = {}
    for name, bits in recipe.items():
        if bits <= 4:
            snapped[name] = 4
        elif bits <= 8:
            snapped[name] = 8
        else:
            snapped[name] = 16

    hist = Counter(snapped.values())
    print(f"[export] Recipe: {dict(sorted(hist.items()))}")
    print(f"[export] Predicted cost: {entry['cost_bytes']/1e9:.2f} GB")

    # layer_config is built inside the model loading block (needs model reference)

    # Stage model if multimodal
    staged, cleanup = stage_multimodal(args.model)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[export] loading {staged}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            staged, torch_dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)

        # Build layer config — this also pre-applies RTN-FP8 to FP8 layers
        layer_config = build_layer_config(snapped, model)

        # Use the NVFP4 scheme as the default (FP4 layers)
        # AutoRound with iters=0 = pure RTN, no gradient optimization
        # FP8 layers are already RTN-quantized above, marked as bits=16 (skip)
        from auto_round.compressors import LLMCompressor

        print(f"[export] running AutoRound iters=0 (pure RTN) for FP4 layers",
              flush=True)
        compressor = LLMCompressor(
            model=model,
            tokenizer=tokenizer,
            bits=4,
            group_size=16,
            sym=True,
            data_type="nv_fp4",
            iters=0,  # RTN only — no gradient optimization
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            batch_size=1,
            dataset="NeelNanda/pile-10k",
            layer_config=layer_config,
        )

        t0 = time.time()
        compressor.quantize()
        print(f"[export] quantization done in {time.time() - t0:.0f}s", flush=True)

        # Save in compressed-tensors format
        print(f"[export] saving to {args.output}", flush=True)
        compressor.save_quantized(args.output, format="auto_round")

        print(f"[export] done in {time.time() - t_start:.0f}s", flush=True)
        print(f"[export] serve with: vllm serve {args.output} --trust-remote-code")

    finally:
        if cleanup:
            import shutil
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
