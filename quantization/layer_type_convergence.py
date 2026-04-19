#!/usr/bin/env python3
"""
Layer-type convergence analysis using a small proxy model.

Runs AutoRound on every layer and records the full loss trajectory,
then groups results by layer TYPE (not index) to produce findings
that transfer to larger models in the same architecture family.

Output: per-layer-type convergence curves, asymptotes, and
recommendations for FP4/FP8/BF16 assignment.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_layer_type(name: str) -> str:
    """Extract the layer type pattern (strip layer index)."""
    return re.sub(r'\.layers\.\d+\.', '.layers.*.', name)


def run_convergence_analysis(
    model_path: str,
    output_dir: str,
    nsamples: int = 128,
    seqlen: int = 2048,
    iters: int = 200,
    batch_size: int = 4,
):
    print(f"Loading model: {model_path}", flush=True)

    # Remove vision config to avoid MLLM detection
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    removed_keys = {}
    for key in ["vision_config", "image_token_id", "video_token_id",
                "vision_start_token_id", "vision_end_token_id"]:
        if key in config_data:
            removed_keys[key] = config_data.pop(key)

    temp_config = config_path.with_suffix(".json.bak")
    config_path.rename(temp_config)
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        n_layers = len(model.model.layers)
        print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params, {n_layers} layers", flush=True)

        # Catalog all linear layers by type
        layer_catalog = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                ltype = get_layer_type(name)
                params = module.weight.numel()
                if ltype not in layer_catalog:
                    layer_catalog[ltype] = {"count": 0, "params": params}
                layer_catalog[ltype]["count"] += 1

        print(f"\nLayer types found:", flush=True)
        for ltype, info in sorted(layer_catalog.items()):
            print(f"  {ltype}: {info['params']/1e6:.1f}M x{info['count']}", flush=True)

        # Only skip truly structural layers (exact matches)
        structural_skip = ["lm_head", "embed_tokens"]

        layer_config = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                for pat in structural_skip:
                    if pat in name:
                        layer_config[name] = {"bits": 16, "group_size": -1}
                        break

        # Also skip norms, conv1d, A_log, dt_bias (non-Linear, won't match anyway)
        # But skip nothing else — we want convergence data for ALL linear layers

        print(f"\nSkipping {len(layer_config)} structural layers", flush=True)
        print(f"Quantizing everything else to get convergence data\n", flush=True)

        from auto_round.compressors import LLMCompressor

        autoround = LLMCompressor(
            model=model,
            tokenizer=tokenizer,
            bits=4,
            group_size=16,
            sym=True,
            batch_size=batch_size,
            seqlen=seqlen,
            nsamples=nsamples,
            iters=iters,
            dataset="NeelNanda/pile-10k",
            layer_config=layer_config,
        )

        # Run quantization
        t0 = time.time()
        autoround.quantize()
        elapsed = time.time() - t0

        print(f"\nQuantization completed in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

        # Save the quantized model
        os.makedirs(output_dir, exist_ok=True)
        autoround.save_quantized(output_dir, format="auto_round", inplace=True)
        tokenizer.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}", flush=True)

    finally:
        config_path.unlink(missing_ok=True)
        temp_config.rename(config_path)


def main():
    parser = argparse.ArgumentParser(description="Layer-type convergence analysis")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()
    run_convergence_analysis(
        model_path=args.model,
        output_dir=args.output,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        iters=args.iters,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
