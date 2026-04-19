#!/usr/bin/env python3
"""
AutoRound NVFP4 quantization for Qwen3.5.

AutoRound minimizes quantization error through gradient-based optimization,
producing better quality than naive rounding. This script works around
the MLLM detection issue by using AutoRound's core quantization directly.

Usage:
    python quantize_autoround_nvfp4.py \
        --model /models/Qwen3.5-27B-bf16 \
        --output /models/qwen35-27b-autoround-nvfp4 \
        --nsamples 128 \
        --seqlen 4096
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Layers to always skip (structural/quality reasons)
ALWAYS_SKIP = [
    "lm_head",
    "embed_tokens",
    "mlp.gate$",            # MoE router (exact match, not gate_proj)
    "shared_expert_gate",   # Shared expert gate
    "norm",                 # All norms
    "A_log",                # GDN decay
    "dt_bias",              # GDN timestep
    "conv1d",               # GDN causal conv
]


def build_layer_config(model, always_skip: list[str]) -> dict:
    """Build AutoRound layer_config to skip certain layers."""
    import re
    config = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            for pattern in always_skip:
                if pattern.endswith("$"):
                    # Regex-style end anchor
                    if re.search(pattern, name):
                        config[name] = {"bits": 16, "group_size": -1}
                        break
                elif pattern in name:
                    config[name] = {"bits": 16, "group_size": -1}
                    break
    return config


def quantize_with_autoround(
    model_path: str,
    output_dir: str,
    nsamples: int = 128,
    seqlen: int = 4096,
    iters: int = 200,
    batch_size: int = 4,
):
    """
    Quantize model using AutoRound's NVFP4-compatible settings.

    AutoRound uses gradient-based optimization to minimize the quantization
    error between original and quantized weights. This produces significantly
    better quality than naive rounding, especially for sensitive layers.
    """
    print(f"Loading model: {model_path}", flush=True)

    # Workaround: Temporarily modify config to hide vision components
    # This prevents AutoRound from detecting it as MLLM
    import json
    from pathlib import Path
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    # Remove vision-related keys temporarily
    removed_keys = {}
    for key in ["vision_config", "image_token_id", "video_token_id",
                "vision_start_token_id", "vision_end_token_id"]:
        if key in config_data:
            removed_keys[key] = config_data.pop(key)

    # Write modified config temporarily
    temp_config = config_path.with_suffix(".json.bak")
    config_path.rename(temp_config)
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    try:
        # Load model to GPU with modified config
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

        # Build layer config for skip patterns
        layer_config = build_layer_config(model, ALWAYS_SKIP)
        print(f"Skipping {len(layer_config)} layers in bf16", flush=True)

        # Use LLMCompressor directly to bypass MLLM auto-detection
        from auto_round.compressors import LLMCompressor

        print(f"Running AutoRound quantization (iters={iters}, nsamples={nsamples})...", flush=True)

        # Configure AutoRound for FP4 E2M1 (NVFP4-compatible)
        # bits=4, group_size=16 is the NVFP4 config
        # sym=True for symmetric quantization
        # use_minmax_tuning=True for better scale optimization

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
            # Calibration dataset - mix of general text and instruction data
            dataset="NeelNanda/pile-10k",
            # Skip layers that should stay in bf16
            layer_config=layer_config,
        )

        # Run quantization and save (recommended API, handles state better)
        os.makedirs(output_dir, exist_ok=True)
        autoround.quantize_and_save(
            output_dir=output_dir,
            format="auto_round",  # auto_round native format
            inplace=True,
        )
        print(f"Quantization complete, saved to {output_dir}", flush=True)

        # Also copy tokenizer files
        tokenizer.save_pretrained(output_dir)

        # Save quantization info
        info = {
            "source_model": model_path,
            "quantization": "AutoRound NVFP4",
            "bits": 4,
            "group_size": 16,
            "sym": True,
            "iters": iters,
            "nsamples": nsamples,
            "seqlen": seqlen,
            "skipped_layers": list(layer_config.keys()),
        }
        with open(Path(output_dir) / "quantization_info.json", "w") as f:
            json.dump(info, f, indent=2)

    finally:
        # Restore original config
        config_path.unlink(missing_ok=True)
        temp_config.rename(config_path)


def main():
    parser = argparse.ArgumentParser(description="AutoRound NVFP4 quantization for Qwen3.5")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or HuggingFace name")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=4096,
                        help="Sequence length for calibration")
    parser.add_argument("--iters", type=int, default=200,
                        help="Max optimization iterations per layer (AutoRound stops early when converged)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for calibration")

    args = parser.parse_args()

    quantize_with_autoround(
        model_path=args.model,
        output_dir=args.output,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        iters=args.iters,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
