#!/usr/bin/env python3
"""
Sensitivity analysis for Qwen3.5 layers.
Identifies which layers are most sensitive to quantization.

Uses weight-based sensitivity scoring (not AutoRound, which has MLLM issues).
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Layers that should NEVER be quantized (structural reasons)
ALWAYS_SKIP = [
    "lm_head",           # Final logits projection - high sensitivity
    "embed_tokens",      # Vocabulary embeddings - discrete lookup
    "mlp.gate$",         # MoE router exact (not gate_proj which is SwiGLU)
    "shared_expert_gate",# MoE shared expert gate
    "norm",              # All RMSNorm layers
    "A_log",             # GDN decay parameter (scalar)
    "dt_bias",           # GDN timestep bias (scalar)
    "conv1d",            # GDN causal conv - local mixing
]


def should_skip(name: str) -> bool:
    """Check if a layer should be skipped based on ALWAYS_SKIP patterns."""
    for pattern in ALWAYS_SKIP:
        if pattern in name:
            return True
    return False


def compute_fp4_quant_error(weight: torch.Tensor) -> float:
    """
    Compute relative quantization error for FP4 E2M1 format.
    FP4 E2M1 has range [-6, 6] with values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6

    Uses efficient rounding approach instead of distance matrix.
    """
    with torch.no_grad():
        weight = weight.float()
        max_val = weight.abs().max()
        if max_val == 0:
            return 0.0

        # Scale to FP4 range [-6, 6]
        scale = max_val / 6.0
        scaled = weight / scale

        # FP4 E2M1 quantization: round to nearest representable value
        # Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
        # Simplified: for values |x| > 2, use log-scale rounding
        abs_scaled = scaled.abs()
        sign = scaled.sign()

        # For |x| <= 2: round to nearest 0.5
        # For |x| > 2: use {3, 4, 6}
        quantized_abs = torch.where(
            abs_scaled <= 2.0,
            (abs_scaled * 2).round() / 2,  # Round to nearest 0.5
            torch.where(
                abs_scaled <= 2.5, torch.full_like(abs_scaled, 2.0),
                torch.where(
                    abs_scaled <= 3.5, torch.full_like(abs_scaled, 3.0),
                    torch.where(
                        abs_scaled <= 5.0, torch.full_like(abs_scaled, 4.0),
                        torch.full_like(abs_scaled, 6.0)
                    )
                )
            )
        )

        quantized = sign * quantized_abs * scale

        # Relative error
        error = (weight - quantized).abs()
        rel_error = error.mean() / (weight.abs().mean() + 1e-8)
        return rel_error.item()


def run_sensitivity_analysis(
    model_name: str,
    output_dir: str,
    nsamples: int = 128,
    seqlen: int = 2048,
    batch_size: int = 4,
    device: str = "cuda",
):
    """
    Run sensitivity analysis by computing per-layer FP4 quantization error.
    Higher error = more sensitive layer.
    """
    print(f"Loading model: {model_name}", flush=True)

    # Load model to GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",  # Explicitly use GPU
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    print("Computing per-layer FP4 quantization sensitivity...", flush=True)

    # Compute per-layer quantization error
    sensitivity_scores = {}
    total_layers = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear))
    processed = 0

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        # Skip tiny layers and always-skip patterns
        if module.weight.numel() < 1000 or should_skip(name):
            continue

        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed}/{total_layers} linear layers...", flush=True)

        # Compute FP4 quantization error
        quant_error = compute_fp4_quant_error(module.weight.data)

        # Weight by layer size (larger layers have more impact)
        size_weight = module.weight.numel() / 1e6  # in millions

        sensitivity_scores[name] = {
            "quant_error": quant_error,
            "params_M": size_weight,
            "score": quant_error * size_weight,  # Error weighted by size
        }

    # Sort by sensitivity (highest = most sensitive)
    sorted_layers = sorted(
        sensitivity_scores.items(),
        key=lambda x: x[1].get('score', x[1].get('loss', 0)),
        reverse=True
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / "sensitivity_scores.json"

    results = {
        "model": model_name,
        "nsamples": nsamples,
        "seqlen": seqlen,
        "always_skip": ALWAYS_SKIP,
        "layer_scores": dict(sorted_layers),
        "top_10_sensitive": [name for name, _ in sorted_layers[:10]],
        "top_20_sensitive": [name for name, _ in sorted_layers[:20]],
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSensitivity analysis complete. Results saved to {output_file}")
    print("\nTop 10 most sensitive layers:")
    for i, (name, scores) in enumerate(sorted_layers[:10], 1):
        print(f"  {i}. {name}: {scores}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run sensitivity analysis on Qwen3.5")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B",
                        help="Model name or path")
    parser.add_argument("--output", type=str, default="/workspace/sensitivity_output",
                        help="Output directory for results")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length for calibration")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for calibration")

    args = parser.parse_args()

    run_sensitivity_analysis(
        model_name=args.model,
        output_dir=args.output,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
