#!/usr/bin/env python3
"""
Quantization Error Propagation Analysis

Builds a theoretical framework to predict end-to-end quality degradation
from per-layer quantization errors, without running full evaluations.

Key quantities:
1. Per-layer reconstruction error (ε_i): How much does layer output change?
2. Layer Lipschitz constant (L_i): How much does layer amplify input errors?
3. Gradient sensitivity: Which layers most affect final loss?

Hypothesis: End-to-end perplexity increase is predictable from:
    E_total ≈ Σᵢ (εᵢ × wᵢ)
where wᵢ = sensitivity weight derived from Lipschitz constants and gradients.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def estimate_lipschitz_constant(module: torch.nn.Module, input_sample: torch.Tensor,
                                 n_samples: int = 10, eps: float = 0.01) -> float:
    """
    Estimate Lipschitz constant of a module by sampling.
    L = max ||f(x+δ) - f(x)|| / ||δ||
    """
    module.eval()
    with torch.no_grad():
        base_output = module(input_sample)
        if isinstance(base_output, tuple):
            base_output = base_output[0]

        max_ratio = 0.0
        for _ in range(n_samples):
            # Random perturbation
            delta = torch.randn_like(input_sample) * eps
            perturbed_output = module(input_sample + delta)
            if isinstance(perturbed_output, tuple):
                perturbed_output = perturbed_output[0]

            output_diff = (perturbed_output - base_output).norm()
            input_diff = delta.norm()

            if input_diff > 0:
                ratio = (output_diff / input_diff).item()
                max_ratio = max(max_ratio, ratio)

        return max_ratio


def compute_fp4_reconstruction_error(weight: torch.Tensor, group_size: int = 16) -> dict:
    """
    Compute reconstruction error for FP4 E2M1 quantization.
    Returns both absolute and relative errors.
    """
    with torch.no_grad():
        weight = weight.float()
        original_shape = weight.shape

        # Reshape for group-wise quantization
        if weight.numel() % group_size != 0:
            # Pad if needed
            pad_size = group_size - (weight.numel() % group_size)
            weight_flat = F.pad(weight.flatten(), (0, pad_size))
        else:
            weight_flat = weight.flatten()

        weight_grouped = weight_flat.reshape(-1, group_size)

        # Per-group scaling (NVFP4 style)
        scales = weight_grouped.abs().max(dim=1, keepdim=True).values / 6.0
        scales = scales.clamp(min=1e-8)

        scaled = weight_grouped / scales

        # FP4 E2M1 quantization (simplified)
        # Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
        abs_scaled = scaled.abs()
        sign = scaled.sign()

        quantized_abs = torch.where(
            abs_scaled <= 0.25, torch.zeros_like(abs_scaled),
            torch.where(
                abs_scaled <= 0.75, torch.full_like(abs_scaled, 0.5),
                torch.where(
                    abs_scaled <= 1.25, torch.ones_like(abs_scaled),
                    torch.where(
                        abs_scaled <= 1.75, torch.full_like(abs_scaled, 1.5),
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
                )
            )
        )

        quantized = sign * quantized_abs * scales
        quantized_flat = quantized.flatten()[:weight.numel()]
        quantized_weight = quantized_flat.reshape(original_shape)

        # Error metrics
        abs_error = (weight.reshape(original_shape) - quantized_weight).abs()
        rel_error = abs_error / (weight.reshape(original_shape).abs() + 1e-8)

        # Frobenius norm error (what matters for layer output)
        frob_error = abs_error.norm() / (weight.reshape(original_shape).norm() + 1e-8)

        return {
            "mean_abs_error": abs_error.mean().item(),
            "max_abs_error": abs_error.max().item(),
            "mean_rel_error": rel_error.mean().item(),
            "frobenius_rel_error": frob_error.item(),
            "weight_norm": weight.norm().item(),
            "error_norm": abs_error.norm().item(),
        }


def compute_layer_sensitivity(model, layer_idx: int, calibration_data: torch.Tensor) -> dict:
    """
    Compute how sensitive the final output is to errors in this layer.
    Uses gradient-based sensitivity: ||∂Loss/∂layer_output||
    """
    model.eval()

    # Get the specific layer
    layer = model.model.layers[layer_idx]

    # Hook to capture gradients
    gradients = []
    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradients.append(grad_output[0].detach())

    handle = layer.register_full_backward_hook(hook)

    try:
        # Forward pass with gradient tracking
        model.zero_grad()
        outputs = model(calibration_data, labels=calibration_data)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        if gradients:
            grad_norm = gradients[0].norm().item()
            grad_mean = gradients[0].abs().mean().item()
        else:
            grad_norm = 0.0
            grad_mean = 0.0

    finally:
        handle.remove()

    return {
        "gradient_norm": grad_norm,
        "gradient_mean": grad_mean,
        "loss_value": loss.item(),
    }


def analyze_error_propagation(
    model_path: str,
    output_dir: str,
    n_calibration: int = 8,
    seqlen: int = 512,
):
    """
    Full analysis of quantization error propagation.
    """
    print(f"Loading model: {model_path}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    n_layers = len(model.model.layers)
    print(f"Model has {n_layers} layers", flush=True)

    # Load calibration data
    print("Loading calibration data...", flush=True)
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [s["text"][:seqlen*4] for s in dataset.select(range(n_calibration))]

    encodings = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=seqlen, padding=True
    )
    calibration_data = encodings["input_ids"].to(model.device)

    results = {
        "model": model_path,
        "n_layers": n_layers,
        "analysis_config": {
            "n_calibration": n_calibration,
            "seqlen": seqlen,
        },
        "per_layer": {},
        "theory": {},
    }

    # Analyze each layer
    print("Analyzing per-layer errors and sensitivity...", flush=True)

    total_weighted_error = 0.0
    lipschitz_product = 1.0  # Cumulative Lipschitz from end

    # Process layers in reverse (for Lipschitz accumulation)
    layer_data = []

    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        print(f"  Layer {layer_idx}/{n_layers}...", flush=True)

        layer_info = {
            "layer_idx": layer_idx,
            "sublayers": {},
        }

        # Analyze each linear sublayer
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Quantization error
                quant_error = compute_fp4_reconstruction_error(module.weight)
                layer_info["sublayers"][name] = quant_error

        # Aggregate layer error (sum of sublayer Frobenius errors)
        layer_error = sum(
            s["frobenius_rel_error"]
            for s in layer_info["sublayers"].values()
        )
        layer_info["total_frobenius_error"] = layer_error

        layer_data.append(layer_info)
        results["per_layer"][f"layer_{layer_idx}"] = layer_info

    # Compute theoretical bounds
    print("Computing theoretical error bounds...", flush=True)

    # Simple model: errors add (conservative upper bound)
    total_error_sum = sum(ld["total_frobenius_error"] for ld in layer_data)

    # Weighted by position (later layers have less propagation)
    # Weight decays geometrically: layer i has weight (1/2)^(n-i)
    weighted_error = sum(
        ld["total_frobenius_error"] * (0.9 ** (n_layers - i - 1))
        for i, ld in enumerate(layer_data)
    )

    # Most sensitive layers (top-10 by error)
    sorted_layers = sorted(
        enumerate(layer_data),
        key=lambda x: x[1]["total_frobenius_error"],
        reverse=True
    )

    results["theory"] = {
        "total_error_sum": total_error_sum,
        "weighted_error": weighted_error,
        "mean_layer_error": total_error_sum / n_layers,
        "predicted_ppl_increase_factor": 1.0 + weighted_error,  # Rough estimate
        "most_sensitive_layers": [
            {"layer": i, "error": ld["total_frobenius_error"]}
            for i, ld in sorted_layers[:10]
        ],
    }

    # Hypothesis formulation
    results["hypothesis"] = {
        "statement": "End-to-end perplexity increase is approximately (1 + weighted_error) where weighted_error accounts for error propagation through subsequent layers.",
        "prediction": f"PPL should increase by factor of approximately {1.0 + weighted_error:.4f}",
        "to_validate": [
            "1. Run bf16 baseline perplexity",
            "2. Run FP4 quantized perplexity",
            "3. Compare ratio to predicted factor",
            "4. If ratio differs significantly, refine weighting scheme",
        ],
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / "error_propagation_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis complete. Results saved to {output_file}", flush=True)
    print(f"\nKey findings:", flush=True)
    print(f"  Total error sum: {total_error_sum:.6f}", flush=True)
    print(f"  Weighted error: {weighted_error:.6f}", flush=True)
    print(f"  Predicted PPL increase: {1.0 + weighted_error:.4f}x", flush=True)
    print(f"\nTop 5 most sensitive layers:", flush=True)
    for i, ld in sorted_layers[:5]:
        print(f"    Layer {i}: error = {ld['total_frobenius_error']:.6f}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze quantization error propagation")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--n-calibration", type=int, default=8, help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=512, help="Sequence length")

    args = parser.parse_args()

    analyze_error_propagation(
        model_path=args.model,
        output_dir=args.output,
        n_calibration=args.n_calibration,
        seqlen=args.seqlen,
    )


if __name__ == "__main__":
    main()
