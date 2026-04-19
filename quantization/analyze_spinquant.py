#!/usr/bin/env python3
"""
Analyze SpinQuant's effect on weight distributions and FP4 quantization error.
No AutoRound optimization - just measure the impact of Hadamard rotation.
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from scipy.stats import kurtosis


def hadamard_matrix(n: int) -> torch.Tensor:
    """Generate normalized Hadamard matrix of size n (must be power of 2)."""
    if n == 1:
        return torch.tensor([[1.0]])
    h = hadamard_matrix(n // 2)
    return torch.cat([
        torch.cat([h, h], dim=1),
        torch.cat([h, -h], dim=1)
    ], dim=0) / np.sqrt(2)


def apply_hadamard_rotation(weight: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation to weight matrix."""
    out_features, in_features = weight.shape

    # Pad to nearest power of 2 if needed
    in_pad = 2 ** int(np.ceil(np.log2(in_features))) - in_features
    if in_pad > 0:
        weight = torch.nn.functional.pad(weight, (0, in_pad))

    h = hadamard_matrix(weight.shape[1]).to(weight.device, weight.dtype)
    rotated = weight @ h

    # Remove padding
    if in_pad > 0:
        rotated = rotated[:, :in_features]

    return rotated


def compute_fp4_error(weight: torch.Tensor, group_size: int = 16) -> dict:
    """
    Compute FP4 E2M1 quantization error (RTN, no optimization).
    Returns per-group and overall statistics.
    """
    weight = weight.float()
    out_features, in_features = weight.shape

    # Reshape into groups
    n_groups = (in_features + group_size - 1) // group_size
    padded = torch.nn.functional.pad(weight, (0, n_groups * group_size - in_features))
    grouped = padded.view(out_features, n_groups, group_size)

    # Per-group scale (max abs value)
    scales = grouped.abs().max(dim=2, keepdim=True).values.clamp(min=1e-8)

    # Normalize to [-6, 6] range (FP4 E2M1)
    normalized = grouped / scales * 6.0

    # FP4 E2M1 values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
    abs_norm = normalized.abs()
    sign = normalized.sign()

    # Quantize
    quantized_abs = torch.where(
        abs_norm <= 2.0,
        (abs_norm * 2).round() / 2,
        torch.where(
            abs_norm <= 2.5, torch.full_like(abs_norm, 2.0),
            torch.where(
                abs_norm <= 3.5, torch.full_like(abs_norm, 3.0),
                torch.where(
                    abs_norm <= 5.0, torch.full_like(abs_norm, 4.0),
                    torch.full_like(abs_norm, 6.0)
                )
            )
        )
    )

    quantized = sign * quantized_abs * scales / 6.0

    # Compute error
    error = (grouped - quantized).abs()
    rel_error = error.sum() / (grouped.abs().sum() + 1e-8)

    return {
        "mse": (grouped - quantized).pow(2).mean().item(),
        "mae": error.mean().item(),
        "rel_error": rel_error.item(),
        "max_error": error.max().item(),
    }


def analyze_weight_distribution(weight: torch.Tensor) -> dict:
    """Compute distribution statistics."""
    w = weight.float().flatten().cpu().numpy()
    return {
        "mean": float(np.mean(w)),
        "std": float(np.std(w)),
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "kurtosis": float(kurtosis(w)),
        "outlier_ratio": float(np.mean(np.abs(w) > 3 * np.std(w))),
        "range_ratio": float(np.max(np.abs(w)) / (np.std(w) + 1e-8)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.6B")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers to analyze")
    args = parser.parse_args()

    print(f"Loading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    n_layers = min(args.layers, len(model.model.layers))

    print(f"\nAnalyzing {n_layers} layers: original vs SpinQuant rotated\n")
    print("=" * 90)
    print(f"{'Layer':<40} | {'Kurtosis':>8} | {'Outlier%':>8} | {'FP4 RelErr':>10} | {'Δ':>8}")
    print("=" * 90)

    results = []

    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]

        # Analyze key projections
        for name, proj in [
            ("q_proj", layer.self_attn.q_proj),
            ("k_proj", layer.self_attn.k_proj),
            ("v_proj", layer.self_attn.v_proj),
            ("o_proj", layer.self_attn.o_proj),
            ("gate_proj", layer.mlp.gate_proj),
            ("up_proj", layer.mlp.up_proj),
            ("down_proj", layer.mlp.down_proj),
        ]:
            weight = proj.weight.data.clone()

            # Original stats
            orig_dist = analyze_weight_distribution(weight)
            orig_fp4 = compute_fp4_error(weight)

            # Rotated stats
            rotated = apply_hadamard_rotation(weight)
            rot_dist = analyze_weight_distribution(rotated)
            rot_fp4 = compute_fp4_error(rotated)

            # Improvement
            delta = (orig_fp4["rel_error"] - rot_fp4["rel_error"]) / orig_fp4["rel_error"] * 100

            layer_name = f"layers.{layer_idx}.{name}"
            print(f"{layer_name:<40} | {orig_dist['kurtosis']:>8.2f} | {orig_dist['outlier_ratio']*100:>7.3f}% | {orig_fp4['rel_error']:>10.6f} | ", end="")
            print(f"{rot_dist['kurtosis']:>8.2f} | {rot_dist['outlier_ratio']*100:>7.3f}% | {rot_fp4['rel_error']:>10.6f} | {delta:>+7.1f}%")

            results.append({
                "layer": layer_name,
                "orig_kurtosis": orig_dist["kurtosis"],
                "rot_kurtosis": rot_dist["kurtosis"],
                "orig_outlier": orig_dist["outlier_ratio"],
                "rot_outlier": rot_dist["outlier_ratio"],
                "orig_fp4_err": orig_fp4["rel_error"],
                "rot_fp4_err": rot_fp4["rel_error"],
                "improvement": delta,
            })

    print("=" * 90)

    # Summary
    avg_orig_err = np.mean([r["orig_fp4_err"] for r in results])
    avg_rot_err = np.mean([r["rot_fp4_err"] for r in results])
    avg_improvement = (avg_orig_err - avg_rot_err) / avg_orig_err * 100

    print(f"\nSummary:")
    print(f"  Average original FP4 error:  {avg_orig_err:.6f}")
    print(f"  Average rotated FP4 error:   {avg_rot_err:.6f}")
    print(f"  Average improvement:         {avg_improvement:+.1f}%")
    print(f"  Kurtosis reduction:          {np.mean([r['orig_kurtosis'] for r in results]):.2f} → {np.mean([r['rot_kurtosis'] for r in results]):.2f}")


if __name__ == "__main__":
    main()
