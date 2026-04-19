#!/usr/bin/env python3
"""
QEP (Quantization Error Propagation) comparison.

Compares reconstruction error between:
1. Standard: Each layer optimized against original bf16 inputs
2. QEP-aware: Each layer optimized against degraded inputs from quantized upstream

Uses simple RTN quantization to isolate the QEP effect.
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def fp4_quantize(weight: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Quantize weight to FP4 E2M1 format and return dequantized version."""
    orig_shape = weight.shape
    out_f, in_f = weight.shape

    # Pad to group size
    n_groups = (in_f + group_size - 1) // group_size
    pad_size = n_groups * group_size - in_f
    if pad_size > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_size))

    # Reshape to groups
    grouped = weight.view(out_f, n_groups, group_size)

    # Per-group scale
    scales = grouped.abs().max(dim=2, keepdim=True).values.clamp(min=1e-8)

    # Normalize to [-6, 6] (FP4 E2M1 range)
    normalized = grouped / scales * 6.0
    abs_n = normalized.abs()
    sign = normalized.sign()

    # Quantize to FP4 values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
    q_abs = torch.where(abs_n <= 2.0, (abs_n * 2).round() / 2,
            torch.where(abs_n <= 2.5, torch.full_like(abs_n, 2.0),
            torch.where(abs_n <= 3.5, torch.full_like(abs_n, 3.0),
            torch.where(abs_n <= 5.0, torch.full_like(abs_n, 4.0),
                        torch.full_like(abs_n, 6.0)))))

    # Dequantize
    dequantized = sign * q_abs * scales / 6.0
    dequantized = dequantized.view(out_f, -1)[:, :in_f]

    return dequantized


def quantize_linear_layer(module: torch.nn.Linear) -> None:
    """In-place quantize a linear layer to FP4."""
    with torch.no_grad():
        module.weight.data = fp4_quantize(module.weight.data)


def get_layer_output(model, layer_idx: int, hidden_states: torch.Tensor,
                     attention_mask: torch.Tensor = None, position_ids: torch.Tensor = None):
    """Get output from a specific layer."""
    layer = model.model.layers[layer_idx]

    # Handle different layer types
    with torch.no_grad():
        if hasattr(layer, 'linear_attn'):
            # GatedDeltaNet layer
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = layer.linear_attn(hidden_states)[0]
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            # Standard attention layer - use the layer's forward
            outputs = layer(hidden_states, attention_mask=attention_mask,
                          position_ids=position_ids)
            hidden_states = outputs[0]

    return hidden_states


def compute_layer_error(orig_output: torch.Tensor, quant_output: torch.Tensor) -> float:
    """Compute relative MSE between original and quantized outputs."""
    mse = (orig_output - quant_output).pow(2).mean()
    ref = orig_output.pow(2).mean()
    return (mse / (ref + 1e-8)).item()


def run_comparison(model_name: str, n_samples: int = 32, seq_len: int = 512, n_layers: int = 8):
    """Run QEP comparison experiment."""

    print(f"Loading {model_name}...", flush=True)

    # Load model and tokenizer
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Clone for quantized version
    model_quant = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    total_layers = len(model_bf16.model.layers)
    n_layers = min(n_layers, total_layers)

    print(f"Model has {total_layers} layers, analyzing first {n_layers}")

    # Load calibration data
    print("Loading calibration data...", flush=True)
    dataset = load_dataset("NeelNanda/pile-10k", split="train")

    # Prepare calibration samples
    calibration_inputs = []
    for i in range(min(n_samples * 2, len(dataset))):
        text = dataset[i]["text"]
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if tokens["input_ids"].shape[1] >= seq_len // 2:
            calibration_inputs.append(tokens["input_ids"].cuda())
            if len(calibration_inputs) >= n_samples:
                break

    print(f"Using {len(calibration_inputs)} calibration samples")

    # Get embeddings for all samples
    print("Computing embeddings...", flush=True)
    with torch.no_grad():
        all_embeds_bf16 = []
        for input_ids in calibration_inputs:
            embeds = model_bf16.model.embed_tokens(input_ids)
            all_embeds_bf16.append(embeds)

    print()
    print("=" * 80)
    print(f"{'Layer':<10} | {'Standard Error':>15} | {'QEP Error':>15} | {'Improvement':>12}")
    print("=" * 80)

    results = []

    # Track activations through quantized model for QEP
    qep_hidden_states = [e.clone() for e in all_embeds_bf16]

    for layer_idx in range(n_layers):
        layer_bf16 = model_bf16.model.layers[layer_idx]
        layer_quant = model_quant.model.layers[layer_idx]

        # === Standard approach: measure error against bf16 inputs ===
        # Get bf16 outputs with bf16 inputs (ground truth)
        bf16_outputs = []
        with torch.no_grad():
            for embeds in all_embeds_bf16:
                # Run through all layers up to this one
                h = embeds
                for i in range(layer_idx + 1):
                    h = get_layer_output(model_bf16, i, h)
                bf16_outputs.append(h)

        # Quantize this layer in the quant model
        for name, module in layer_quant.named_modules():
            if isinstance(module, torch.nn.Linear):
                if module.weight.numel() > 1000:  # Skip tiny layers
                    quantize_linear_layer(module)

        # Standard: measure error using bf16 inputs up to this layer, then quant layer
        standard_errors = []
        with torch.no_grad():
            for embeds, bf16_out in zip(all_embeds_bf16, bf16_outputs):
                # Run bf16 up to layer_idx-1, then quantized layer
                h = embeds
                for i in range(layer_idx):
                    h = get_layer_output(model_bf16, i, h)
                # Now run through quantized layer
                h_quant = get_layer_output(model_quant, layer_idx, h)
                standard_errors.append(compute_layer_error(bf16_out, h_quant))

        standard_error = np.mean(standard_errors)

        # === QEP approach: measure error against already-degraded inputs ===
        qep_errors = []
        with torch.no_grad():
            for i, (qep_h, bf16_out) in enumerate(zip(qep_hidden_states, bf16_outputs)):
                # Run quantized layer with QEP (degraded) inputs
                h_qep = get_layer_output(model_quant, layer_idx, qep_h)
                qep_errors.append(compute_layer_error(bf16_out, h_qep))
                # Update QEP hidden states for next layer
                qep_hidden_states[i] = h_qep

        qep_error = np.mean(qep_errors)

        # Compute improvement (negative means QEP is worse, which is expected
        # since we're comparing to bf16 ground truth, not optimizing)
        improvement = (standard_error - qep_error) / standard_error * 100

        print(f"Layer {layer_idx:<4} | {standard_error:>15.6f} | {qep_error:>15.6f} | {improvement:>+11.1f}%")

        results.append({
            "layer": layer_idx,
            "standard_error": standard_error,
            "qep_error": qep_error,
            "improvement": improvement,
        })

    print("=" * 80)

    # Summary
    avg_std = np.mean([r["standard_error"] for r in results])
    avg_qep = np.mean([r["qep_error"] for r in results])

    print(f"\nSummary:")
    print(f"  Average standard error: {avg_std:.6f}")
    print(f"  Average QEP error:      {avg_qep:.6f}")
    print(f"  Error ratio (QEP/std):  {avg_qep/avg_std:.2f}x")
    print()
    print("Note: QEP error is higher because we're comparing to bf16 ground truth.")
    print("The value of QEP is that layers optimize for realistic degraded inputs,")
    print("not that raw error is lower. True QEP benefit requires re-optimization.")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    args = parser.parse_args()

    run_comparison(args.model, args.samples, args.seq_len, args.layers)


if __name__ == "__main__":
    main()
