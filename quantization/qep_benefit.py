#!/usr/bin/env python3
"""
Quantify the benefit of QEP vs standard quantization.

Compares:
1. Standard: Quantize each block using cached bf16 inputs
2. QEP: Quantize each block using degraded inputs from quantized upstream

Both models are then evaluated on the SAME test data against bf16 reference.
"""

import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import copy


def fp4_quantize_layer(module: nn.Linear, group_size: int = 16) -> None:
    """In-place FP4 quantization of a linear layer."""
    weight = module.weight.data
    out_f, in_f = weight.shape

    n_groups = (in_f + group_size - 1) // group_size
    pad_size = n_groups * group_size - in_f
    if pad_size > 0:
        weight_padded = torch.nn.functional.pad(weight, (0, pad_size))
    else:
        weight_padded = weight

    grouped = weight_padded.view(out_f, n_groups, group_size)
    scales = grouped.abs().max(dim=2, keepdim=True).values.clamp(min=1e-8)
    normalized = grouped / scales * 6.0
    abs_n = normalized.abs()
    sign = normalized.sign()

    q_abs = torch.where(abs_n <= 2.0, (abs_n * 2).round() / 2,
            torch.where(abs_n <= 2.5, torch.full_like(abs_n, 2.0),
            torch.where(abs_n <= 3.5, torch.full_like(abs_n, 3.0),
            torch.where(abs_n <= 5.0, torch.full_like(abs_n, 4.0),
                        torch.full_like(abs_n, 6.0)))))

    dequantized = sign * q_abs * scales / 6.0
    module.weight.data = dequantized.view(out_f, -1)[:, :in_f].contiguous()


def quantize_block(block: nn.Module, group_size: int = 16) -> None:
    """Quantize all linear layers in a block."""
    for module in block.modules():
        if isinstance(module, nn.Linear) and module.weight.numel() > 1000:
            fp4_quantize_layer(module, group_size)


@torch.no_grad()
def get_model_output(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Get final hidden states from model."""
    outputs = model(input_ids, output_hidden_states=True)
    return outputs.hidden_states[-1]


@torch.no_grad()
def compute_output_error(
    model_quant: nn.Module,
    model_bf16: nn.Module,
    test_inputs: List[torch.Tensor],
    device: torch.device,
) -> float:
    """Compute relative MSE between quantized and bf16 model outputs."""
    total_error = 0.0
    total_ref = 0.0

    for input_ids in test_inputs:
        input_ids = input_ids.to(device)

        out_q = get_model_output(model_quant, input_ids)
        out_bf16 = get_model_output(model_bf16, input_ids)

        error = (out_q - out_bf16).pow(2).sum().item()
        ref = out_bf16.pow(2).sum().item()

        total_error += error
        total_ref += ref

    return total_error / (total_ref + 1e-8)


def quantize_standard(model: nn.Module) -> None:
    """
    Standard quantization: quantize all blocks.
    (In real AutoRound, each block would be optimized with cached bf16 inputs.
    Here we just do RTN to isolate the input-propagation effect.)
    """
    blocks = model.model.layers
    for block in blocks:
        quantize_block(block)


def quantize_qep(model: nn.Module, calibration_inputs: List[torch.Tensor], device: torch.device) -> None:
    """
    QEP quantization: quantize blocks sequentially, running inference
    through already-quantized blocks before quantizing the next one.

    This ensures each block's quantization "sees" the degraded inputs
    it will receive during actual inference.
    """
    blocks = model.model.layers

    for block_idx, block in enumerate(blocks):
        # Run calibration inputs through model to "warm up" with current state
        # (In real QEP, this would inform the optimization. Here it's a no-op
        # since we're doing RTN, but we still propagate to maintain the pattern.)
        if block_idx > 0:
            for input_ids in calibration_inputs:
                input_ids = input_ids.to(device)
                with torch.no_grad():
                    _ = model(input_ids)

        # Quantize this block
        quantize_block(block)

        gc.collect()
        torch.cuda.empty_cache()


def run_comparison(model_name: str, n_samples: int = 8, seq_len: int = 256):
    """Compare standard vs QEP quantization."""

    print(f"Loading {model_name}...", flush=True)

    # Load bf16 reference (never modified)
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )

    # Load two copies for quantization
    model_standard = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model_qep = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    device = next(model_bf16.parameters()).device

    # Generate random test inputs
    vocab_size = model_bf16.config.vocab_size
    test_inputs = [
        torch.randint(0, vocab_size, (1, seq_len), device="cpu")
        for _ in range(n_samples)
    ]

    print(f"Using {n_samples} test samples, seq_len={seq_len}")

    # Baseline: bf16 error (should be ~0)
    print("\nComputing bf16 self-error (sanity check)...", flush=True)
    bf16_error = compute_output_error(model_bf16, model_bf16, test_inputs, device)
    print(f"  bf16 self-error: {bf16_error:.10f} (should be ~0)")

    # Standard quantization
    print("\nQuantizing with STANDARD approach...", flush=True)
    quantize_standard(model_standard)
    standard_error = compute_output_error(model_standard, model_bf16, test_inputs, device)
    print(f"  Standard error: {standard_error:.6f}")

    # QEP quantization
    print("\nQuantizing with QEP approach...", flush=True)
    quantize_qep(model_qep, test_inputs, device)
    qep_error = compute_output_error(model_qep, model_bf16, test_inputs, device)
    print(f"  QEP error: {qep_error:.6f}")

    # Comparison
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Standard quantization error: {standard_error:.6f}")
    print(f"QEP quantization error:      {qep_error:.6f}")

    if qep_error < standard_error:
        improvement = (standard_error - qep_error) / standard_error * 100
        print(f"\nQEP is BETTER by {improvement:.1f}%")
    else:
        degradation = (qep_error - standard_error) / standard_error * 100
        print(f"\nQEP is WORSE by {degradation:.1f}%")

    print("\nNote: This uses RTN (naive rounding), not optimized quantization.")
    print("QEP benefit should be larger with gradient-based optimization")
    print("because the optimizer can actually adapt to degraded inputs.")

    return {
        "standard_error": standard_error,
        "qep_error": qep_error,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    run_comparison(args.model, args.samples, args.seq_len)
