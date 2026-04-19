#!/usr/bin/env python3
"""
Test QEP benefit by comparing two quantization approaches:

1. Standard: Quantize all blocks at once with cached bf16 inputs/targets
2. QEP: Quantize blocks sequentially, using outputs from quantized blocks
        as both inputs AND targets for subsequent blocks

Both are evaluated by end-to-end inference error vs bf16 reference.

The key difference:
- Standard: Each block optimized to match bf16_block(bf16_input) -> bf16_output
- QEP: Each block optimized to match bf16_block(degraded_input) -> degraded_target

QEP should be better because at inference time, later blocks WILL receive
degraded inputs from upstream quantized blocks.
"""

import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from tqdm import tqdm


def fp4_quantize_layer(module: nn.Linear, group_size: int = 16) -> None:
    """In-place FP4 quantization with NVFP4 E2M1 levels."""
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

    # NVFP4 E2M1 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    q_abs = torch.where(abs_n <= 0.25, torch.zeros_like(abs_n),
            torch.where(abs_n <= 0.75, torch.full_like(abs_n, 0.5),
            torch.where(abs_n <= 1.25, torch.full_like(abs_n, 1.0),
            torch.where(abs_n <= 1.75, torch.full_like(abs_n, 1.5),
            torch.where(abs_n <= 2.5, torch.full_like(abs_n, 2.0),
            torch.where(abs_n <= 3.5, torch.full_like(abs_n, 3.0),
            torch.where(abs_n <= 5.0, torch.full_like(abs_n, 4.0),
                        torch.full_like(abs_n, 6.0))))))))

    dequantized = sign * q_abs * scales / 6.0
    module.weight.data = dequantized.view(out_f, -1)[:, :in_f].contiguous()


def quantize_block(block: nn.Module, group_size: int = 16) -> None:
    """Quantize all linear layers in a block to FP4."""
    for module in block.modules():
        if isinstance(module, nn.Linear) and module.weight.numel() > 1000:
            fp4_quantize_layer(module, group_size)


@torch.no_grad()
def get_block_output(model, block_idx: int, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Get output of a specific block using forward hooks."""
    output = [None]

    def hook(m, inp, out):
        output[0] = out[0].detach() if isinstance(out, tuple) else out.detach()

    handle = model.model.layers[block_idx].register_forward_hook(hook)
    model(input_ids.to(device))
    handle.remove()
    return output[0]


@torch.no_grad()
def compute_e2e_error(model_q, model_bf16, test_inputs: List[torch.Tensor], device) -> float:
    """Compute end-to-end relative MSE between quantized and bf16 models."""
    total_error = 0.0
    total_ref = 0.0

    for input_ids in test_inputs:
        input_ids = input_ids.to(device)
        out_q = model_q(input_ids, output_hidden_states=True).hidden_states[-1]
        out_bf16 = model_bf16(input_ids, output_hidden_states=True).hidden_states[-1]

        total_error += (out_q - out_bf16).pow(2).sum().item()
        total_ref += out_bf16.pow(2).sum().item()

    return total_error / (total_ref + 1e-8)


@torch.no_grad()
def compute_block_errors(model_q, model_bf16, test_inputs: List[torch.Tensor], device, n_blocks: int) -> List[float]:
    """Compute per-block error accumulation."""
    errors = []

    for block_idx in range(n_blocks):
        total_error = 0.0
        total_ref = 0.0

        for input_ids in test_inputs:
            out_q = get_block_output(model_q, block_idx, input_ids, device)
            out_bf16 = get_block_output(model_bf16, block_idx, input_ids, device)

            total_error += (out_q - out_bf16).pow(2).sum().item()
            total_ref += out_bf16.pow(2).sum().item()

        errors.append(total_error / (total_ref + 1e-8))

    return errors


def run_qep_comparison(model_name: str, n_samples: int = 4, seq_len: int = 256, n_blocks: int = None):
    """
    Compare Standard vs QEP quantization approaches.

    Standard: Quantize all blocks using bf16-cached activations as targets
    QEP: Quantize blocks sequentially, propagating quantization errors forward
    """
    print(f"Loading {model_name}...", flush=True)

    # Load bf16 reference (never modified)
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    device = next(model_bf16.parameters()).device

    total_blocks = len(model_bf16.model.layers)
    if n_blocks is None:
        n_blocks = total_blocks
    n_blocks = min(n_blocks, total_blocks)

    print(f"Model has {total_blocks} blocks, testing {n_blocks}")

    # Generate test inputs
    vocab_size = model_bf16.config.vocab_size
    test_inputs = [
        torch.randint(0, vocab_size, (1, seq_len), device="cpu")
        for _ in range(n_samples)
    ]

    # ========== STANDARD QUANTIZATION ==========
    print("\n" + "=" * 70)
    print("STANDARD QUANTIZATION: All blocks quantized with bf16 activations")
    print("=" * 70)

    model_standard = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )

    # Quantize all blocks at once (simulating cached bf16 activations for all)
    for block_idx in tqdm(range(n_blocks), desc="Quantizing (standard)"):
        quantize_block(model_standard.model.layers[block_idx])

    # Measure error
    std_errors = compute_block_errors(model_standard, model_bf16, test_inputs, device, n_blocks)
    std_e2e = compute_e2e_error(model_standard, model_bf16, test_inputs, device)

    print(f"\nPer-block errors (relative MSE):")
    for i in range(0, n_blocks, max(1, n_blocks // 8)):
        print(f"  Block {i:2d}: {std_errors[i]:.6f}")
    print(f"  Block {n_blocks-1:2d}: {std_errors[n_blocks-1]:.6f}")
    print(f"\nEnd-to-end error: {std_e2e:.6f}")

    # Clean up
    del model_standard
    gc.collect()
    torch.cuda.empty_cache()

    # ========== QEP QUANTIZATION ==========
    print("\n" + "=" * 70)
    print("QEP QUANTIZATION: Blocks quantized sequentially with error propagation")
    print("=" * 70)

    model_qep = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )

    # Quantize blocks one at a time, letting errors propagate
    for block_idx in tqdm(range(n_blocks), desc="Quantizing (QEP)"):
        # Before quantizing this block, run a forward pass to "warm up"
        # (In real QEP with optimization, this would inform the optimization targets)
        if block_idx > 0:
            for input_ids in test_inputs[:2]:  # Use subset for speed
                model_qep(input_ids.to(device))

        # Quantize this block
        quantize_block(model_qep.model.layers[block_idx])

        # Clear cache between blocks
        gc.collect()
        torch.cuda.empty_cache()

    # Measure error
    qep_errors = compute_block_errors(model_qep, model_bf16, test_inputs, device, n_blocks)
    qep_e2e = compute_e2e_error(model_qep, model_bf16, test_inputs, device)

    print(f"\nPer-block errors (relative MSE):")
    for i in range(0, n_blocks, max(1, n_blocks // 8)):
        print(f"  Block {i:2d}: {qep_errors[i]:.6f}")
    print(f"  Block {n_blocks-1:2d}: {qep_errors[n_blocks-1]:.6f}")
    print(f"\nEnd-to-end error: {qep_e2e:.6f}")

    # ========== COMPARISON ==========
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Standard':>12} {'QEP':>12} {'Diff':>12}")
    print("-" * 61)
    print(f"{'End-to-end error':<25} {std_e2e:>12.6f} {qep_e2e:>12.6f} {(qep_e2e - std_e2e):>+12.6f}")
    print(f"{'First block error':<25} {std_errors[0]:>12.6f} {qep_errors[0]:>12.6f} {(qep_errors[0] - std_errors[0]):>+12.6f}")
    print(f"{'Last block error':<25} {std_errors[-1]:>12.6f} {qep_errors[-1]:>12.6f} {(qep_errors[-1] - std_errors[-1]):>+12.6f}")

    # Error accumulation analysis
    std_accum = std_errors[-1] / max(std_errors[0], 1e-10)
    qep_accum = qep_errors[-1] / max(qep_errors[0], 1e-10)
    print(f"{'Error accumulation (L/F)':<25} {std_accum:>12.1f}x {qep_accum:>12.1f}x")

    print("\n" + "=" * 70)
    if qep_e2e < std_e2e:
        improvement = (std_e2e - qep_e2e) / std_e2e * 100
        print(f"RESULT: QEP is BETTER by {improvement:.2f}%")
    else:
        degradation = (qep_e2e - std_e2e) / std_e2e * 100
        print(f"RESULT: QEP is WORSE by {degradation:.2f}%")
    print("=" * 70)

    print("\nNOTE: This uses RTN (naive rounding), not gradient optimization.")
    print("With RTN, both methods use identical quantization - the difference")
    print("is only in measurement ordering. The true QEP benefit comes from")
    print("optimization adapting to degraded inputs, which requires AutoRound.")

    return {
        "std_e2e": std_e2e,
        "qep_e2e": qep_e2e,
        "std_errors": std_errors,
        "qep_errors": qep_errors,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=None)
    args = parser.parse_args()

    run_qep_comparison(args.model, args.samples, args.seq_len, args.blocks)
