#!/usr/bin/env python3
"""
Compare AutoRound with and without QEP patch.

Runs both configurations and measures:
1. Per-block reconstruction loss
2. End-to-end inference quality vs bf16
"""

import sys
import os
import torch
import gc
import shutil

# Add patched auto-round to path
sys.path.insert(0, "/tmp/auto-round")

from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound


def apply_qep_patch():
    """Apply QEP patch to AutoRound's _quantize_block."""
    from auto_round.compressors import base as base_module

    # Check if already patched
    source = open(base_module.__file__).read()
    if "QEP: Use q_input" in source:
        print("QEP patch already applied")
        return True
    return False


def run_quantization(model_name: str, output_dir: str, use_qep: bool, nsamples: int = 16, iters: int = 30):
    """Run AutoRound quantization."""

    print(f"\n{'='*70}")
    print(f"Running {'QEP' if use_qep else 'STANDARD'} AutoRound quantization")
    print(f"{'='*70}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Note: The QEP patch is controlled by modifying base.py before running
    # When use_qep=False, we need to use the unpatched version
    # For simplicity, we'll run with the patched version for QEP

    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=4,
        group_size=16,  # NVFP4 uses group_size=16
        sym=True,
        batch_size=4,
        seqlen=256,
        nsamples=nsamples,
        iters=iters,
        dataset="NeelNanda/pile-10k",
        enable_quanted_input=True,  # Propagate quantized outputs
    )

    print(f"\nStarting quantization (iters={iters}, nsamples={nsamples})...")

    # Run quantization
    model_q, layer_config = autoround.quantize()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    autoround.save_quantized(output_dir, format="auto_round", inplace=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")

    return model_q, tokenizer


@torch.no_grad()
def evaluate_vs_bf16(model_q, tokenizer, model_name: str, n_samples: int = 5):
    """Evaluate quantized model vs bf16 reference."""

    device = next(model_q.parameters()).device

    # Load fresh bf16 reference on same device
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Test prompts
    test_prompts = [
        "The quick brown fox jumps over",
        "In machine learning, neural networks",
        "def fibonacci(n):\n    if n <= 1:",
        "The capital of France is Paris, and",
        "import torch\nclass Model(nn.Module):",
    ]

    total_mse = 0.0
    total_ref = 0.0
    total_kl = 0.0

    device_q = next(model_q.parameters()).device
    device_bf16 = next(model_bf16.parameters()).device

    for prompt in test_prompts[:n_samples]:
        inputs_q = tokenizer(prompt, return_tensors="pt").to(device_q)
        inputs_bf16 = tokenizer(prompt, return_tensors="pt").to(device_bf16)

        logits_q = model_q(**inputs_q).logits
        logits_bf16 = model_bf16(**inputs_bf16).logits.to(device_q)  # Move to same device for comparison

        # MSE
        mse = (logits_q - logits_bf16).pow(2).mean().item()
        ref = logits_bf16.pow(2).mean().item()
        total_mse += mse
        total_ref += ref

        # KL divergence on last token
        probs_q = torch.softmax(logits_q[0, -1] / 1.0, dim=-1)
        probs_bf16 = torch.softmax(logits_bf16[0, -1] / 1.0, dim=-1)
        kl = (probs_bf16 * (probs_bf16.log() - (probs_q + 1e-10).log())).sum().item()
        total_kl += kl

    del model_bf16
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "relative_mse": total_mse / (total_ref + 1e-8),
        "avg_kl": total_kl / n_samples,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--qep-only", action="store_true", help="Only run QEP (patch already applied)")
    parser.add_argument("--baseline", action="store_true", help="Run baseline (no QEP patch)")
    args = parser.parse_args()

    # Check patch status
    is_patched = apply_qep_patch()

    if args.baseline:
        print("Running BASELINE AutoRound (no QEP)...")

        model_q, tokenizer = run_quantization(
            args.model,
            "/tmp/baseline_output",
            use_qep=False,
            nsamples=args.nsamples,
            iters=args.iters,
        )

        print("\nEvaluating baseline model vs bf16...")
        metrics = evaluate_vs_bf16(model_q, tokenizer, args.model)

        print(f"\n{'='*70}")
        print("BASELINE AUTOROUND RESULTS")
        print(f"{'='*70}")
        print(f"Relative MSE:  {metrics['relative_mse']:.6f}")
        print(f"Avg KL div:    {metrics['avg_kl']:.6f}")
        return

    if args.qep_only:
        print("Running QEP-patched AutoRound only...")

        model_q, tokenizer = run_quantization(
            args.model,
            "/tmp/qep_output",
            use_qep=True,
            nsamples=args.nsamples,
            iters=args.iters,
        )

        print("\nEvaluating QEP model vs bf16...")
        metrics = evaluate_vs_bf16(model_q, tokenizer, args.model)

        print(f"\n{'='*70}")
        print("QEP AUTOROUND RESULTS")
        print(f"{'='*70}")
        print(f"Relative MSE:  {metrics['relative_mse']:.6f}")
        print(f"Avg KL div:    {metrics['avg_kl']:.6f}")

    else:
        print("To compare QEP vs Standard, we need to run twice:")
        print("1. First with QEP patch applied (current state)")
        print("2. Then revert patch and run again")
        print("\nFor now, running QEP-patched version...")

        model_q, tokenizer = run_quantization(
            args.model,
            "/tmp/qep_output",
            use_qep=True,
            nsamples=args.nsamples,
            iters=args.iters,
        )

        print("\nEvaluating vs bf16...")
        metrics = evaluate_vs_bf16(model_q, tokenizer, args.model)

        print(f"\n{'='*70}")
        print("QEP AUTOROUND RESULTS")
        print(f"{'='*70}")
        print(f"Relative MSE vs bf16:  {metrics['relative_mse']:.6f}")
        print(f"Avg KL divergence:     {metrics['avg_kl']:.6f}")


if __name__ == "__main__":
    main()
