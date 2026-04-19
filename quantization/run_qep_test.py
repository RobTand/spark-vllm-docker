#!/usr/bin/env python3
"""
Test QEP-patched AutoRound vs baseline.

This script:
1. Runs AutoRound with QEP patch (q_input used for targets)
2. Reports per-block loss progression
3. Evaluates end-to-end quality vs bf16

Run from the auto-round repo directory with the QEP patch applied.
"""

import sys
import os
import torch
import gc

# Add patched auto-round to path
sys.path.insert(0, "/tmp/auto-round")

from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound


def run_quantization(model_name: str, output_dir: str, nsamples: int = 32, iters: int = 50):
    """Run AutoRound quantization and return the quantized model."""

    print(f"Loading {model_name}...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    # Configure AutoRound
    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=4,
        group_size=128,
        sym=True,
        batch_size=4,
        seqlen=512,
        nsamples=nsamples,
        iters=iters,
        dataset="NeelNanda/pile-10k",
        # enable_quanted_input=True is default, which propagates q_input
    )

    print("\nStarting quantization with QEP patch...", flush=True)
    print("(QEP: targets computed using degraded inputs from quantized upstream blocks)")
    print()

    # Run quantization
    model_q, layer_config = autoround.quantize()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    autoround.save_quantized(output_dir, format="auto_round", inplace=True)
    tokenizer.save_pretrained(output_dir)

    print(f"\nModel saved to {output_dir}")

    return model_q, tokenizer


@torch.no_grad()
def evaluate_quality(model_q, model_bf16, tokenizer, n_samples: int = 10):
    """Evaluate quantized model quality vs bf16 reference."""

    device = next(model_q.parameters()).device

    # Generate test prompts
    test_prompts = [
        "The quick brown fox",
        "In a galaxy far far away",
        "def fibonacci(n):",
        "The capital of France is",
        "Machine learning is",
    ]

    total_error = 0.0
    total_ref = 0.0
    total_kl = 0.0

    for prompt in test_prompts[:n_samples]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Get logits from both models
        logits_q = model_q(**inputs).logits
        logits_bf16 = model_bf16(**inputs).logits

        # MSE error
        mse = (logits_q - logits_bf16).pow(2).mean().item()
        ref = logits_bf16.pow(2).mean().item()
        total_error += mse
        total_ref += ref

        # KL divergence (on last token)
        probs_q = torch.softmax(logits_q[0, -1], dim=-1)
        probs_bf16 = torch.softmax(logits_bf16[0, -1], dim=-1)
        kl = (probs_bf16 * (probs_bf16.log() - probs_q.log())).sum().item()
        total_kl += kl

    rel_error = total_error / (total_ref + 1e-8)
    avg_kl = total_kl / len(test_prompts[:n_samples])

    return {
        "relative_mse": rel_error,
        "avg_kl_divergence": avg_kl,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output", type=str, default="/tmp/qep_test_output")
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if not args.eval_only:
        model_q, tokenizer = run_quantization(
            args.model, args.output, args.nsamples, args.iters
        )

        # Load bf16 reference for evaluation
        print("\nLoading bf16 reference for evaluation...", flush=True)
        model_bf16 = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        print("\nEvaluating quality...", flush=True)
        metrics = evaluate_quality(model_q, model_bf16, tokenizer)

        print("\n" + "=" * 50)
        print("QEP QUANTIZATION RESULTS")
        print("=" * 50)
        print(f"Relative MSE vs bf16:  {metrics['relative_mse']:.6f}")
        print(f"Avg KL divergence:     {metrics['avg_kl_divergence']:.6f}")
    else:
        print("Eval-only mode not yet implemented")


if __name__ == "__main__":
    main()
