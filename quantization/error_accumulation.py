#!/usr/bin/env python3
"""
Measure error accumulation through quantized layers.

Shows how quantization error compounds layer-over-layer when running
inference through already-quantized upstream layers. This is the problem
that QEP (Quantization Error Propagation) addresses during optimization.
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def fp4_quantize_inplace(module: torch.nn.Linear, group_size: int = 16) -> None:
    """In-place quantize a linear layer to FP4 E2M1."""
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


def run_analysis(model_name: str, n_samples: int = 8, seq_len: int = 256):
    print(f"Loading {model_name}...", flush=True)

    # Load bf16 reference
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)

    # Load quantized version
    model_q = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    n_layers = len(model_bf16.model.layers)

    # Quantize all layers upfront
    print("Quantizing model...", flush=True)
    for layer in model_q.model.layers:
        for module in layer.modules():
            if isinstance(module, torch.nn.Linear) and module.weight.numel() > 1000:
                fp4_quantize_inplace(module)

    # Generate random input tokens for calibration
    print("Generating calibration data...", flush=True)
    vocab_size = model_bf16.config.vocab_size
    inputs = [torch.randint(0, vocab_size, (1, seq_len), device="cuda") for _ in range(n_samples)]
    print(f"Using {len(inputs)} random samples\n")

    # Measure error at each layer using hooks
    print("=" * 70)
    print(f"{'Layer':<8} | {'Output Error':>14} | {'Cumulative':>12} | {'vs Layer 0':>12}")
    print("=" * 70)

    results = []
    layer_0_error = None

    # Collect hidden states using hooks
    bf16_states = {}
    q_states = {}

    def make_hook(storage, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[layer_idx] = output[0].detach()
            else:
                storage[layer_idx] = output.detach()
        return hook

    # Register hooks
    bf16_hooks = []
    q_hooks = []
    for i in range(n_layers):
        bf16_hooks.append(model_bf16.model.layers[i].register_forward_hook(make_hook(bf16_states, i)))
        q_hooks.append(model_q.model.layers[i].register_forward_hook(make_hook(q_states, i)))

    # Run forward passes
    with torch.no_grad():
        for input_ids in inputs:
            _ = model_bf16(input_ids)
            _ = model_q(input_ids)

            for layer_idx in range(n_layers):
                h_bf16 = bf16_states[layer_idx]
                h_q = q_states[layer_idx]
                error = (h_bf16 - h_q).pow(2).mean() / (h_bf16.pow(2).mean() + 1e-8)

                if layer_idx not in [r["layer"] for r in results]:
                    results.append({"layer": layer_idx, "errors": []})

                for r in results:
                    if r["layer"] == layer_idx:
                        r["errors"].append(error.item())

    # Remove hooks
    for h in bf16_hooks + q_hooks:
        h.remove()

    # Compute averages and print
    for r in results:
        r["error"] = np.mean(r["errors"])
        if r["layer"] == 0:
            layer_0_error = r["error"]
        r["ratio"] = r["error"] / layer_0_error if layer_0_error else 1.0

    for r in sorted(results, key=lambda x: x["layer"]):
        print(f"Layer {r['layer']:<3} | {r['error']:>14.6f} | {r['error']:>12.6f} | {r['ratio']:>11.1f}x")

    print("=" * 70)

    # Compute growth rate
    errors = [r["error"] for r in results]
    log_errors = np.log(np.array(errors) + 1e-10)
    growth_rate = np.polyfit(range(len(errors)), log_errors, 1)[0]

    print(f"\nError accumulation analysis:")
    print(f"  Layer 0 error:     {results[0]['error']:.6f}")
    print(f"  Final layer error: {results[-1]['error']:.6f}")
    print(f"  Accumulation:      {results[-1]['ratio']:.1f}x")
    print(f"  Growth rate:       {np.exp(growth_rate):.3f}x per layer")
    print()
    print("This exponential growth is what QEP addresses by optimizing each")
    print("layer's quantization against realistic degraded inputs, not clean bf16.")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    run_analysis(args.model, args.samples, args.seq_len)


if __name__ == "__main__":
    main()
