#!/usr/bin/env python3
"""
NVFP4 quantization for Qwen3.5 models using AutoRound + llm-compressor.
Produces vLLM-compatible checkpoints with mixed precision support.

Variants:
1. all-fp4: Everything NVFP4 except embeds/lm_head/gates/norms
2. fp4-fp8-sensitive: FP4 everywhere, FP8 for top-N sensitive layers
3. fp4-bf16-critical: FP4 everywhere, bf16 for most critical layers
4. autoscheme: Let AutoRound budget optimizer choose

Usage:
    python quantize_nvfp4.py --model Qwen/Qwen3.5-27B --variant all-fp4 --output /models/output
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot


# Layers that should NEVER be quantized (structural/quality reasons)
ALWAYS_SKIP = [
    "lm_head",
    "embed_tokens",
    "re:.*mlp\\.gate$",           # MoE router
    "re:.*shared_expert_gate$",   # Shared expert gate
    "re:.*norm.*",                # All norms
    "re:.*A_log$",                # GDN decay
    "re:.*dt_bias$",              # GDN timestep
    "re:.*conv1d.*",              # GDN causal conv
]

# Layers known to be sensitive in Qwen3.5 GDN architecture
GDN_SENSITIVE = [
    "re:.*linear_attn\\.in_proj_a$",
    "re:.*linear_attn\\.in_proj_b$",
    "re:.*linear_attn\\.in_proj_qkvz$",
    "re:.*linear_attn\\.in_proj_ba$",
    "re:.*linear_attn\\.out_proj$",
]


def load_sensitivity_scores(path: str) -> dict:
    """Load pre-computed sensitivity scores if available."""
    score_file = Path(path) / "sensitivity_scores.json"
    if score_file.exists():
        with open(score_file) as f:
            return json.load(f)
    return {}


def quantize_all_fp4(
    model_name: str,
    output_dir: str,
    nsamples: int = 256,
    seqlen: int = 4096,
):
    """
    Variant 1: Quantize everything to NVFP4 except structural exclusions.
    Maximum compression, maximum speed potential.
    """
    print(f"[all-fp4] Quantizing {model_name}")

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=ALWAYS_SKIP,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset="HuggingFaceH4/ultrachat_200k",
        recipe=recipe,
        max_seq_length=seqlen,
        num_calibration_samples=nsamples,
        output_dir=output_dir,
    )

    print(f"[all-fp4] Saved to {output_dir}")


def quantize_fp4_with_fp8_sensitive(
    model_name: str,
    output_dir: str,
    sensitivity_dir: str = None,
    top_n: int = 10,
    nsamples: int = 256,
    seqlen: int = 4096,
):
    """
    Variant 2: FP4 everywhere, but FP8 for the top-N most sensitive layers.
    Balances speed with quality for sensitive GDN layers.
    """
    print(f"[fp4-fp8-sensitive] Quantizing {model_name}")

    # Load sensitivity scores or use defaults
    sensitive_layers = list(GDN_SENSITIVE)  # Start with known sensitive
    if sensitivity_dir:
        scores = load_sensitivity_scores(sensitivity_dir)
        if "top_20_sensitive" in scores:
            sensitive_layers = scores["top_20_sensitive"][:top_n]

    print(f"Using FP8 for {len(sensitive_layers)} sensitive layers")

    # For llm-compressor, we need to use config_groups for mixed precision
    # This requires the modelopt_mixed format

    # First pass: quantize everything to FP4
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Build ignore list: always_skip + sensitive layers (they'll be FP8)
    fp4_ignore = ALWAYS_SKIP + sensitive_layers

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=fp4_ignore,
    )

    # First quantize the FP4 layers
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset="HuggingFaceH4/ultrachat_200k",
        recipe=recipe,
        max_seq_length=seqlen,
        num_calibration_samples=nsamples,
        output_dir=output_dir + "-step1",
    )

    # Second pass: quantize sensitive layers to FP8
    # Load the partially quantized model
    model = AutoModelForCausalLM.from_pretrained(
        output_dir + "-step1",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Target only the sensitive layers for FP8
    fp8_recipe = QuantizationModifier(
        targets=sensitive_layers,
        scheme="FP8",
        ignore=ALWAYS_SKIP,
    )

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset="HuggingFaceH4/ultrachat_200k",
        recipe=fp8_recipe,
        max_seq_length=seqlen,
        num_calibration_samples=nsamples,
        output_dir=output_dir,
    )

    # Cleanup intermediate
    shutil.rmtree(output_dir + "-step1", ignore_errors=True)

    print(f"[fp4-fp8-sensitive] Saved to {output_dir}")


def quantize_fp4_with_bf16_critical(
    model_name: str,
    output_dir: str,
    nsamples: int = 256,
    seqlen: int = 4096,
):
    """
    Variant 3: FP4 everywhere, but keep the most critical GDN layers in bf16.
    More conservative than fp4-fp8-sensitive.
    """
    print(f"[fp4-bf16-critical] Quantizing {model_name}")

    # Keep all GDN linear_attn layers in bf16 (matches Sehyo's approach)
    critical_layers = GDN_SENSITIVE

    ignore_list = ALWAYS_SKIP + critical_layers

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=ignore_list,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset="HuggingFaceH4/ultrachat_200k",
        recipe=recipe,
        max_seq_length=seqlen,
        num_calibration_samples=nsamples,
        output_dir=output_dir,
    )

    print(f"[fp4-bf16-critical] Saved to {output_dir}")


def quantize_with_autoscheme(
    model_name: str,
    output_dir: str,
    target_bits: float = 4.5,
    nsamples: int = 256,
    seqlen: int = 4096,
):
    """
    Variant 4: Let AutoRound's AutoScheme optimizer choose bit allocation.
    Uses gradient-based sensitivity to assign bits per layer.
    """
    print(f"[autoscheme] Quantizing {model_name} with target avg bits = {target_bits}")

    from auto_round import AutoRound, AutoScheme

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # AutoScheme with NVFP4-compatible options
    scheme = AutoScheme(
        avg_bits=target_bits,
        options=("W4A16", "W8A16"),  # FP4 and FP8 equivalents
    )

    # Fix structural exclusions
    layer_config = {}
    for pattern in ALWAYS_SKIP:
        layer_config[pattern] = {"bits": 16}  # Keep in bf16

    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        scheme=scheme,
        batch_size=4,
        seqlen=seqlen,
        nsamples=nsamples,
        iters=200,
        dataset="NeelNanda/pile-10k,HuggingFaceH4/ultrachat_200k:split=train_sft:num=64",
        layer_config=layer_config,
    )

    autoround.quantize()
    autoround.save_quantized(output_dir, format="llm_compressor")

    print(f"[autoscheme] Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="NVFP4 quantization for Qwen3.5")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B",
                        help="Model name or path")
    parser.add_argument("--variant", type=str, required=True,
                        choices=["all-fp4", "fp4-fp8-sensitive", "fp4-bf16-critical", "autoscheme"],
                        help="Quantization variant")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--sensitivity-dir", type=str, default=None,
                        help="Directory with sensitivity_scores.json (for fp4-fp8-sensitive)")
    parser.add_argument("--nsamples", type=int, default=256,
                        help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=4096,
                        help="Sequence length for calibration")
    parser.add_argument("--target-bits", type=float, default=4.5,
                        help="Target average bits for autoscheme variant")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.variant == "all-fp4":
        quantize_all_fp4(args.model, args.output, args.nsamples, args.seqlen)
    elif args.variant == "fp4-fp8-sensitive":
        quantize_fp4_with_fp8_sensitive(
            args.model, args.output, args.sensitivity_dir,
            nsamples=args.nsamples, seqlen=args.seqlen
        )
    elif args.variant == "fp4-bf16-critical":
        quantize_fp4_with_bf16_critical(args.model, args.output, args.nsamples, args.seqlen)
    elif args.variant == "autoscheme":
        quantize_with_autoscheme(
            args.model, args.output, args.target_bits, args.nsamples, args.seqlen
        )


if __name__ == "__main__":
    main()
