#!/usr/bin/env python3
"""Compress a model with mixed NVFP4 + MXFP8 quantization.

Reads an AutoRound AutoScheme recipe (quantization_config.json) to determine
which layers should be NVFP4 (default) vs MXFP8 (higher precision), then uses
llm_compressor to produce a compressed-tensors checkpoint that vLLM can serve
with CUDA-accelerated kernels for both formats.

Usage:
    python compress_mixed_nvfp4_mxfp8.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --recipe /tmp/autoround_qwen35_35b_nvfp4_mxfp8_fixed/autoround_textonly_777850a6-w4g16/quantization_config.json \
        --output ./Qwen3.5-35B-A3B-NVFP4-MXFP8 \
        --mxfp8-flavor W8A8 \
        --num-calibration-samples 512

Supports both MXFP8 flavors:
    W8A8   - MXFP8 weights + dynamic MXFP8 activations (better throughput on Blackwell)
    W8A16  - MXFP8 weights + BF16 activations (simpler, no activation quant overhead)
"""

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import torch
from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization.quant_scheme import MXFP8, MXFP8A16, NVFP4
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def parse_autoround_recipe(recipe_path: str) -> list[str]:
    """Extract MXFP8 layer names from AutoRound AutoScheme output.

    Returns list of full module names that should use MXFP8 instead of NVFP4.
    """
    with open(recipe_path) as f:
        cfg = json.load(f)

    extra = cfg.get("extra_config", {})
    mxfp8_layers = []
    for name, layer_cfg in extra.items():
        if layer_cfg.get("data_type") == "mx_fp" and layer_cfg.get("bits", 4) == 8:
            mxfp8_layers.append(name)

    return sorted(expand_fused_layer_groups(mxfp8_layers))


def expand_fused_layer_groups(layer_names: list[str]) -> set[str]:
    """Expand mixed-format layer picks to whole fused groups required by vLLM.

    vLLM packs some projections into fused modules such as qkv_proj and
    gate_up_proj. All component projections inside a fused module must use the
    same quantization scheme, otherwise the loader sees incompatible parameter
    layouts inside a single fused parameter set.

    To keep vLLM changes minimal, we normalize AutoScheme picks upward here:
    - any q/k/v projection promotion expands to the full q_proj/k_proj/v_proj set
    - any gate/up promotion expands to the full gate_proj/up_proj set
    """
    expanded = set(layer_names)
    for name in list(layer_names):
        if ".self_attn." in name:
            prefix, suffix = name.rsplit(".self_attn.", 1)
            if suffix in {"q_proj", "k_proj", "v_proj"}:
                expanded.update(
                    f"{prefix}.self_attn.{proj}"
                    for proj in ("q_proj", "k_proj", "v_proj")
                )
        elif ".mlp." in name:
            prefix, suffix = name.rsplit(".mlp.", 1)
            if suffix in {"gate_proj", "up_proj"}:
                expanded.update(
                    f"{prefix}.mlp.{proj}" for proj in ("gate_proj", "up_proj")
                )
    return expanded


def build_regex_targets(layer_names: list[str]) -> list[str]:
    """Convert exact layer names to regex targets for llm_compressor.

    For efficiency, groups layers by common prefix where possible.
    Falls back to exact match when grouping isn't clean.
    """
    targets = []
    for name in layer_names:
        # Escape dots for regex, use exact match anchors
        targets.append(f"re:^{name.replace('.', '[.]')}$")
    return targets


def build_nvfp4_targets(
    model: torch.nn.Module, mxfp8_layers: list[str], ignore: set[str]
) -> list[str]:
    """Build explicit NVFP4 targets for all remaining linear modules.

    Avoiding a broad ``Linear`` catch-all is important for vLLM compatibility:
    vLLM checks module-class matches before fused-layer expansion, so a catch-all
    can accidentally assign the default scheme to fused modules like
    ``qkv_proj``/``gate_up_proj`` before the per-shard q/k/v or gate/up targets
    are consulted.

    Restricting this to actual ``torch.nn.Linear`` modules also avoids
    quantizing non-linear 2D-weight modules such as token embeddings and MoE
    router gates, which llm-compressor does not handle through the NVFP4 linear
    path.
    """
    explicit = []
    promoted = set(mxfp8_layers)
    for name, module in model.named_modules():
        if not name or name in ignore or name in promoted:
            continue
        if isinstance(module, torch.nn.Linear):
            explicit.append(name)
    return build_regex_targets(sorted(explicit))


def main():
    parser = argparse.ArgumentParser(
        description="Mixed NVFP4+MXFP8 compression via llm_compressor"
    )
    parser.add_argument(
        "--model", required=True, help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help="Path to AutoRound quantization_config.json with AutoScheme output",
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for compressed checkpoint"
    )
    parser.add_argument(
        "--mxfp8-flavor",
        choices=["W8A8", "W8A16"],
        default="W8A8",
        help="MXFP8 flavor: W8A8 (weight+activation) or W8A16 (weight-only)",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=512,
        help="Number of calibration samples for NVFP4 activation scales",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for calibration",
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceH4/ultrachat_200k",
        help="Calibration dataset ID",
    )
    parser.add_argument(
        "--dataset-split",
        default="train_sft",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=["lm_head"],
        help="Layers to ignore (not quantize)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print recipe and exit without compressing",
    )
    args = parser.parse_args()

    # Parse AutoRound recipe
    print(f"Parsing AutoRound recipe: {args.recipe}")
    mxfp8_layers = parse_autoround_recipe(args.recipe)
    print(f"  MXFP8 layers: {len(mxfp8_layers)}")
    if not mxfp8_layers:
        print("  WARNING: No MXFP8 layers found. All layers will be NVFP4.")

    # Build config_groups
    mxfp8_scheme = deepcopy(MXFP8 if args.mxfp8_flavor == "W8A8" else MXFP8A16)
    mxfp8_scheme["targets"] = build_regex_targets(mxfp8_layers)

    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    nvfp4_scheme = deepcopy(NVFP4)
    nvfp4_scheme["targets"] = build_nvfp4_targets(
        model, mxfp8_layers, set(args.ignore)
    )

    # MXFP8 group first so it matches before the remaining explicit NVFP4 set.
    config_groups = {}
    if mxfp8_layers:
        config_groups["group_mxfp8"] = mxfp8_scheme
    config_groups["group_nvfp4"] = nvfp4_scheme

    recipe = QuantizationModifier(config_groups=config_groups, ignore=args.ignore)

    print(f"\nRecipe:")
    print(f"  MXFP8 ({args.mxfp8_flavor}): {len(mxfp8_layers)} layers")
    print(f"  NVFP4: {len(nvfp4_scheme['targets'])} explicit layers")
    print(f"  Ignored: {args.ignore}")

    if mxfp8_layers:
        print(f"\n  Sample MXFP8 layers (first 10):")
        for name in mxfp8_layers[:10]:
            print(f"    {name}")
        if len(mxfp8_layers) > 10:
            print(f"    ... and {len(mxfp8_layers) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] Recipe built successfully. Exiting.")
        recipe_dict = {
            "config_groups": {},
            "ignore": args.ignore,
        }
        for gname, gscheme in config_groups.items():
            recipe_dict["config_groups"][gname] = {
                k: str(v) if hasattr(v, "model_dump") else v
                for k, v in gscheme.items()
            }
        print(json.dumps(recipe_dict, indent=2))
        return

    # Load calibration dataset. Support both Hub datasets and local JSON/JSONL/TXT
    # files so we can iterate quickly with tiny calibration corpora.
    print(f"Loading calibration dataset: {args.dataset}")
    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        suffix = dataset_path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            ds = load_dataset("json", data_files=str(dataset_path), split="train")
        elif suffix in {".txt", ".text"}:
            ds = load_dataset("text", data_files=str(dataset_path), split="train")
        else:
            raise ValueError(
                f"Unsupported local dataset format: {dataset_path}. "
                "Use .json, .jsonl, .txt, or a Hugging Face dataset ID."
            )
        if args.num_calibration_samples:
            ds = ds.select(range(min(args.num_calibration_samples, len(ds))))
    else:
        ds = load_dataset(
            args.dataset,
            split=f"{args.dataset_split}[:{args.num_calibration_samples}]",
        )
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        if "messages" in example:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        elif "text" in example:
            text = example["text"]
        else:
            raise ValueError(
                "Calibration dataset rows must contain either 'messages' or 'text'."
            )
        return {"text": text}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # Run compression
    print(f"\nRunning oneshot compression...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
        processor=tokenizer,
    )

    # Verify
    print("\n========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(input_ids, max_new_tokens=20)
    print(tokenizer.decode(output[0]))
    print("==========================================")

    # Save
    print(f"\nSaving to: {args.output}")
    model.save_pretrained(args.output, save_compressed=True)
    tokenizer.save_pretrained(args.output)

    # Preserve source tokenizer metadata verbatim.
    # Some Qwen-family tokenizers emit a simplified tokenizer_config.json when
    # re-saved, which drops fields needed by current Transformers/vLLM
    # tokenization. Copying the original tokenizer-side files keeps the mixed
    # compressed checkpoint loadable without changing any quantized weights.
    source_dir = Path(args.model)
    output_dir = Path(args.output)
    for name in (
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
    ):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    # Verify output config
    config_path = Path(args.output) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        qconfig = config.get("quantization_config", {})
        fmt = qconfig.get("format", "unknown")
        groups = qconfig.get("config_groups", {})
        print(f"\nOutput checkpoint:")
        print(f"  Format: {fmt}")
        print(f"  Config groups: {list(groups.keys())}")
        for gname, gcfg in groups.items():
            gfmt = gcfg.get("format", "inferred")
            targets = gcfg.get("targets", [])
            print(f"    {gname}: format={gfmt}, targets={len(targets)} layers")

    print("\nDone! Serve with:")
    print(f"  vllm serve {args.output} --quantization compressed-tensors")


if __name__ == "__main__":
    main()
