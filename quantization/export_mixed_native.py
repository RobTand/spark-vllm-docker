#!/usr/bin/env python3
"""Export a DynaQuant native mixed-format recipe to compressed-tensors.

This is the deployment-oriented bridge from DynaQuant -> llm-compressor -> vLLM.

It accepts a per-module native assignment such as:
  - NVFP4
  - MXFP8
  - MXFP4
  - FP8
  - BF16

and emits a standard compressed-tensors checkpoint with explicit config groups,
matching the mixed artifact shape that vLLM already loads successfully.

The intent is to keep vLLM changes minimal:
  - use only native compressed-tensors schemes
  - emit explicit regex targets per group
  - normalize fused q/k/v and gate/up groups so one serving unit has one format
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization.quant_scheme import (
    FP8,
    FP8_DYNAMIC,
    MXFP4,
    MXFP4A16,
    MXFP8,
    MXFP8A16,
    NVFP4,
    NVFP4A16,
)
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

sys.path.insert(0, str(Path(__file__).parent))
from build_rtn_cache import stage_multimodal


CONFIG_RE = __import__("re").compile(r"^w(?P<w>\d+)_s(?P<s>\d+)_g(?P<g>\d+)$")

FORMAT_ALIASES = {
    "nvfp4": "NVFP4",
    "nvfp4a16": "NVFP4",
    "mxfp4": "MXFP4",
    "mxfp4_e2m1": "MXFP4",
    "mxfp4a16": "MXFP4",
    "mxfp8": "MXFP8",
    "mxfp8_e4m3": "MXFP8",
    "mxfp8_e5m2": "MXFP8",
    "mxfp8a16": "MXFP8",
    "fp8": "FP8",
    "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8",
    "bf16": "BF16",
    "bfloat16": "BF16",
    "16": "BF16",
    "8": "FP8",
    "4": "NVFP4",
}

# Lower is cheaper / lower precision.
FORMAT_RANK = {
    "NVFP4": 0,
    "MXFP4": 1,
    "MXFP8": 2,
    "FP8": 3,
    "BF16": 4,
}


def parse_config_string(config: str) -> tuple[int, int, int]:
    match = CONFIG_RE.match(config)
    if not match:
        raise ValueError(f"unrecognized config string: {config}")
    return (
        int(match.group("w")),
        int(match.group("s")),
        int(match.group("g")),
    )


def snap_bits_to_native(bits: int, *, allow_mxfp4: bool = True) -> str:
    if bits <= 4:
        return "NVFP4" if not allow_mxfp4 else "NVFP4"
    if bits <= 8:
        return "MXFP8"
    return "BF16"


def canonicalize_format(value, *, snap_legacy: bool, allow_mxfp4: bool) -> str:
    if isinstance(value, int):
        return snap_bits_to_native(value, allow_mxfp4=allow_mxfp4)
    if isinstance(value, float) and value.is_integer():
        return snap_bits_to_native(int(value), allow_mxfp4=allow_mxfp4)
    if not isinstance(value, str):
        raise ValueError(f"unsupported recipe value type: {type(value).__name__}")

    text = value.strip()
    lowered = text.lower()
    if lowered in FORMAT_ALIASES:
        fmt = FORMAT_ALIASES[lowered]
        if fmt == "MXFP4" and not allow_mxfp4:
            return "NVFP4"
        return fmt

    if CONFIG_RE.match(text):
        if not snap_legacy:
            raise ValueError(
                f"legacy config string {text!r} requires --snap-legacy-configs"
            )
        w_bits, _s_bits, _group_size = parse_config_string(text)
        return snap_bits_to_native(w_bits, allow_mxfp4=allow_mxfp4)

    raise ValueError(f"unrecognized recipe value: {value!r}")


def load_entry(recipe_path: str, curve: str, step: str) -> dict:
    with open(recipe_path) as f:
        data = json.load(f)

    if "refined_assignment" in data:
        return data["refined_assignment"]

    if "recipe" in data and isinstance(data["recipe"], dict):
        return data["recipe"]

    if all(isinstance(v, str) for v in data.values()):
        return data

    curve_key = "promotion_curve" if curve == "promotion" else "pareto_curve"
    knee_key = "promotion_knee" if curve == "promotion" else "knee"
    legacy_curve_key = "pareto"

    if step == "knee":
        if knee_key in data and isinstance(data[knee_key], dict):
            return data[knee_key]["recipe"]
        if "knee" in data and isinstance(data["knee"], dict) and "recipe" in data["knee"]:
            return data["knee"]["recipe"]
    if curve_key in data:
        entries = data[curve_key]
    elif legacy_curve_key in data:
        entries = data[legacy_curve_key]
    else:
        raise KeyError(f"could not find recipe entry in {recipe_path}")

    if step == "final":
        return entries[-1]["recipe"]

    target = int(step)
    best = min(entries, key=lambda p: abs(int(p["step"]) - target))
    return best["recipe"]


def strip_weight_suffix(name: str) -> str:
    return name[:-7] if name.endswith(".weight") else name


def explicit_regex(name: str) -> str:
    return f"re:^{name.replace('.', '[.]')}$"


def promote_serving_units(assignment: dict[str, str]) -> dict[str, str]:
    """Normalize known fused serving units to a single highest-rank format."""
    import re

    out = dict(assignment)
    groups: dict[tuple[str, str], list[str]] = {}
    for name in list(out):
        if ".self_attn." in name:
            m = re.search(r"^(?P<pre>.+\.self_attn)\.(?P<sib>q_proj|k_proj|v_proj)$", name)
            if m:
                key = (m.group("pre"), "qkv")
                groups.setdefault(key, []).append(name)
        if ".mlp." in name or ".shared_expert." in name:
            m = re.search(
                r"^(?P<pre>.+\.(?:mlp|shared_expert))\.(?P<sib>gate_proj|up_proj)$",
                name,
            )
            if m:
                key = (m.group("pre"), "gate_up")
                groups.setdefault(key, []).append(name)

    for (_prefix, _kind), names in groups.items():
        best = max((out[n] for n in names), key=lambda fmt: FORMAT_RANK[fmt])
        for name in names:
            out[name] = best
    return out


def resolve_assignment(
    raw_recipe: dict,
    *,
    snap_legacy: bool,
    allow_mxfp4: bool,
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for raw_name, raw_value in raw_recipe.items():
        name = strip_weight_suffix(raw_name)
        fmt = canonicalize_format(
            raw_value,
            snap_legacy=snap_legacy,
            allow_mxfp4=allow_mxfp4,
        )
        assignment[name] = fmt
    return promote_serving_units(assignment)


def scheme_for_format(fmt: str, *, mxfp8_flavor: str, fp8_flavor: str):
    if fmt == "NVFP4":
        return deepcopy(NVFP4)
    if fmt == "MXFP4":
        return deepcopy(MXFP4)
    if fmt == "MXFP8":
        return deepcopy(MXFP8 if mxfp8_flavor == "W8A8" else MXFP8A16)
    if fmt == "FP8":
        return deepcopy(FP8_DYNAMIC if fp8_flavor == "dynamic" else FP8)
    raise ValueError(f"no export scheme for format {fmt}")


def build_config_groups(
    assignment: dict[str, str],
    *,
    ignore: Iterable[str],
    mxfp8_flavor: str,
    fp8_flavor: str,
) -> tuple[dict[str, dict], list[str]]:
    ignored = set(ignore)
    by_format: dict[str, list[str]] = {}
    bf16_ignore = []
    for name, fmt in sorted(assignment.items()):
        if name in ignored:
            continue
        if fmt == "BF16":
            bf16_ignore.append(name)
            continue
        by_format.setdefault(fmt, []).append(name)

    config_groups: dict[str, dict] = {}
    ordered_formats = sorted(by_format, key=lambda fmt: FORMAT_RANK[fmt], reverse=True)
    for idx, fmt in enumerate(ordered_formats):
        scheme = scheme_for_format(fmt, mxfp8_flavor=mxfp8_flavor, fp8_flavor=fp8_flavor)
        scheme["targets"] = [explicit_regex(name) for name in sorted(by_format[fmt])]
        config_groups[f"group_{idx}"] = scheme
    return config_groups, sorted(set(bf16_ignore) | set(ignored))


def preprocess_dataset(ds, tokenizer, max_seq_length: int):
    def preprocess(example):
        if "messages" in example:
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
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
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    return ds.map(tokenize, remove_columns=ds.column_names)


def load_calibration_dataset(dataset_arg: str, split: str, num_samples: int):
    dataset_path = Path(dataset_arg)
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
        if num_samples:
            ds = ds.select(range(min(num_samples, len(ds))))
    else:
        ds = load_dataset(dataset_arg, split=f"{split}[:{num_samples}]")
    return ds.shuffle(seed=42)


def preserve_tokenizer_files(source_model: str, output_dir: Path):
    source_dir = Path(source_model)
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


def main():
    parser = argparse.ArgumentParser(
        description="Export a DynaQuant mixed native recipe to compressed-tensors."
    )
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--recipe", required=True, help="Recipe JSON path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--curve", choices=("promotion", "pareto"), default="promotion")
    parser.add_argument("--step", default="knee")
    parser.add_argument(
        "--snap-legacy-configs",
        action="store_true",
        help="Allow legacy w4_s8_g16-style entries by snapping them to native formats",
    )
    parser.add_argument(
        "--allow-mxfp4",
        action="store_true",
        help="Allow MXFP4 in the export path; otherwise 4-bit values stay NVFP4",
    )
    parser.add_argument(
        "--mxfp8-flavor",
        choices=("W8A8", "W8A16"),
        default="W8A8",
        help="MXFP8 export flavor",
    )
    parser.add_argument(
        "--fp8-flavor",
        choices=("dynamic", "tensor"),
        default="dynamic",
        help="FP8 export flavor",
    )
    parser.add_argument("--num-calibration-samples", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument(
        "--dataset",
        default="HuggingFaceH4/ultrachat_200k",
        help="Calibration dataset or local json/jsonl/txt path",
    )
    parser.add_argument("--dataset-split", default="train_sft")
    parser.add_argument("--ignore", nargs="*", default=["lm_head"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw_recipe = load_entry(args.recipe, args.curve, args.step)
    assignment = resolve_assignment(
        raw_recipe,
        snap_legacy=args.snap_legacy_configs,
        allow_mxfp4=args.allow_mxfp4,
    )
    config_groups, ignore = build_config_groups(
        assignment,
        ignore=args.ignore,
        mxfp8_flavor=args.mxfp8_flavor,
        fp8_flavor=args.fp8_flavor,
    )

    manifest = {
        "source_model": args.model,
        "source_recipe": args.recipe,
        "curve": args.curve,
        "step": args.step,
        "formats": dict(sorted(Counter(assignment.values()).items())),
        "n_targets": {group: len(cfg.get("targets", [])) for group, cfg in config_groups.items()},
        "ignore": ignore,
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "mixed_native_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("[export] format histogram:", manifest["formats"], flush=True)
    print("[export] groups:", manifest["n_targets"], flush=True)
    print("[export] ignore:", len(ignore), flush=True)

    if args.dry_run:
        print(json.dumps({"config_groups": config_groups, "ignore": ignore}, indent=2, default=str))
        return

    staged, cleanup = stage_multimodal(args.model)
    try:
        local_model = Path(staged).exists()
        model = AutoModelForCausalLM.from_pretrained(
            staged,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_model,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            staged,
            trust_remote_code=True,
            local_files_only=local_model,
        )
        ds = load_calibration_dataset(
            args.dataset,
            args.dataset_split,
            args.num_calibration_samples,
        )
        ds = preprocess_dataset(ds, tokenizer, args.max_seq_length)

        recipe = QuantizationModifier(config_groups=config_groups, ignore=ignore)
        print("[export] running oneshot compression...", flush=True)
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=args.max_seq_length,
            num_calibration_samples=args.num_calibration_samples,
            processor=tokenizer,
        )

        print("[export] saving compressed checkpoint...", flush=True)
        model.save_pretrained(args.output, save_compressed=True)
        tokenizer.save_pretrained(args.output)
        preserve_tokenizer_files(args.model, output_dir)

        config_path = output_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            qconfig = config.get("quantization_config", {})
            print("[export] output format:", qconfig.get("format"), flush=True)
            print("[export] output groups:", list(qconfig.get("config_groups", {}).keys()), flush=True)

        print("[export] sample generation...", flush=True)
        dispatch_model(model)
        input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
        output = model.generate(input_ids, max_new_tokens=16)
        print(tokenizer.decode(output[0]), flush=True)
        print(f"[export] done. Serve with: vllm serve {args.output} --quantization compressed-tensors", flush=True)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
