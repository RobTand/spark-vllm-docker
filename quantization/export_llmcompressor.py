#!/usr/bin/env python3
"""
export_llmcompressor.py — export a DynaQuant promotion recipe through
llm-compressor so vLLM can load it natively via compressed-tensors.

This exporter intentionally targets only native vLLM/Blackwell buckets:
  - NVFP4 (W4A4, g=16)
  - FP8
  - BF16 passthrough

The optimizer may emit richer configs such as ``w5_s16_g32``. Those are snapped
to native buckets with a quality-preserving policy by default:
  - sub-FP4 → FP4
  - non-native FP4 or 5-8 bit → FP8
  - 9+ bit → BF16

The output is therefore deployment-oriented rather than a byte-exact replay of
the optimizer recipe.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent))
from build_rtn_cache import stage_multimodal


CONFIG_RE = re.compile(r"^w(?P<w>\d+)_s(?P<s>\d+)_g(?P<g>\d+)$")


def parse_config_string(config: str) -> tuple[int, int, int]:
    match = CONFIG_RE.match(config)
    if not match:
        raise ValueError(f"unrecognized recipe config: {config}")
    return (
        int(match.group("w")),
        int(match.group("s")),
        int(match.group("g")),
    )


def load_recipe_entry(pareto_path: str, curve_name: str, step: str) -> tuple[dict, dict]:
    with open(pareto_path) as f:
        data = json.load(f)

    curve_key = "promotion_curve" if curve_name == "promotion" else "pareto_curve"
    if curve_key not in data:
        raise KeyError(f"{pareto_path} does not contain {curve_key}")
    curve = data[curve_key]

    if step == "knee":
        knee_key = "promotion_knee" if curve_name == "promotion" else "knee"
        if knee_key not in data:
            raise KeyError(f"{pareto_path} does not contain {knee_key}")
        return data[knee_key], data

    if step == "final":
        return curve[-1], data

    target_step = int(step)
    return min(curve, key=lambda p: abs(p["step"] - target_step)), data


def snap_native_bucket(w_bits: int, s_bits: int, group_size: int,
                       snap_mode: str) -> int:
    if snap_mode == "nearest":
        if w_bits <= 4:
            return 4
        if w_bits <= 8:
            return 8
        return 16

    # Default: conservative/quality-preserving.
    if w_bits < 4:
        return 4
    if w_bits == 4:
        return 4 if (s_bits == 8 and group_size == 16) else 8
    if w_bits <= 8:
        return 8
    return 16


def snap_fused_groups(native_recipe: dict[str, int]) -> dict[str, int]:
    """Make only genuinely fused groups uniform.

    vLLM/llm-compressor care about fused QKV and fused gate/up pairs. Snapping
    whole transformer blocks is unnecessarily destructive.
    """
    snapped = dict(native_recipe)
    groups = defaultdict(list)

    for name in native_recipe:
        if not name.endswith(".weight"):
            continue

        if any(tok in name for tok in (".linear_attn.in_proj_qkv.weight",
                                       ".mlp.experts.gate_up_proj.weight",
                                       ".mlp.gate_up_proj.weight")):
            continue

        if re.search(r"\.self_attn\.(q_proj|k_proj|v_proj)\.weight$", name):
            base = re.sub(r"\.(q_proj|k_proj|v_proj)\.weight$", "", name)
            groups[(base, "qkv")].append(name)
        elif re.search(r"\.(mlp|shared_expert)\.(gate_proj|up_proj)\.weight$", name):
            base = re.sub(r"\.(gate_proj|up_proj)\.weight$", "", name)
            groups[(base, "gate_up")].append(name)

    for (_, _), names in groups.items():
        fused_bits = max(snapped[n] for n in names)
        for name in names:
            snapped[name] = fused_bits

    return snapped


def estimate_native_bpw(recipe: dict[str, str], native_recipe: dict[str, int]) -> float:
    total_numel = 0
    total_bits = 0.0
    for name, config in recipe.items():
        w_bits, _s_bits, _group_size = parse_config_string(config)
        native_bits = native_recipe[name]
        total_numel += 1
        total_bits += native_bits
        if w_bits == 4 and native_bits == 8:
            pass
    return total_bits / total_numel if total_numel else 0.0


def build_recipe(recipe: dict, snap_mode: str) -> tuple[list, dict]:
    """Convert optimizer recipe to llm-compressor QuantizationModifier(s)."""
    from llmcompressor.modifiers.quantization import QuantizationModifier

    raw_hist = Counter(recipe.values())
    print(f"[export] raw config histogram: {dict(sorted(raw_hist.items()))}", flush=True)

    native_recipe = {
        name: snap_native_bucket(*parse_config_string(config), snap_mode=snap_mode)
        for name, config in recipe.items()
    }
    native_recipe = snap_fused_groups(native_recipe)

    by_bits = defaultdict(list)
    for name, bits in native_recipe.items():
        by_bits[bits].append(name.replace(".weight", ""))

    native_hist = Counter(native_recipe.values())
    print(f"[export] native bucket histogram: {dict(sorted(native_hist.items()))}", flush=True)

    config_groups = {}
    ignore = []

    if 4 in by_bits:
        config_groups["dynaquant_fp4"] = {
            "weights": {
                "num_bits": 4,
                "type": "float",
                "strategy": "tensor_group",
                "group_size": 16,
                "symmetric": True,
                "scale_dtype": "torch.float8_e4m3fn",
                "observer": "memoryless_minmax",
            },
            "input_activations": {
                "num_bits": 4,
                "type": "float",
                "strategy": "tensor_group",
                "group_size": 16,
                "symmetric": True,
                "dynamic": "local",
                "observer": "static_minmax",
                "scale_dtype": "torch.float8_e4m3fn",
            },
            "targets": sorted(by_bits[4]),
        }

    if 8 in by_bits:
        config_groups["dynaquant_fp8"] = {
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "channel",
                "symmetric": True,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "strategy": "token",
                "symmetric": True,
                "dynamic": True,
            },
            "targets": sorted(by_bits[8]),
        }

    if 16 in by_bits:
        ignore.extend(sorted(by_bits[16]))

    manifest = {
        "native_bucket_histogram": dict(sorted(native_hist.items())),
        "snap_mode": snap_mode,
        "n_targets_fp4": len(by_bits[4]),
        "n_targets_fp8": len(by_bits[8]),
        "n_targets_bf16": len(by_bits[16]),
    }

    modifier = QuantizationModifier(config_groups=config_groups, ignore=ignore)
    return [modifier], manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pareto", required=True)
    parser.add_argument("--curve", choices=("promotion", "pareto"), default="promotion")
    parser.add_argument("--step", default="knee")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-calibration-samples", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--snap-mode", choices=("conservative", "nearest"),
                        default="conservative")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entry, pareto_data = load_recipe_entry(args.pareto, args.curve, args.step)
    raw_recipe = entry["recipe"]

    print(f"[export] selected {args.curve} step {entry['step']}", flush=True)
    print(f"[export] predicted memory: {entry['cost_bytes']/1e9:.3f} GB", flush=True)
    print(f"[export] predicted weighted error: {entry['weighted_error']:.4e}", flush=True)

    recipe, manifest = build_recipe(raw_recipe, snap_mode=args.snap_mode)
    manifest.update(
        {
            "source_model": args.model,
            "pareto_source": args.pareto,
            "curve": args.curve,
            "step": entry["step"],
            "predicted_cost_bytes": entry["cost_bytes"],
            "predicted_weighted_error": entry["weighted_error"],
        }
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "native_export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[export] wrote manifest to {output_dir / 'native_export_manifest.json'}", flush=True)

    if args.dry_run:
        print("[export] dry-run requested; skipping llm-compressor quantization", flush=True)
        return

    staged, cleanup = stage_multimodal(args.model)
    model_path = staged

    try:
        from llmcompressor import oneshot

        print(f"[export] running oneshot quantization via llm-compressor", flush=True)
        print(f"[export] forcing sequential pipeline for activation calibration", flush=True)
        oneshot(
            model=model_path,
            recipe=recipe,
            dataset="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            num_calibration_samples=args.num_calibration_samples,
            max_seq_length=args.max_seq_length,
            trust_remote_code_model=True,
            save_compressed=True,
            output_dir=args.output,
            pipeline="sequential",  # force calibration-based pipeline
        )
        print(f"[export] done! Model saved to {args.output}", flush=True)
        print(f"[export] serve with: vllm serve {args.output} --trust-remote-code")

    finally:
        if cleanup:
            import shutil
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
