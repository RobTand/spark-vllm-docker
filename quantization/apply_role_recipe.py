#!/usr/bin/env python3
"""
Apply a per-role recipe to an AutoRound cache to produce a mixed-precision
deployable model — bypassing DPQ gradient descent entirely.

Inputs:
  --model       : source BF16 model path
  --cache-dir   : dir containing fp4_weights.safetensors, fp8_weights.safetensors,
                  cache_manifest.json
  --recipe      : role → target FP4 retention fraction (JSON file)
  --output      : where to save the final mixed-precision model
  --ar-log      : optional AutoRound log file to rank per-block losses (used
                  to decide WHICH linears within each role to upgrade).
                  If not provided, uses layer index as a tiebreaker.

Output: saved model where each Linear has BF16 weights from either the
FP4 or FP8 AutoRound pass, chosen by the role recipe.

Example recipe.json (4B knee at eff=0.50):
  {
    "attn.k_proj": 0.88,
    "attn.q_proj": 0.50,
    "attn.v_proj": 0.50,
    "attn.o_proj": 0.12,
    "mlp.gate_proj": 0.41,
    "mlp.up_proj": 0.38,
    "mlp.down_proj": 0.28,
    "other.in_proj_a": 1.00,
    "other.in_proj_b": 1.00,
    "other.in_proj_qkv": 0.62,
    "other.in_proj_z": 0.79,
    "other.out_proj": 0.54
  }
"""
import argparse
import gc
import json
import re
import tempfile
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from safetensors.torch import load_file


def load_weights_possibly_sharded(cache_dir: Path, prefix: str) -> Dict[str, torch.Tensor]:
    """Load {prefix}_weights from cache_dir. Supports both single-file
    (legacy: prefix_weights.safetensors) and sharded (new: prefix_weights-NN-of-MM.safetensors)."""
    # Try sharded first
    shards = sorted(cache_dir.glob(f"{prefix}_weights-*-of-*.safetensors"))
    if shards:
        merged: Dict[str, torch.Tensor] = {}
        for s in shards:
            merged.update(load_file(str(s)))
        return merged
    # Fall back to single file (legacy 1.5B / 4B caches)
    single = cache_dir / f"{prefix}_weights.safetensors"
    if single.exists():
        return load_file(str(single))
    raise FileNotFoundError(f"no {prefix} weights found in {cache_dir}")


def stage_multimodal(model_path: str):
    src_cfg_path = Path(model_path) / "config.json"
    if not src_cfg_path.exists():
        return model_path, None
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    if "vision_config" not in cfg and "text_config" not in cfg:
        return model_path, None
    for k in ["vision_config", "image_token_id", "video_token_id",
              "vision_start_token_id", "vision_end_token_id"]:
        cfg.pop(k, None)
    if "text_config" in cfg:
        text_cfg = cfg.pop("text_config")
        for k, v in text_cfg.items():
            if k not in cfg:
                cfg[k] = v
        if "model_type" in text_cfg:
            cfg["model_type"] = text_cfg["model_type"]
    archs = cfg.get("architectures", [])
    if archs:
        cfg["architectures"] = [
            a.replace("ForConditionalGeneration", "ForCausalLM") for a in archs
        ]
    staged = tempfile.mkdtemp(prefix="apply_recipe_staged_")
    for p in Path(model_path).iterdir():
        if p.name == "config.json":
            continue
        (Path(staged) / p.name).symlink_to(p.resolve())
    with open(Path(staged) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return staged, staged


def role(name: str) -> str:
    """Map a weight name to a role category. Order matters — more specific
    patterns first. Strips a trailing ".weight" before matching the last
    segment."""
    # MoE fused experts (3D tensors) — no .weight suffix
    if ".mlp.experts.gate_up_proj" in name:
        return "moe.gate_up_proj"
    if ".mlp.experts.down_proj" in name:
        return "moe.down_proj"
    # Shared expert (dense nn.Linear within MoE layers)
    if "shared_expert.gate_proj" in name:
        return "shared.gate_proj"
    if "shared_expert.up_proj" in name:
        return "shared.up_proj"
    if "shared_expert.down_proj" in name:
        return "shared.down_proj"
    # Standard attention
    if ".self_attn." in name:
        last = name.rstrip(".weight").split(".")[-1]
        return "attn." + last
    # Dense MLP (non-MoE models)
    if ".mlp." in name and ".shared_expert" not in name and ".experts" not in name:
        last = name.rstrip(".weight").split(".")[-1]
        return "mlp." + last
    # Linear-attention (DeltaNet) projections
    if ".linear_attn." in name:
        last = name.rstrip(".weight").split(".")[-1]
        return "other." + last
    # Fallback
    last = name.rstrip(".weight").split(".")[-1]
    return "other." + last


def parse_ar_losses(logfile: str):
    """Parse AutoRound log for per-block FP4 reconstruction losses.
    Returns {block_idx: fp4_final_loss}."""
    LOSS_RE = re.compile(r"loss iter (\d+): ([\d.]+) -> iter (\d+): ([\d.]+)")
    with open(logfile) as f:
        text = f.read()
    fp4_start = text.find("Stage 3a")
    fp8_start = text.find("Stage 3b")
    fp4_section = text[fp4_start:fp8_start] if fp8_start > 0 else text[fp4_start:]
    losses = []
    for m in LOSS_RE.finditer(fp4_section):
        losses.append(float(m.group(4)))
    return {i: l for i, l in enumerate(losses)}


def build_decisions(fp4_weights, recipe, block_loss):
    """For each Linear in the cache, decide fp4 or fp8 based on the recipe.

    Strategy: for each role, compute how many Linears to keep as fp4
    (recipe fraction * total). Rank the Linears in that role by their
    block's FP4 loss (higher = more sensitive → upgrade first). Keep
    the N least-sensitive as FP4, upgrade the rest to FP8."""
    by_role = defaultdict(list)
    LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
    for name in fp4_weights.keys():
        r = role(name)
        mo = LAYER_RE.search(name)
        block_idx = int(mo.group(1)) if mo else -1
        loss = block_loss.get(block_idx, 0.0)
        by_role[r].append((name, loss))

    decisions = {}
    for r, entries in by_role.items():
        n_total = len(entries)
        if r in recipe:
            frac = recipe[r]
        else:
            # Unknown role — keep as FP4
            frac = 1.0
            print(f"  warning: role {r!r} not in recipe, keeping 100% FP4")
        n_fp4 = round(frac * n_total)
        # Sort by block loss ascending; keep the LOWEST-loss as FP4 (least
        # sensitive), upgrade the HIGHEST-loss to FP8 (most sensitive).
        sorted_entries = sorted(entries, key=lambda x: x[1])
        for i, (name, _) in enumerate(sorted_entries):
            decisions[name] = "fp4" if i < n_fp4 else "fp8"
    return decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--recipe", required=True,
                        help="JSON file with role → target FP4 fraction")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ar-log", default=None,
                        help="Optional AutoRound log for per-block loss ranking")
    args = parser.parse_args()

    # Load recipe
    with open(args.recipe) as f:
        recipe = json.load(f)
    print(f"[recipe] loaded {len(recipe)} role entries")

    # Load cache (handles both single-file and sharded formats)
    cache_p = Path(args.cache_dir)
    fp4_weights = load_weights_possibly_sharded(cache_p, "fp4")
    fp8_weights = load_weights_possibly_sharded(cache_p, "fp8")
    with open(cache_p / "cache_manifest.json") as f:
        cache_meta = json.load(f)
    print(f"[recipe] cache: {len(fp4_weights)} fp4 weights, {len(fp8_weights)} fp8 weights")
    print(f"[recipe]   RTN-FP4 baseline KL: {cache_meta.get('rtn_fp4_baseline_kl', '?')}")
    print(f"[recipe]   AutoRound-FP4 KL:    {cache_meta.get('autoround_fp4_kl', '?')}")
    print(f"[recipe]   AutoRound-FP8 KL:    {cache_meta.get('autoround_fp8_kl', '?')}")

    # Parse AR log for per-block losses (optional)
    block_loss = {}
    if args.ar_log:
        block_loss = parse_ar_losses(args.ar_log)
        print(f"[recipe] parsed per-block losses for {len(block_loss)} blocks")
    else:
        print(f"[recipe] no AR log — using layer index for intra-role tiebreaking")

    # Build per-linear decisions
    decisions = build_decisions(fp4_weights, recipe, block_loss)
    counts = defaultdict(int)
    for fmt in decisions.values():
        counts[fmt] += 1
    print(f"[recipe] decisions: {dict(counts)}")

    # Compute average cost (fp4=1, fp8=2)
    total_cost = sum(1 if f == "fp4" else 2 for f in decisions.values())
    avg_cost = total_cost / max(1, len(decisions))
    print(f"[recipe] avg cost vs FP4: {avg_cost:.3f}")

    # Per-role breakdown
    print("[recipe] per-role breakdown:")
    role_counts = defaultdict(lambda: defaultdict(int))
    for name, fmt in decisions.items():
        role_counts[role(name)][fmt] += 1
    for r in sorted(role_counts.keys()):
        rc = role_counts[r]
        total = sum(rc.values())
        fp4_pct = rc.get("fp4", 0) / total * 100
        print(f"  {r:<20} fp4={rc.get('fp4', 0):>3}/{total:<3}  ({fp4_pct:.0f}%)")

    # Load source model + stage if multimodal
    from transformers import AutoModelForCausalLM, AutoTokenizer
    staged, cleanup = stage_multimodal(args.model)
    try:
        print(f"[recipe] loading source model from {staged}")
        model = AutoModelForCausalLM.from_pretrained(
            staged, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
        print(f"[recipe] model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

        # Apply the chosen weights. Handle both nn.Linear (name ends in .weight)
        # and fused MoE experts (name ends in .gate_up_proj or .down_proj).
        n_applied = 0
        n_missing = 0
        for name, fmt in decisions.items():
            src_dict = fp4_weights if fmt == "fp4" else fp8_weights
            if name not in src_dict:
                n_missing += 1
                continue
            src_tensor = src_dict[name]
            # Resolve name → (module, attribute) on the model
            if name.endswith(".weight"):
                mod_path = name[: -len(".weight")]
                mod = dict(model.named_modules()).get(mod_path)
                if mod is None or not hasattr(mod, "weight"):
                    n_missing += 1
                    continue
                with torch.no_grad():
                    mod.weight.data.copy_(src_tensor.to(mod.weight.dtype))
                n_applied += 1
            else:
                # Fused expert: path = model.layers.N.mlp.experts.{gate_up_proj|down_proj}
                parts = name.rsplit(".", 1)
                mod_path, attr = parts[0], parts[1]
                mod = dict(model.named_modules()).get(mod_path)
                if mod is None or not hasattr(mod, attr):
                    n_missing += 1
                    continue
                with torch.no_grad():
                    target = getattr(mod, attr)
                    target.data.copy_(src_tensor.to(target.dtype))
                n_applied += 1
        print(f"[recipe] applied {n_applied} weights  ({n_missing} missing)")

        # Save
        Path(args.output).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)

        # Manifest
        manifest = {
            "source_model": args.model,
            "cache_dir": args.cache_dir,
            "recipe_file": args.recipe,
            "recipe": recipe,
            "counts": dict(counts),
            "avg_cost_vs_fp4": avg_cost,
            "decisions": decisions,
            "rtn_fp4_baseline_kl": cache_meta.get("rtn_fp4_baseline_kl"),
            "autoround_fp4_kl": cache_meta.get("autoround_fp4_kl"),
            "autoround_fp8_kl": cache_meta.get("autoround_fp8_kl"),
        }
        with open(Path(args.output) / "apply_role_recipe_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[recipe] saved to {args.output}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
