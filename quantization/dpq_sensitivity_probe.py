#!/usr/bin/env python3
"""
Leave-one-out KL sensitivity probe.

For each block, measures how much the model KL changes when that block's
weights are upgraded from AutoRound-FP4 to AutoRound-FP8 (with the rest
of the model held at AutoRound-FP4).

This is a GLOBAL measurement with LOCAL perturbation — it directly
captures each block's contribution to final KL, which is what DPQ's
gradient descent optimizes. Much better predictor than AutoRound's
per-block reconstruction loss (which is block-isolated).

Output:
  - sensitivities.json: {block_idx: {delta_kl, cumulative_delta_kl, ...}}
  - A recommended recipe: upgrade top-K blocks to FP8 until the
    cumulative ΔKL flattens (Kneedle)

Usage:
    python3 dpq_sensitivity_probe.py \\
        --model /path/to/bf16/model \\
        --cache-dir /tmp/dpq_cache/xxx \\
        --output /tmp/sensitivity_probe_output.json
"""
import argparse
import gc
import json
import math
import re
import tempfile
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Model staging (multimodal vision-strip workaround)
# ---------------------------------------------------------------------------

def stage_multimodal(model_path: str) -> Tuple[str, str | None]:
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
    staged = tempfile.mkdtemp(prefix="probe_staged_")
    for p in Path(model_path).iterdir():
        if p.name == "config.json":
            continue
        (Path(staged) / p.name).symlink_to(p.resolve())
    with open(Path(staged) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return staged, staged


# ---------------------------------------------------------------------------
# Calibration + KL measurement (same as dpq_autoround_first.py)
# ---------------------------------------------------------------------------

def load_wikitext_calibration(tokenizer, n_samples: int, seqlen: int) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    enc = tokenizer(text, return_tensors="pt", truncation=False).input_ids
    total = enc.size(1)
    max_start = total - seqlen
    import random
    random.seed(42)
    starts = random.sample(range(max_start), n_samples) if max_start >= n_samples else \
             [i * (max_start // n_samples) for i in range(n_samples)]
    batches = torch.stack([enc[0, s:s + seqlen] for s in starts], dim=0)
    return batches  # [n_samples, seqlen]


@torch.no_grad()
def cache_reference_log_probs(model, calib_ids, device):
    log_probs = []
    for i in range(calib_ids.size(0)):
        batch = calib_ids[i:i + 1].to(device)
        logits = model(batch).logits
        lp = F.log_softmax(logits, dim=-1).cpu()
        log_probs.append(lp)
    return log_probs


def kl_divergence(student_logits, teacher_log_probs):
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher = teacher_log_probs.to(student_logits.device)
    return F.kl_div(student_log_probs, teacher, reduction="batchmean", log_target=True)


@torch.no_grad()
def measure_kl(model, calib_ids, ref_log_probs, device) -> float:
    kls = []
    for i in range(calib_ids.size(0)):
        batch = calib_ids[i:i + 1].to(device)
        logits = model(batch).logits
        kls.append(kl_divergence(logits, ref_log_probs[i]).item())
    return sum(kls) / len(kls)


# ---------------------------------------------------------------------------
# Sensitivity probe
# ---------------------------------------------------------------------------

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(.+)$")


def group_linears_by_block(linear_names: List[str]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = defaultdict(list)
    for name in linear_names:
        mo = LAYER_RE.match(name)
        if mo:
            out[int(mo.group(1))].append(name)
    return dict(out)


def apply_weights_to_model(model: nn.Module, weights: Dict[str, torch.Tensor],
                           target_names: List[str] | None = None, device: str = "cuda"):
    """Copy `weights[name]` into the corresponding Linear weight (in place)."""
    name_to_mod = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    targets = target_names if target_names is not None else list(weights.keys())
    for name in targets:
        if name in weights and name in name_to_mod:
            name_to_mod[name].weight.data.copy_(weights[name].to(device))


def find_knee_idx(cumsum: List[float]) -> int:
    """Kneedle: max perpendicular distance from chord."""
    n = len(cumsum)
    if n < 3:
        return n - 1
    xs = list(range(n))
    ys = cumsum
    x_min, x_max = 0, n - 1
    y_min, y_max = min(ys), max(ys)
    xr = x_max - x_min or 1
    yr = (y_max - y_min) or 1
    norm = [((x - x_min) / xr, (y - y_min) / yr) for x, y in zip(xs, ys)]
    x1, y1 = norm[0]
    x2, y2 = norm[-1]
    denom = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) or 1.0
    best_i, best_d = 0, -1.0
    for i, (x, y) in enumerate(norm):
        d = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
        if d > best_d:
            best_i, best_d = i, d
    return best_i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-calib-samples", type=int, default=8)
    parser.add_argument("--calib-seqlen", type=int, default=256)
    args = parser.parse_args()

    t_start = time.time()

    # Load cache
    print(f"[probe] loading cache from {args.cache_dir}")
    cache_paths = {
        "fp4": Path(args.cache_dir) / "fp4_weights.safetensors",
        "fp8": Path(args.cache_dir) / "fp8_weights.safetensors",
        "meta": Path(args.cache_dir) / "cache_manifest.json",
    }
    fp4_weights = load_file(str(cache_paths["fp4"]))
    fp8_weights = load_file(str(cache_paths["fp8"]))
    with open(cache_paths["meta"]) as f:
        cache_meta = json.load(f)
    print(f"[probe]   fp4: {len(fp4_weights)} weights, fp8: {len(fp8_weights)} weights")

    # Load source model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    staged, cleanup = stage_multimodal(args.model)
    try:
        print(f"[probe] loading model from {staged}")
        model = AutoModelForCausalLM.from_pretrained(
            staged, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
        device = next(model.parameters()).device

        # Cache reference log probs from the original BF16 model
        calib_ids = load_wikitext_calibration(tokenizer, args.n_calib_samples, args.calib_seqlen)
        print(f"[probe] computing BF16 reference log_probs ({args.n_calib_samples}×{args.calib_seqlen})")
        ref_log_probs = cache_reference_log_probs(model, calib_ids, device)

        # Group FP4/FP8 linears by block index
        linear_names = list(fp4_weights.keys())
        by_block = group_linears_by_block(linear_names)
        print(f"[probe] grouped {len(linear_names)} linears into {len(by_block)} blocks")

        # Set model to all-FP4 baseline
        print(f"[probe] applying all-FP4 baseline")
        apply_weights_to_model(model, fp4_weights, device=device)
        baseline_kl = measure_kl(model, calib_ids, ref_log_probs, device)
        print(f"[probe]   baseline (all-FP4) KL = {baseline_kl:.6f}")

        # Leave-one-out probe: for each block, upgrade ALL its linears to FP8
        print(f"[probe] starting leave-one-out probe over {len(by_block)} blocks")
        sensitivity = []
        for block_idx in sorted(by_block.keys()):
            names_in_block = by_block[block_idx]
            # Upgrade this block to FP8
            apply_weights_to_model(model, fp8_weights, target_names=names_in_block, device=device)
            kl_after = measure_kl(model, calib_ids, ref_log_probs, device)
            delta_kl = baseline_kl - kl_after  # positive = upgrading this block helped
            # Revert block to FP4
            apply_weights_to_model(model, fp4_weights, target_names=names_in_block, device=device)
            sensitivity.append({
                "block": block_idx,
                "n_linears": len(names_in_block),
                "kl_after": kl_after,
                "delta_kl": delta_kl,
            })
            if (block_idx + 1) % 10 == 0 or block_idx == sorted(by_block.keys())[-1]:
                print(f"[probe]   block {block_idx:3d}: KL={kl_after:.6f} ΔKL={delta_kl:+.6f}")

        # Sort by ΔKL descending (most beneficial first)
        sensitivity.sort(key=lambda r: -r["delta_kl"])
        cumulative = []
        cumsum = 0.0
        for r in sensitivity:
            cumsum += r["delta_kl"]
            cumulative.append(cumsum)
        knee_idx = find_knee_idx(cumulative)

        print(f"\n[probe] top-{knee_idx + 1} most sensitive blocks (upgrade these to FP8):")
        for i, r in enumerate(sensitivity[:knee_idx + 1]):
            print(f"   {i+1:3d}. block {r['block']:3d}  ΔKL={r['delta_kl']:+.6f}  cumsum={cumulative[i]:.6f}")
        print(f"\n[probe] knee at upgrade-count = {knee_idx + 1}/{len(sensitivity)} "
              f"(total ΔKL recovered: {cumulative[knee_idx]:.6f})")
        print(f"[probe] diminishing returns zone: blocks {knee_idx + 2}..{len(sensitivity)}")

        # Save full results
        result = {
            "model": args.model,
            "cache_dir": args.cache_dir,
            "baseline_all_fp4_kl": baseline_kl,
            "all_fp8_reference_kl": cache_meta.get("autoround_fp8_kl"),
            "n_blocks": len(sensitivity),
            "knee_block_count": knee_idx + 1,
            "cumulative_delta_at_knee": cumulative[knee_idx],
            "blocks_sorted_by_sensitivity": sensitivity,
            "elapsed_sec": time.time() - t_start,
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[probe] saved to {args.output}")
        print(f"[probe] total time: {time.time() - t_start:.0f}s")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
