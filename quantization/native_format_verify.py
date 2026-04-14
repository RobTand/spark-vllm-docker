#!/usr/bin/env python3
"""
Model-aware verification for native deployment buckets.

This complements native_format_study.py by replacing selected tensors in a
real model and measuring last-token KL divergence on calibration prompts.
Use it when weight-MSE proxies are no longer trustworthy enough, e.g. for
plain FP8 vs MXFP8 comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantization.build_rtn_cache import (
    cache_reference_log_probs,
    iter_quantizable_tensors,
    kl_divergence,
    load_wikitext_calibration,
    stage_multimodal,
)
from quantization.joint_knapsack_optimizer import (
    candidate_sensitivity_names,
    discover_model_layers,
    load_hawq_sensitivity,
    load_layer_tensor,
)
from quantization.native_format_study import BUCKETS, quantize_tensor_to_bucket


def build_module_param_map(model) -> Dict[str, Tuple[torch.nn.Module, str]]:
    out = {}
    for full_name, mod, attr in iter_quantizable_tensors(model):
        out[full_name] = (mod, attr)
        if full_name.startswith("model."):
            out[f"model.language_model.{full_name[len('model.') :]}"] = (mod, attr)
    return out


def score_layers_from_study(
    study_path: Path,
    hawq_path: Path,
    source_bucket: str,
) -> List[Tuple[str, float]]:
    with open(study_path) as f:
        study = json.load(f)
    hawq = load_hawq_sensitivity(str(hawq_path))
    recipe = study["promotion_knee"]["recipe"]
    scored = []
    for name, bucket in recipe.items():
        if bucket != source_bucket:
            continue
        sens = None
        for cand in candidate_sensitivity_names(name):
            sens = hawq.get(cand)
            if sens is not None:
                break
        scored.append((name, float(sens or 0.0)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def measure_avg_last_token_kl(model, calib_ids: torch.Tensor, ref_log_probs: List[torch.Tensor], device: torch.device) -> float:
    kls = []
    with torch.no_grad():
        for i in range(calib_ids.size(0)):
            batch = calib_ids[i:i + 1].to(device)
            logits = model(batch).logits[:, -1:, :]
            teacher = ref_log_probs[i][:, -1:, :]
            kls.append(float(kl_divergence(logits, teacher).item()))
    return sum(kls) / max(len(kls), 1)


def main():
    parser = argparse.ArgumentParser(description="Verify native buckets with model-aware KL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--study", required=True, help="native_format_study JSON")
    parser.add_argument("--sensitivity", required=True)
    parser.add_argument("--source-bucket", default="fp8_e4m3")
    parser.add_argument("--bucket-a", default="fp8_e4m3")
    parser.add_argument("--bucket-b", default="mxfp8_e4m3")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--n-calib-samples", type=int, default=4)
    parser.add_argument("--calib-seqlen", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    staged, cleanup = stage_multimodal(args.model)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[verify] loading {args.model}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            staged,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        device = next(model.parameters()).device
        print(f"[verify] model on {device}", flush=True)

        print(f"[verify] loading calibration", flush=True)
        calib_ids = load_wikitext_calibration(tokenizer, args.n_calib_samples, args.calib_seqlen)
        print(f"[verify] caching baseline log-probs", flush=True)
        ref_log_probs = cache_reference_log_probs(model, calib_ids, device)
        base_kl = measure_avg_last_token_kl(model, calib_ids, ref_log_probs, device)

        quant_map = build_module_param_map(model)
        raw_layers = discover_model_layers(Path(args.model), None, "text-only")
        layer_lookup = {r["name"]: r for r in raw_layers}

        candidates = score_layers_from_study(Path(args.study), Path(args.sensitivity), args.source_bucket)[: args.top_k]
        print(f"[verify] selected {len(candidates)} layers from {args.source_bucket}", flush=True)

        results = []
        for idx, (name, sens) in enumerate(candidates, start=1):
            if name not in quant_map or name not in layer_lookup:
                print(f"[verify] skipping unmapped layer {name}", flush=True)
                continue
            mod, attr = quant_map[name]
            ref = layer_lookup[name]
            original = getattr(mod, attr).data.detach().clone()
            source = load_layer_tensor(ref).to(dtype=original.dtype)
            if tuple(source.shape) != tuple(original.shape):
                print(f"[verify] skipping shape mismatch {name}: checkpoint {tuple(source.shape)} vs model {tuple(original.shape)}", flush=True)
                continue

            entry = {"name": name, "sensitivity": sens}
            for label, bucket_name in (("a", args.bucket_a), ("b", args.bucket_b)):
                bucket = BUCKETS[bucket_name]
                quant = quantize_tensor_to_bucket(source, bucket).to(device=original.device, dtype=original.dtype)
                getattr(mod, attr).data.copy_(quant)
                kl = measure_avg_last_token_kl(model, calib_ids, ref_log_probs, device)
                entry[f"{label}_bucket"] = bucket_name
                entry[f"{label}_kl"] = kl
                entry[f"{label}_delta"] = kl - base_kl
                del quant
                torch.cuda.empty_cache()

            getattr(mod, attr).data.copy_(original)
            results.append(entry)
            better = "a" if entry["a_kl"] <= entry["b_kl"] else "b"
            print(
                f"[verify] {idx}/{len(candidates)} {name} "
                f"{args.bucket_a}={entry['a_kl']:.4e} {args.bucket_b}={entry['b_kl']:.4e} "
                f"winner={better}",
                flush=True,
            )

        with open(args.output, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "study": args.study,
                    "source_bucket": args.source_bucket,
                    "bucket_a": args.bucket_a,
                    "bucket_b": args.bucket_b,
                    "n_calib_samples": args.n_calib_samples,
                    "calib_seqlen": args.calib_seqlen,
                    "baseline_kl": base_kl,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"[verify] saved to {args.output}", flush=True)
    finally:
        if cleanup:
            import shutil

            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
