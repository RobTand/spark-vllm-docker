#!/usr/bin/env python3
"""calibrate_allocator.py — empirical calibration for DynaQuant frontier points.

Given:
  - sensitivity probe pickle
  - measured per-format cost pickle
  - a target set of average-bit budgets

This script:
  1. Rebuilds DynaQuant assignments for each target
  2. Applies the chosen native formats in-memory to a real model
  3. Measures actual KL against the BF16 reference logits on a small
     calibration corpus

The goal is not to replace the allocator. The goal is to empirically
calibrate the predicted frontier so we can answer:
  - is the knee actually a good operating point?
  - do predicted Δloss and measured KL rank frontier points consistently?
  - does a particular format bundle systematically under/over-predict quality?
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from quantization.build_rtn_cache import (
    cache_reference_log_probs,
    iter_quantizable_tensors,
    kl_divergence,
    load_wikitext_calibration,
    stage_multimodal,
)
from quantization.dynaquant import format_registry as fr
from quantization.dynaquant.allocator import (
    aggregate_moe_candidates,
    build_candidates,
    compute_achieved,
    expand_moe_assignment,
    kneedle,
    promote_fused,
    solve_allocation,
)


def load_inputs(probe_path: Path, costs_path: Path, fmt_names: list[str]):
    import pickle

    with open(probe_path, "rb") as f:
        probe = pickle.load(f)
    with open(costs_path, "rb") as f:
        cost_data = pickle.load(f)

    stats = probe["stats"]
    costs = cost_data["costs"]
    specs = [fr.get_format(n) for n in fmt_names]
    specs_sorted = sorted(specs, key=lambda s: s.effective_bits)
    return stats, costs, specs_sorted


def build_curve(stats: dict, costs: dict, specs_sorted, targets: list[float], bit_precision: float,
                no_fused_promote: bool, expert_granularity: str):
    format_rank = {s.name: i for i, s in enumerate(specs_sorted)}
    format_specs = {s.name: s for s in specs_sorted}

    candidates = build_candidates(stats, costs, specs_sorted)
    stats_alloc = stats
    costs_alloc = costs
    if expert_granularity == "layer":
        stats_alloc, costs_alloc, candidates = aggregate_moe_candidates(
            stats, costs, specs_sorted, candidates
        )

    curve = []
    for t in targets:
        assignment = solve_allocation(stats_alloc, candidates, t, bit_precision)
        if assignment is None:
            curve.append({"target_bits": t, "feasible": False})
            continue
        if not no_fused_promote:
            assignment = promote_fused(assignment, format_rank)
        achieved, _ = compute_achieved(stats_alloc, assignment, format_specs)
        predicted_dloss = 0.0
        for name, fmt in assignment.items():
            entry = costs_alloc[name].get(fmt, {})
            d_out = stats_alloc[name]["out_features"]
            predicted_dloss += 0.5 * stats_alloc[name]["h_trace"] * entry.get("output_mse", 0.0) * d_out
        curve.append({
            "target_bits": t,
            "feasible": True,
            "achieved_bits": achieved,
            "predicted_dloss": predicted_dloss,
            "assignment": assignment,
            "stats_scope": "aggregated" if expert_granularity == "layer" else "expert",
        })
    return curve, stats_alloc, costs_alloc, format_rank


def build_module_param_map(model):
    out = {}
    for full_name, mod, attr in iter_quantizable_tensors(model):
        out[full_name] = (mod, attr)
        bare_name = full_name[:-7] if full_name.endswith(".weight") else full_name
        out[bare_name] = (mod, attr)
        if full_name.startswith("model."):
            out[f"model.language_model.{full_name[len('model.') :]}"] = (mod, attr)
            out[
                f"model.language_model.{bare_name[len('model.') :]}"
                if bare_name.startswith("model.")
                else f"model.language_model.{bare_name}"
            ] = (mod, attr)
    return out


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    if xr.std() == 0 or yr.std() == 0:
        return None
    return float(np.corrcoef(xr.astype(np.float64), yr.astype(np.float64))[0, 1])


@torch.no_grad()
def measure_avg_last_token_kl(model, calib_ids: torch.Tensor, ref_log_probs, device: torch.device) -> float:
    kls = []
    for i in range(calib_ids.size(0)):
        batch = calib_ids[i : i + 1].to(device)
        logits = model(batch).logits[:, -1:, :]
        teacher = ref_log_probs[i][:, -1:, :]
        kls.append(float(kl_divergence(logits, teacher).item()))
    return sum(kls) / max(len(kls), 1)


def apply_recipe_in_place(model, assignment_expanded: dict[str, str], quant_map: dict[str, tuple]):
    originals = {}
    for name, fmt in assignment_expanded.items():
        target = quant_map.get(name)
        if target is None:
            continue
        mod, attr = target
        original = getattr(mod, attr).data.detach().clone()
        originals[name] = (mod, attr, original)
        q = fr.get_format(fmt).quantize_dequantize(original)
        getattr(mod, attr).data.copy_(q.to(device=original.device, dtype=original.dtype))
    return originals


def install_activation_hooks(
    assignment_expanded: dict[str, str],
    quant_map: dict[str, tuple],
):
    module_specs = {}
    skipped = []
    for name, fmt in assignment_expanded.items():
        target = quant_map.get(name)
        if target is None:
            continue
        mod, _attr = target
        spec = fr.get_format(fmt)
        key = id(mod)
        prev = module_specs.get(key)
        if prev is None:
            module_specs[key] = (mod, spec, [name])
            continue
        prev_mod, prev_spec, prev_names = prev
        prev_names.append(name)
        if prev_spec.name != spec.name:
            skipped.append(
                {
                    "module": type(mod).__name__,
                    "weights": sorted(prev_names),
                    "formats": sorted({prev_spec.name, spec.name}),
                }
            )
            module_specs[key] = (prev_mod, None, prev_names)

    handles = []
    active = []
    for mod, spec, names in module_specs.values():
        if spec is None:
            continue
        if spec.act_bits is None or spec.act_bits >= 16:
            continue
        quant_fn = spec.activation_quantize_dequantize

        def _pre_hook(_mod, args, kwargs, quant_fn=quant_fn):
            if args:
                x = args[0]
                qx = quant_fn(x)
                args = (qx,) + tuple(args[1:])
            if kwargs and "hidden_states" in kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = quant_fn(kwargs["hidden_states"])
            return args, kwargs

        handles.append(mod.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        active.append({"module": type(mod).__name__, "weights": sorted(names), "format": spec.name})
    return handles, active, skipped


def restore_in_place(originals: dict):
    for _name, (mod, attr, original) in originals.items():
        getattr(mod, attr).data.copy_(original)


def select_targets(curve: list[dict], mode: str) -> list[int]:
    feasible = [i for i, row in enumerate(curve) if row.get("feasible")]
    if not feasible:
        return []
    if mode == "all":
        return feasible
    if mode == "knee":
        rows = [curve[i] for i in feasible]
        knee_local = kneedle([r["achieved_bits"] for r in rows], [r["predicted_dloss"] for r in rows])
        return [feasible[knee_local]]
    if mode == "baseline,knee,high":
        rows = [curve[i] for i in feasible]
        knee_local = kneedle([r["achieved_bits"] for r in rows], [r["predicted_dloss"] for r in rows])
        picks = {feasible[0], feasible[knee_local], feasible[-1]}
        return sorted(picks)
    raise ValueError(f"unknown selection mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--formats", required=True,
                    help="Comma-separated format names, e.g. NVFP4,MXFP8")
    ap.add_argument("--pareto-targets", default="4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25")
    ap.add_argument("--selection", default="baseline,knee,high",
                    choices=["baseline,knee,high", "knee", "all"])
    ap.add_argument("--bit-precision", type=float, default=0.001)
    ap.add_argument("--expert-granularity", choices=["layer", "expert"], default="layer")
    ap.add_argument("--no-fused-promote", action="store_true")
    ap.add_argument("--n-calib-samples", type=int, default=4)
    ap.add_argument("--calib-seqlen", type=int, default=128)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    fmt_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    stats, costs, specs_sorted = load_inputs(Path(args.probe), Path(args.costs), fmt_names)
    curve, stats_alloc, _costs_alloc, format_rank = build_curve(
        stats,
        costs,
        specs_sorted,
        [float(x) for x in args.pareto_targets.split(",")],
        args.bit_precision,
        args.no_fused_promote,
        args.expert_granularity,
    )
    selected = select_targets(curve, args.selection)
    if not selected:
        raise SystemExit("no feasible points to calibrate")

    model_arg = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    staged, cleanup = stage_multimodal(model_arg)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if args.device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = args.device
        device = torch.device(device_str)
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer_kwargs = dict(trust_remote_code=True)
        if Path(staged).exists():
            load_kwargs["local_files_only"] = True
            tokenizer_kwargs["local_files_only"] = True
        if device.type == "cuda":
            load_kwargs["device_map"] = device_str

        model = AutoModelForCausalLM.from_pretrained(
            staged,
            **load_kwargs,
        )
        if device.type != "cuda":
            model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(staged, **tokenizer_kwargs)
        quant_map = build_module_param_map(model)

        calib_ids = load_wikitext_calibration(tokenizer, args.n_calib_samples, args.calib_seqlen)
        ref_log_probs = cache_reference_log_probs(model, calib_ids, device)
        baseline_kl = measure_avg_last_token_kl(model, calib_ids, ref_log_probs, device)

        results = []
        for idx in selected:
            row = curve[idx]
            assignment = row["assignment"]
            if args.expert_granularity == "layer":
                assignment_expanded = expand_moe_assignment(assignment, stats_alloc)
            else:
                assignment_expanded = assignment

            originals = apply_recipe_in_place(model, assignment_expanded, quant_map)
            hook_handles, active_hooks, skipped_hooks = install_activation_hooks(
                assignment_expanded, quant_map
            )
            try:
                actual_kl = measure_avg_last_token_kl(model, calib_ids, ref_log_probs, device)
            finally:
                for handle in hook_handles:
                    handle.remove()
                restore_in_place(originals)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            counts = {}
            for fmt in assignment.values():
                counts[fmt] = counts.get(fmt, 0) + 1

            results.append({
                "curve_index": idx,
                "target_bits": row["target_bits"],
                "achieved_bits": row["achieved_bits"],
                "predicted_dloss": row["predicted_dloss"],
                "actual_last_token_kl": actual_kl,
                "delta_from_baseline_kl": actual_kl - baseline_kl,
                "format_counts": counts,
                "activation_hook_count": len(active_hooks),
                "activation_hook_modules": active_hooks,
                "activation_hook_skipped": skipped_hooks,
            })
            print(
                f"[cal] idx={idx} target={row['target_bits']:.3f} "
                f"achieved={row['achieved_bits']:.3f} pred={row['predicted_dloss']:.4e} "
                f"kl={actual_kl:.4e} hooks={len(active_hooks)} skipped={len(skipped_hooks)}",
                flush=True,
            )

        predicted = np.array([r["predicted_dloss"] for r in results], dtype=np.float64)
        actual = np.array([r["actual_last_token_kl"] for r in results], dtype=np.float64)
        pearson = (
            float(np.corrcoef(predicted, actual)[0, 1])
            if len(results) >= 2 and predicted.std() > 0 and actual.std() > 0
            else None
        )
        spearman = _spearman_corr(predicted, actual)

        out = {
            "model": args.model,
            "formats": fmt_names,
            "pareto_targets": args.pareto_targets,
            "selection": args.selection,
            "bit_precision": args.bit_precision,
            "expert_granularity": args.expert_granularity,
            "baseline_last_token_kl": baseline_kl,
            "correlation_predicted_vs_actual_pearson": pearson,
            "correlation_predicted_vs_actual_spearman": spearman,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[cal] wrote {args.output}", flush=True)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
