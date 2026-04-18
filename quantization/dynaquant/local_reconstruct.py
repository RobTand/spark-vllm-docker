#!/usr/bin/env python3
"""local_reconstruct.py — improve critical native-format candidates locally.

For a small set of important layers, refine per-format costs by grid-searching
simple symmetric clipping factors on weights and activations, minimizing the
measured layer output MSE on cached activations.

This is intentionally conservative:
  - one layer at a time
  - one format at a time
  - tiny clip grids
  - no extra optimizer state
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from . import format_registry as fr
from .allocator import aggregate_moe_candidates, build_candidates, promote_fused, solve_allocation
from .calibrate_allocator import load_inputs
from .interaction_refine import build_refinement_units, select_critical_units
from .measure_quant_cost import ActivationIndex, _load_live_model


def _sym_clip(x: torch.Tensor, factor: float) -> torch.Tensor:
    if factor >= 0.999999:
        return x
    max_abs = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    limit = max_abs * factor
    return x.clamp(-limit, limit)


def _measure_entry(W: torch.Tensor, X: torch.Tensor, spec: fr.FormatSpec, w_clip: float, a_clip: float):
    W_in = _sym_clip(W, w_clip)
    X_in = _sym_clip(X, a_clip)
    W_hat = spec.quantize_dequantize(W_in.clone())
    X_hat = spec.activation_quantize_dequantize(X_in.clone())
    y_ref = X @ W.T
    y_q = X_hat @ W_hat.T
    weight_mse = float((W - W_hat).float().pow(2).mean().item())
    output_mse = float((y_ref - y_q).float().pow(2).mean().item())
    ref_energy = float(y_ref.float().pow(2).mean().item())
    return {
        "weight_mse": weight_mse,
        "output_mse": output_mse,
        "rel_output_mse": output_mse / max(ref_energy, 1e-12),
        "weight_clip": w_clip,
        "act_clip": a_clip,
    }


def expand_live_target_layers(critical_units, stats_alloc: dict) -> set[str]:
    target_layers = set()
    for unit in critical_units:
        for member in unit.members:
            fused_members = stats_alloc.get(member, {}).get("_fused_members")
            if fused_members:
                target_layers.update(fused_members)
            else:
                target_layers.add(member)
    return target_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--activation-cache-dir", required=True)
    ap.add_argument("--formats", required=True)
    ap.add_argument("--target-bits", type=float, required=True)
    ap.add_argument("--top-units", type=int, default=8)
    ap.add_argument("--w-clip-grid", default="1.0,0.995,0.99,0.98,0.95,0.9")
    ap.add_argument("--a-clip-grid", default="1.0,0.995,0.99,0.98,0.95,0.9")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--expert-granularity", choices=["layer", "expert"], default="layer")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    fmt_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    stats, costs, specs_sorted = load_inputs(Path(args.probe), Path(args.costs), fmt_names)
    candidates = build_candidates(stats, costs, specs_sorted)
    stats_alloc = stats
    if args.expert_granularity == "layer":
        stats_alloc, costs, candidates = aggregate_moe_candidates(stats, costs, specs_sorted, candidates)
    format_rank = {s.name: i for i, s in enumerate(specs_sorted)}
    assignment = solve_allocation(stats_alloc, candidates, args.target_bits, 0.001)
    if assignment is None:
        raise SystemExit("no feasible assignment at requested target")
    assignment = promote_fused(assignment, format_rank)
    units = build_refinement_units(stats_alloc, candidates, assignment)
    critical = select_critical_units(units, args.top_units)
    target_layers = expand_live_target_layers(critical, stats_alloc)

    with open(args.costs, "rb") as f:
        cost_blob = pickle.load(f)
    raw_costs = cost_blob["costs"]
    act_cache = ActivationIndex(Path(args.activation_cache_dir), target_layers)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = _load_live_model(args.model, args.device, dtype, unfuse_moe=True)

    module_map = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name in target_layers:
            module_map[name] = mod

    w_grid = [float(x) for x in args.w_clip_grid.split(",") if x.strip()]
    a_grid = [float(x) for x in args.a_clip_grid.split(",") if x.strip()]
    upgraded = {}
    for layer_name in sorted(target_layers):
        if layer_name not in module_map or layer_name not in act_cache:
            continue
        mod = module_map[layer_name]
        W = mod.weight.detach()
        X = act_cache.load(layer_name).to(W.dtype).to(W.device)
        per_fmt = {}
        for spec in specs_sorted:
            if spec.name not in raw_costs.get(layer_name, {}):
                continue
            best = None
            for w_clip in w_grid:
                for a_clip in a_grid:
                    try:
                        entry = _measure_entry(W, X, spec, w_clip, a_clip)
                    except Exception as exc:
                        entry = {"error": str(exc), "weight_clip": w_clip, "act_clip": a_clip}
                    if "error" in entry:
                        continue
                    if best is None or entry["output_mse"] < best["output_mse"]:
                        best = entry
            if best is not None:
                best["source"] = "local_reconstruct"
                per_fmt[spec.name] = best
        if per_fmt:
            upgraded[layer_name] = per_fmt
            for fmt, entry in per_fmt.items():
                raw_costs.setdefault(layer_name, {})[fmt] = entry
                print(
                    f"[reconstruct] {layer_name} {fmt} output_mse={entry['output_mse']:.4e} "
                    f"w_clip={entry['weight_clip']:.3f} a_clip={entry['act_clip']:.3f}",
                    flush=True,
                )

    cost_blob["costs"] = raw_costs
    meta = dict(cost_blob.get("meta", {}))
    meta["local_reconstruct"] = {
        "target_bits": args.target_bits,
        "top_units": args.top_units,
        "formats": fmt_names,
        "w_clip_grid": w_grid,
        "a_clip_grid": a_grid,
        "layers_refined": sorted(upgraded),
    }
    cost_blob["meta"] = meta
    with open(args.output, "wb") as f:
        pickle.dump(cost_blob, f)
    print(f"[reconstruct] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
