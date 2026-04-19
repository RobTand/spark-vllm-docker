#!/usr/bin/env python3
"""nvfp4_mxfp8_allocator.py — turn HAWQ stats into an NVFP4/MXFP8 recipe.

Takes the streaming_hawq.py output and a target average bit rate, emits:
  - AutoRound layer_config JSON (drop-in for --layer_config)
  - Pareto curve CSV (avg_bits vs predicted ΔKL)

Format constraints enforced (microscaling preferred block sizes):
    NVFP4 : num_bits=4, type=float,  strategy=tensor_group, group_size=16
            scale_dtype=fp8_e4m3       (per NVIDIA NVFP4 spec)
    MXFP8 : num_bits=8, type=float,  strategy=group,        group_size=32
            scale_dtype=uint8          (per OCP MX spec, E8M0)

Fused-projection siblings are promoted together (q/k/v share the highest
scheme picked; gate/up share the highest scheme picked) so vLLM's fused
qkv_proj / gate_up_proj loader sees a consistent per-layer type.

ΔKL prediction (analytical):

    For uniform symmetric quantization into 2^b bins across range ±w_max,
    MSE per parameter ≈ w_max² / (12 · (2^(b-1) - 1)²)  [mid-tread model]

    Fisher ΔKL ≈ 0.5 · H_trace · MSE
                = 0.5 · H_trace · w_max² / (12 · (2^(b-1) - 1)²)

    For group-quant with block size g, MSE is reduced roughly by group
    efficiency factor ~√(min(g, 16)/16). Below we use group factors:
        NVFP4  (g=16, fp8 scales): kMSE ≈ 0.12   (measured fit)
        MXFP8  (g=32, uint8 e8m0):  kMSE ≈ 0.0004 (FP8 is much finer)
    These are rough constants — the ALLOCATOR only cares about the RATIO
    ΔKL_nvfp4 / ΔKL_mxfp8 which is bit-count dominated anyway.

Storage cost per parameter (effective bits):
    NVFP4: weights 4 + scales 8/g = 4 + 0.5 = 4.5 bits
    MXFP8: weights 8 + scales 8/g = 8 + 0.25 = 8.25 bits

Greedy allocator:
    Sort layers by "elevation utility" = (ΔKL_nvfp4 - ΔKL_mxfp8) / extra_bits
    Elevate highest-utility layers to MXFP8 until we hit target avg_bits.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path


NVFP4_BITS = 4.5        # weights 4 + fp8 scales / group 16 = 4 + 0.5
MXFP8_BITS = 8.25       # weights 8 + uint8 scales / group 32 = 8 + 0.25
NVFP4_KMSE = 0.12       # empirical group-quant factor
MXFP8_KMSE = 0.0004


def predict_delta_kl(h_trace: float, w_max_abs: float, bits: float,
                     kmse: float) -> float:
    """Analytical Fisher ΔKL estimate for a layer under given bit depth."""
    if h_trace <= 0.0 or w_max_abs <= 0.0:
        return 0.0
    levels = (2 ** (bits - 1) - 1)
    mse_per_param = (w_max_abs ** 2) / (12.0 * (levels ** 2))
    return 0.5 * h_trace * mse_per_param * kmse


def fused_sibling_key(layer_name: str):
    """Return (parent, kind, members) tuple if this layer is part of a fused
    projection group vLLM merges at load time. Else None."""
    # Attention q/k/v
    m = re.search(r"^(?P<pre>.*)\.self_attn\.(?P<suf>q_proj|k_proj|v_proj)$",
                  layer_name)
    if m:
        return (m["pre"], "qkv",
                tuple(f"{m['pre']}.self_attn.{s}" for s in ("q_proj", "k_proj", "v_proj")))
    # MLP gate/up
    m = re.search(r"^(?P<pre>.*)\.mlp\.(?P<suf>gate_proj|up_proj)$", layer_name)
    if m:
        return (m["pre"], "gate_up",
                tuple(f"{m['pre']}.mlp.{s}" for s in ("gate_proj", "up_proj")))
    return None


def promote_fused_groups(assignment: dict) -> dict:
    """Ensure fused-projection siblings share the highest-precision assignment."""
    # Collect siblings
    groups = defaultdict(dict)
    for name, bits in assignment.items():
        key = fused_sibling_key(name)
        if key is None:
            continue
        parent, kind, members = key
        groups[(parent, kind, members)][name] = bits

    out = dict(assignment)
    for (parent, kind, members), present in groups.items():
        if len(present) < 2:
            continue
        best = max(present.values())
        for name in present:
            if out.get(name, 0) < best:
                out[name] = best
    return out


def nvfp4_scheme():
    return {
        "bits": 4,
        "group_size": 16,
        "sym": True,
        "data_type": "nv_fp",
        "act_bits": 4,
        "act_group_size": 16,
        "act_sym": True,
        "act_data_type": "nv_fp4_with_static_gs",
        "act_dynamic": True,
    }


def mxfp8_scheme():
    return {
        "bits": 8,
        "group_size": 32,
        "sym": True,
        "data_type": "mx_fp",
        "act_bits": 8,
        "act_group_size": 32,
        "act_sym": True,
        "act_data_type": "mx_fp",
        "act_dynamic": True,
    }


def allocate_at_budget(stats: dict, target_avg_bits: float):
    """Greedy NVFP4→MXFP8 allocation. Returns (assignment, achieved_avg).

    Utility: how much predicted ΔKL we save per extra bit by upgrading to
    MXFP8. Sort descending, elevate until budget exhausted.
    """
    # Baseline: everyone NVFP4
    total_params = sum(s["n_params"] for s in stats.values())
    if total_params == 0:
        return {}, 0.0

    baseline_bits = total_params * NVFP4_BITS
    target_bits   = total_params * target_avg_bits
    budget_extra  = target_bits - baseline_bits   # may be negative if below min
    if budget_extra <= 0:
        # Target at or below pure NVFP4 cost — everyone gets NVFP4
        return {name: 4 for name in stats}, NVFP4_BITS

    # Compute per-layer upgrade utility
    rows = []
    for name, s in stats.items():
        dl_nv = predict_delta_kl(s["h_trace"], s["w_max_abs"], 4.0, NVFP4_KMSE)
        dl_mx = predict_delta_kl(s["h_trace"], s["w_max_abs"], 8.0, MXFP8_KMSE)
        saved = dl_nv - dl_mx
        extra_bits = s["n_params"] * (MXFP8_BITS - NVFP4_BITS)  # = 3.75 × params
        utility = saved / max(extra_bits, 1e-9)
        rows.append((name, utility, extra_bits, saved, s["n_params"]))
    rows.sort(key=lambda r: -r[1])

    assignment = {name: 4 for name in stats}
    spent_extra = 0.0
    elevated_params = 0
    for name, utility, extra_bits, saved, n_params in rows:
        if spent_extra + extra_bits > budget_extra:
            continue
        assignment[name] = 8
        spent_extra += extra_bits
        elevated_params += n_params

    # Promote fused siblings
    assignment = promote_fused_groups(assignment)

    # Recompute achieved avg_bits after promotion (may exceed target slightly)
    total_cost = sum(stats[n]["n_params"] *
                     (MXFP8_BITS if assignment[n] == 8 else NVFP4_BITS)
                     for n in stats)
    achieved = total_cost / total_params
    return assignment, achieved


def build_pareto_curve(stats: dict, targets):
    """Run the allocator at multiple target bits and record (target, achieved,
    predicted_total_delta_kl, mxfp8_count)."""
    curve = []
    for t in targets:
        assignment, achieved = allocate_at_budget(stats, t)
        total_dkl = 0.0
        mxfp8_params = 0
        mxfp8_count = 0
        for name, s in stats.items():
            if assignment[name] == 8:
                total_dkl += predict_delta_kl(
                    s["h_trace"], s["w_max_abs"], 8.0, MXFP8_KMSE)
                mxfp8_params += s["n_params"]
                mxfp8_count += 1
            else:
                total_dkl += predict_delta_kl(
                    s["h_trace"], s["w_max_abs"], 4.0, NVFP4_KMSE)
        curve.append({
            "target_avg_bits": t,
            "achieved_avg_bits": achieved,
            "predicted_delta_kl": total_dkl,
            "mxfp8_layers": mxfp8_count,
            "mxfp8_params": mxfp8_params,
            "nvfp4_layers": len(stats) - mxfp8_count,
        })
    return curve


def write_layer_config(stats: dict, assignment: dict, output_json: Path):
    cfg = {}
    for name in stats:
        cfg[name] = mxfp8_scheme() if assignment[name] == 8 else nvfp4_scheme()
    with open(output_json, "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawq", required=True,
                    help="Pickle produced by streaming_hawq.py")
    ap.add_argument("--target-bits", type=float, default=4.75,
                    help="Target average bits/parameter")
    ap.add_argument("--layer-config", required=True,
                    help="Output AutoRound layer_config JSON (for --layer_config)")
    ap.add_argument("--pareto-csv", required=True,
                    help="Output Pareto curve CSV for plotting/inspection")
    ap.add_argument("--pareto-targets",
                    default="4.5,4.55,4.6,4.65,4.7,4.75,4.8,4.9,5.0,5.25,5.5,6.0,7.0,8.25",
                    help="Comma-separated target avg_bits values for Pareto")
    args = ap.parse_args()

    with open(args.hawq, "rb") as f:
        hawq = pickle.load(f)
    stats = hawq["stats"]
    print(f"Loaded HAWQ stats for {len(stats)} linears from {args.hawq}")
    print(f"Model: {hawq.get('meta', {}).get('model', '(unknown)')}")

    # Build Pareto curve
    pareto_targets = [float(x) for x in args.pareto_targets.split(",")]
    curve = build_pareto_curve(stats, pareto_targets)

    # Write Pareto CSV
    with open(args.pareto_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        w.writeheader()
        for row in curve:
            w.writerow(row)
    print(f"Pareto curve written to {args.pareto_csv}")
    # Also print inline
    print(f"\n{'target':>8} {'achieved':>9} {'ΔKL (pred)':>14} "
          f"{'MXFP8 lyrs':>11} {'MXFP8 params':>14} {'NVFP4 lyrs':>11}")
    for row in curve:
        print(f"{row['target_avg_bits']:>8.3f} {row['achieved_avg_bits']:>9.3f} "
              f"{row['predicted_delta_kl']:>14.6e} "
              f"{row['mxfp8_layers']:>11,} {row['mxfp8_params']:>14,} "
              f"{row['nvfp4_layers']:>11,}")

    # Produce the final layer_config at the chosen target
    assignment, achieved = allocate_at_budget(stats, args.target_bits)
    write_layer_config(stats, assignment, Path(args.layer_config))
    mx = sum(1 for v in assignment.values() if v == 8)
    nv = sum(1 for v in assignment.values() if v == 4)
    print(f"\nChosen recipe @ target={args.target_bits} → achieved={achieved:.3f}")
    print(f"  NVFP4 layers: {nv}")
    print(f"  MXFP8 layers: {mx}")
    print(f"Layer config: {args.layer_config}")


if __name__ == "__main__":
    main()
