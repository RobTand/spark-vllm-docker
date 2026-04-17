#!/usr/bin/env python3
"""allocator.py — multi-choice knapsack mixed-precision assignment.

Given:
  - per-Linear sensitivity (Fisher trace from sensitivity_probe.py)
  - per-(Linear, format) measured quantization cost (from measure_quant_cost.py)
  - a bit budget (target average bits per parameter)
  - a format registry (any subset of registered formats)

Solve for a per-Linear format assignment that minimizes total predicted
loss increase subject to the bit budget.

Predicted loss increase per layer, per format, under Gauss-Newton/Fisher:
    Δloss_{ℓ,f} ≈ 0.5 · H_trace_ℓ · output_mse_{ℓ,f} · out_features_ℓ

where output_mse is the measured RTN functional error normalized by
numel(W·X). H_trace is the route-aware Fisher trace for MoE experts
(divided by route_prob during the probe), so sparse experts' scores
are comparable to dense layers'.

Solver:
  Multi-choice knapsack via DP with bit-budget discretization (we round
  bit costs to 0.01-bit bins, making the budget an integer). For 35B
  with ~300 Linears × 8 formats × 400 budget bins, runtime is under 1s.

Fused-projection siblings (q/k/v/o, gate/up, ...) are post-processed:
  all siblings promoted to the highest format chosen for any of them,
  to match vLLM's fused-tensor loader constraints.

Auto-Pareto knee via Kneedle (Satopää et al.). Reports the knee target
plus a few flanking points so you can eyeball.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import format_registry as fr


# ---------------------------------------------------------------------------
# Fused-projection sibling detection (general)
# ---------------------------------------------------------------------------
_FUSED_PATTERNS = [
    # Standard HF self-attention
    (r"^(?P<pre>.+)\.self_attn\.(?P<sib>q_proj|k_proj|v_proj|o_proj)$",
     ("q_proj", "k_proj", "v_proj"), "self_attn"),
    # Standard HF MLP
    (r"^(?P<pre>.+)\.mlp\.(?P<sib>gate_proj|up_proj)$",
     ("gate_proj", "up_proj"), "mlp"),
    # Legacy Mixtral/Mistral
    (r"^(?P<pre>.+)\.(?P<sib>w1|w3)$",
     ("w1", "w3"), "w1w3"),
    # Legacy PaLM-style
    (r"^(?P<pre>.+)\.(?P<sib>gate|up)_proj$",
     ("gate_proj", "up_proj"), "gate_up_alt"),
]


def fused_siblings(name: str) -> tuple[tuple[str, ...], str] | None:
    for pat, members, kind in _FUSED_PATTERNS:
        m = re.search(pat, name)
        if m:
            pre = m.group("pre")
            parent = name[: name.rindex(m.group("sib"))]
            # Reconstruct all sibling names that should be promoted together
            sibs = tuple(f"{parent}{s}" for s in members)
            return sibs, kind
    return None


def promote_fused(assignment: dict[str, str],
                  format_rank: dict[str, int]) -> dict[str, str]:
    """After per-Linear selection, bump each fused group's siblings to the
    highest-rank format picked for any group member."""
    out = dict(assignment)
    groups: dict[tuple, list[str]] = defaultdict(list)
    for name in assignment:
        sib = fused_siblings(name)
        if sib is not None:
            sibs, kind = sib
            groups[sibs].append(name)
    for sibs, members_present in groups.items():
        # Only consider siblings we actually have allocations for
        ranks = [format_rank[out[m]] for m in members_present]
        best = max(ranks)
        best_fmt = next(k for k, v in format_rank.items() if v == best)
        for m in members_present:
            if format_rank[out[m]] < best:
                out[m] = best_fmt
    return out


# ---------------------------------------------------------------------------
# Kneedle knee detection
# ---------------------------------------------------------------------------
def kneedle(x: list[float], y: list[float]) -> int:
    """Return index of the knee in a convex-decreasing curve."""
    if len(x) < 3:
        return 0
    xs = [xi for xi in x]
    ys = [yi for yi in y]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin or ymax == ymin:
        return 0
    x_norm = [(xi - xmin) / (xmax - xmin) for xi in xs]
    y_norm = [(yi - ymin) / (ymax - ymin) for yi in ys]
    # For a convex-decreasing curve, the knee is the point with max
    # distance below the chord from (0,1) to (1,0).
    diffs = [yn - (1.0 - xn) for xn, yn in zip(x_norm, y_norm)]
    # Convex-decreasing, so we want the most-negative diff (max dip).
    return min(range(len(diffs)), key=lambda i: diffs[i])


# ---------------------------------------------------------------------------
# Multi-choice knapsack DP
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    fmt: str
    bits_per_param: float
    predicted_dloss: float


def build_candidates(stats: dict, costs: dict, formats: list[fr.FormatSpec]
                     ) -> dict[str, list[Candidate]]:
    """For each Linear, build its candidate list (one per format)."""
    out: dict[str, list[Candidate]] = {}
    for name, s in stats.items():
        if name not in costs:
            continue
        h_trace = s["h_trace"]
        d_out = s["out_features"]
        cands = []
        for spec in formats:
            entry = costs[name].get(spec.name)
            if entry is None or "error" in entry:
                continue
            # Predicted Δloss under Gauss-Newton approximation
            #   Δloss ≈ 0.5 · sensitivity · perturbation
            # where sensitivity (h_trace) is a second-derivative magnitude
            # and perturbation is scaled by dim of the layer output.
            predicted = 0.5 * h_trace * entry["output_mse"] * d_out
            cands.append(Candidate(
                fmt=spec.name,
                bits_per_param=spec.effective_bits,
                predicted_dloss=max(predicted, 0.0),
            ))
        if cands:
            out[name] = cands
    return out


def solve_allocation(stats: dict, candidates: dict[str, list[Candidate]],
                     target_bits: float, bit_precision: float = 0.001
                     ) -> dict[str, str] | None:
    """Solve multi-choice knapsack via DP, working in avg-bits-per-param units.

    The budget is expressed as an average bits-per-parameter target; we
    discretize (target - baseline) into bins of `bit_precision`. Each
    layer's cost is its contribution to the weighted average, which for
    a layer with fraction f = params/total of the total is
        Δavg = (c.bits_per_param - baseline.bits_per_param) · f.
    Total DP budget ~= (target - baseline) / bit_precision, typically
    under 10 000 bins regardless of model size.

    Returns {linear_name: chosen_format_name}, or None if infeasible.
    """
    import numpy as np

    names = list(candidates.keys())
    total_params = sum(stats[n]["n_params"] for n in names)
    if total_params == 0:
        return {}

    baselines = {n: min(cs, key=lambda c: c.bits_per_param)
                 for n, cs in candidates.items()}
    min_bits = sum(baselines[n].bits_per_param * stats[n]["n_params"]
                   for n in names) / total_params

    if target_bits < min_bits - 1e-6:
        return None

    # Budget in bits-per-param units, so the bin count is independent of
    # model size. For a 35B model with 0.001 bit precision this gives
    # ~5000 bins at a 5.0-bit target, trivially small.
    excess = target_bits - min_bits
    n_bins = int(round(excess / bit_precision)) + 2

    # Per-layer: pre-compute (dbins, dgain, cand_idx) option list.
    # dbins is layer's contribution to the avg-bits-per-param budget,
    # scaled into integer bins.
    INF_NEG = -1e30
    dp = np.full(n_bins, INF_NEG, dtype=np.float64)
    dp[0] = 0.0
    choice: list[np.ndarray] = []

    for name in names:
        baseline = baselines[name]
        cs = candidates[name]
        params = stats[name]["n_params"]
        fraction = params / total_params
        baseline_loss = baseline.predicted_dloss
        options = []
        for idx, c in enumerate(cs):
            d_avg_bits = (c.bits_per_param - baseline.bits_per_param) * fraction
            dbins = int(round(d_avg_bits / bit_precision))
            if dbins < 0 or dbins >= n_bins:
                continue
            dgain = baseline_loss - c.predicted_dloss
            options.append((dbins, dgain, idx))
        if not options:
            options = [(0, 0.0, cs.index(baseline))]

        # Convert to arrays for fast inner loop
        opt_dbins = np.asarray([o[0] for o in options], dtype=np.int32)
        opt_dgain = np.asarray([o[1] for o in options], dtype=np.float64)
        opt_idx = np.asarray([o[2] for o in options], dtype=np.int32)

        new_dp = np.full(n_bins, INF_NEG, dtype=np.float64)
        new_choice = np.full(n_bins, -1, dtype=np.int32)

        # Vectorized update: for each option, add (dbins, dgain) to dp
        for db, dg, idx in zip(opt_dbins, opt_dgain, opt_idx):
            if db == 0:
                candidate_vals = dp + dg
                target_slice = new_dp
                mask = candidate_vals > target_slice
                new_dp = np.where(mask, candidate_vals, new_dp)
                new_choice = np.where(mask, idx, new_choice)
            else:
                candidate_vals = dp[:-db] + dg
                target_slice = new_dp[db:]
                mask = candidate_vals > target_slice
                target_slice[:] = np.where(mask, candidate_vals, target_slice)
                new_choice[db:] = np.where(mask, idx, new_choice[db:])
        dp = new_dp
        choice.append(new_choice)

    if not np.isfinite(dp).any() or dp.max() == INF_NEG:
        return None
    best_b = int(np.argmax(dp))

    # Backtrack
    assignment = {}
    cur = best_b
    for layer_idx in range(len(names) - 1, -1, -1):
        idx_chosen = int(choice[layer_idx][cur])
        name = names[layer_idx]
        cs = candidates[name]
        if idx_chosen < 0:
            idx_chosen = 0
        assignment[name] = cs[idx_chosen].fmt
        baseline = baselines[name]
        params = stats[name]["n_params"]
        fraction = params / total_params
        d_avg_bits = (cs[idx_chosen].bits_per_param
                      - baseline.bits_per_param) * fraction
        cur -= int(round(d_avg_bits / bit_precision))
        if cur < 0:
            cur = 0
    return assignment


def compute_achieved(stats: dict, assignment: dict[str, str],
                     format_specs: dict[str, fr.FormatSpec]) -> tuple[float, float]:
    """Return (avg_bits, total_predicted_dloss)."""
    total_params = sum(stats[n]["n_params"] for n in assignment)
    total_bits = sum(format_specs[assignment[n]].effective_bits
                     * stats[n]["n_params"] for n in assignment)
    return total_bits / max(total_params, 1), 0.0  # dloss recomputed separately


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="sensitivity_probe pickle")
    ap.add_argument("--costs", required=True, help="measure_quant_cost pickle")
    ap.add_argument("--target-bits", type=float, default=4.75)
    ap.add_argument("--formats", default="",
                    help="Comma-separated format names to consider; empty=all")
    ap.add_argument("--pareto-targets",
                    default="4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25",
                    help="Comma-separated budgets to sweep for Pareto curve")
    ap.add_argument("--layer-config", required=True,
                    help="Output AutoRound layer_config JSON")
    ap.add_argument("--pareto-csv", required=True, help="Output Pareto CSV")
    ap.add_argument("--no-fused-promote", action="store_true",
                    help="Skip fused-projection sibling promotion")
    ap.add_argument("--bit-precision", type=float, default=0.001,
                    help="Knapsack bit-bin granularity in avg-bits/param "
                         "(smaller = slower; default 0.001 → ~5000 bins)")
    ap.add_argument("--threads", type=int, default=0,
                    help="OMP/numpy threads for DP (0 = default)")
    args = ap.parse_args()

    if args.threads > 0:
        import os
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    with open(args.probe, "rb") as f:
        probe = pickle.load(f)
    with open(args.costs, "rb") as f:
        cost_data = pickle.load(f)
    stats = probe["stats"]
    costs = cost_data["costs"]
    print(f"[alloc] stats: {len(stats)} Linears, costs: {len(costs)} Linears")

    if args.formats:
        fmt_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    else:
        fmt_names = cost_data["formats"]
    specs = [fr.get_format(n) for n in fmt_names]
    specs_sorted = sorted(specs, key=lambda s: s.effective_bits)
    format_rank = {s.name: i for i, s in enumerate(specs_sorted)}
    format_specs = {s.name: s for s in specs}
    print(f"[alloc] formats (low→high bits): "
          f"{[f'{s.name}({s.effective_bits:.2f}b)' for s in specs_sorted]}")

    candidates = build_candidates(stats, costs, specs_sorted)
    print(f"[alloc] candidates built for {len(candidates)} Linears")

    # Pareto sweep
    targets = [float(x) for x in args.pareto_targets.split(",")]
    curve = []
    for t in targets:
        assignment = solve_allocation(stats, candidates, t, args.bit_precision)
        if assignment is None:
            curve.append({"target_bits": t, "feasible": False})
            continue
        if not args.no_fused_promote:
            assignment = promote_fused(assignment, format_rank)
        achieved, _ = compute_achieved(stats, assignment, format_specs)
        total_dloss = 0.0
        format_counts = defaultdict(int)
        format_params = defaultdict(int)
        for name, fmt in assignment.items():
            entry = costs[name].get(fmt, {})
            d_out = stats[name]["out_features"]
            total_dloss += 0.5 * stats[name]["h_trace"] * entry.get(
                "output_mse", 0.0) * d_out
            format_counts[fmt] += 1
            format_params[fmt] += stats[name]["n_params"]
        curve.append({
            "target_bits": t,
            "feasible": True,
            "achieved_bits": achieved,
            "predicted_dloss": total_dloss,
            **{f"layers_{k}": v for k, v in format_counts.items()},
            **{f"params_{k}": v for k, v in format_params.items()},
        })

    # Output Pareto CSV
    keys = sorted({k for row in curve for k in row.keys()})
    with open(args.pareto_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in curve:
            w.writerow(row)
    print(f"[alloc] Pareto curve → {args.pareto_csv}")

    # Kneedle
    feasible = [row for row in curve if row.get("feasible")]
    if len(feasible) >= 3:
        kidx = kneedle([r["achieved_bits"] for r in feasible],
                       [r["predicted_dloss"] for r in feasible])
        knee = feasible[kidx]
        print(f"[alloc] suggested knee: target={knee['target_bits']}, "
              f"achieved={knee['achieved_bits']:.3f}, "
              f"Δloss={knee['predicted_dloss']:.3e}")

    # Print table
    print("\n  target  achieved     Δloss (pred)   " + "   ".join(
        f"{s.name[:11]:>11}" for s in specs_sorted))
    for row in curve:
        if not row.get("feasible"):
            print(f"  {row['target_bits']:>6.3f}  INFEASIBLE")
            continue
        fmt_str = "   ".join(
            f"{row.get(f'layers_{s.name}', 0):>11,}" for s in specs_sorted)
        print(f"  {row['target_bits']:>6.3f}  {row['achieved_bits']:>7.3f}  "
              f"{row['predicted_dloss']:>14.4e}   {fmt_str}")

    # Emit chosen layer_config for target_bits
    assignment = solve_allocation(stats, candidates, args.target_bits,
                                   args.bit_precision)
    if assignment is None:
        raise SystemExit(
            f"Infeasible at target_bits={args.target_bits}. "
            "Consider raising the target or widening the format set.")
    if not args.no_fused_promote:
        assignment = promote_fused(assignment, format_rank)
    achieved, _ = compute_achieved(stats, assignment, format_specs)

    layer_cfg = {}
    for name, fmt in assignment.items():
        layer_cfg[name] = format_specs[fmt].autoround_config()

    out = Path(args.layer_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(layer_cfg, f, indent=2)

    counts = defaultdict(int)
    for fmt in assignment.values():
        counts[fmt] += 1
    print(f"\n[alloc] target={args.target_bits} achieved={achieved:.3f}")
    for fmt, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {fmt:>14}: {n:>5} layers")
    print(f"\nLayer config → {out}")
    print(f"Feed to AutoRound via --layer_config {out}")


if __name__ == "__main__":
    main()
