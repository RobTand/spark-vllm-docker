#!/usr/bin/env python3
"""
DPQ Pareto analysis: collect (cost, quality) points from a sweep of
dpq_autoround_first runs and plot the Pareto curve.

Reads each run's dpq_autoround_first_manifest.json for the avg cost and
counts, runs wikitext PPL on each model directory to get a real quality
number, combines everything into a single (cost, ppl) curve, and identifies
the "knee" — the point where marginal quality-per-cost starts dropping fast.

Usage:
    python3 dpq_pareto.py \\
        --bf16 /path/to/bf16/source \\
        --runs \\
            permissive=/tmp/qwen25-1.5b-af-e0.25 \\
            balanced=/tmp/qwen25-1.5b-af-e0.5 \\
            strict=/tmp/qwen25-1.5b-af-e1.0 \\
        --out /tmp/pareto.png
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# PPL measurement (same sliding-window approach as dpq_wikitext_ppl.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_wikitext_ppl(
    model_path: str,
    max_length: int = 2048,
    stride: int = 1024,
    max_tokens: int = 16384,
) -> float:
    """Standard sliding-window wikitext-2 PPL."""
    from datasets import load_dataset

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = next(model.parameters()).device

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    enc = tokenizer(text, return_tensors="pt").input_ids.to(device)
    seq_len = min(enc.size(1), max_tokens)
    enc = enc[:, :seq_len]

    nlls = []
    n_tokens_seen = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        window = enc[:, begin:end]
        target_ids = window.clone()
        target_ids[:, :-trg_len] = -100
        out = model(window, labels=target_ids)
        nll = out.loss * max(trg_len - 1, 1)
        nlls.append(nll.item())
        n_tokens_seen += trg_len
        prev_end = end
        if end == seq_len:
            break

    ppl = math.exp(sum(nlls) / max(1, n_tokens_seen))

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return ppl


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------

def find_knee(points: List[tuple]) -> int:
    """
    Find the knee (maximum curvature) point on a Pareto curve.

    Input is a list of (cost, ppl) pairs, assumed sorted by cost ascending.
    Returns the INDEX of the knee point.

    The knee is the point where the marginal quality-per-cost starts
    dropping fastest — i.e., where you stop getting good returns on
    additional cost. We compute it via the "Kneedle" algorithm: find
    the point with the largest perpendicular distance to the line
    connecting the first and last points.
    """
    if len(points) < 3:
        return len(points) - 1

    costs = [p[0] for p in points]
    ppls = [p[1] for p in points]

    # Normalize to [0, 1] on both axes so the perpendicular distance is
    # meaningful regardless of units.
    c_min, c_max = min(costs), max(costs)
    p_min, p_max = min(ppls), max(ppls)
    c_range = c_max - c_min or 1.0
    p_range = p_max - p_min or 1.0
    norm = [((c - c_min) / c_range, (p - p_min) / p_range) for c, p in points]

    # Line from (0,1) to (1,0) — this is "high cost → low PPL" direction,
    # assuming the curve is roughly monotone: more cost → lower PPL.
    # The knee is the point furthest from this line in the "too much cost
    # for too little gain" direction.
    # Perpendicular distance from (x, y) to the line from (x1,y1) to (x2,y2):
    #   d = |(y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
    x1, y1 = norm[0]
    x2, y2 = norm[-1]
    denom = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) or 1.0

    best_idx, best_dist = 0, -1.0
    for i, (x, y) in enumerate(norm):
        d = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
        if d > best_dist:
            best_dist, best_idx = d, i
    return best_idx


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_pareto(
    points: List[tuple], labels: List[str], bf16_ppl: float, out_path: str,
    knee_idx: int = None,
    title: str = "DPQ Pareto curve (quality vs cost)",
):
    """Plot a Pareto curve. Each point is (cost, PPL)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    costs = [p[0] for p in points]
    ppls = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(8, 6))

    # The curve itself
    ax.plot(costs, ppls, "o-", color="steelblue", linewidth=2,
            markersize=10, label="DPQ recipes")

    # BF16 reference line
    ax.axhline(bf16_ppl, color="green", linestyle="--", linewidth=1.5,
               label=f"BF16 ceiling (ppl={bf16_ppl:.3f})")

    # Labels for each point
    for i, (c, p, lab) in enumerate(zip(costs, ppls, labels)):
        offset = (10, 8) if i % 2 == 0 else (10, -15)
        ax.annotate(
            f"{lab}\nc={c:.2f} ppl={p:.3f}",
            (c, p), textcoords="offset points", xytext=offset,
            fontsize=9, ha="left",
        )

    # Highlight the knee
    if knee_idx is not None and 0 <= knee_idx < len(points):
        kc, kp = points[knee_idx]
        ax.scatter([kc], [kp], s=300, edgecolors="crimson",
                   facecolors="none", linewidths=2.5,
                   label=f"knee: {labels[knee_idx]}", zorder=5)

    ax.set_xlabel("Average cost (× FP4)")
    ax.set_ylabel("wikitext-2 perplexity")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[pareto] saved plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", required=True, help="BF16 reference model path")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="label=path pairs, e.g. permissive=/tmp/run1")
    parser.add_argument("--out", default="/tmp/pareto.png",
                        help="output plot path (PNG)")
    parser.add_argument("--save-json", default="/tmp/pareto.json",
                        help="save the collected data as JSON")
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()

    # Parse runs
    runs = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"--runs entries must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        runs.append((label, path))

    # Measure BF16 reference PPL
    print(f"[pareto] measuring BF16 reference PPL: {args.bf16}", flush=True)
    bf16_ppl = measure_wikitext_ppl(args.bf16, max_tokens=args.max_tokens)
    print(f"[pareto]   bf16 ppl = {bf16_ppl:.4f}")

    # For each run, collect cost from manifest + PPL from model
    points = []
    manifests = []
    for label, path in runs:
        manifest_path = Path(path) / "dpq_autoround_first_manifest.json"
        if not manifest_path.exists():
            print(f"[pareto] WARNING: no manifest at {manifest_path}, skipping")
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        cost = manifest["avg_cost_vs_fp4"]
        counts = manifest["counts"]

        print(f"\n[pareto] === {label} ({path}) ===")
        print(f"[pareto]   cost={cost:.3f}, counts={counts}")
        print(f"[pareto]   measuring PPL...", flush=True)
        ppl = measure_wikitext_ppl(path, max_tokens=args.max_tokens)
        print(f"[pareto]   ppl = {ppl:.4f}  (vs bf16: {(ppl - bf16_ppl) / bf16_ppl * 100:+.2f}%)")

        points.append((cost, ppl))
        manifests.append({
            "label": label,
            "path": path,
            "cost": cost,
            "ppl": ppl,
            "counts": counts,
            "min_efficiency": manifest.get("min_efficiency"),
            "rtn_fp4_baseline_kl": manifest.get("rtn_fp4_baseline_kl"),
            "autoround_fp4_kl": manifest.get("autoround_fp4_kl_vs_bf16"),
            "autoround_fp8_kl": manifest.get("autoround_fp8_kl_vs_bf16"),
        })

    if len(points) == 0:
        print("[pareto] no valid runs found, exiting")
        return

    # Sort by cost ascending
    sorted_idx = sorted(range(len(points)), key=lambda i: points[i][0])
    points = [points[i] for i in sorted_idx]
    manifests = [manifests[i] for i in sorted_idx]
    labels = [m["label"] for m in manifests]

    # Find the knee
    knee_idx = find_knee(points) if len(points) >= 3 else None

    # Print summary table
    print("\n" + "=" * 70)
    print("PARETO CURVE SUMMARY")
    print("=" * 70)
    print(f"{'label':<15} {'cost':>8} {'ppl':>10} {'+% bf16':>10}  {'counts'}")
    print("-" * 70)
    for i, (cost, ppl) in enumerate(points):
        counts = manifests[i]["counts"]
        delta = (ppl - bf16_ppl) / bf16_ppl * 100
        marker = "  ← KNEE" if i == knee_idx else ""
        print(f"{labels[i]:<15} {cost:>8.3f} {ppl:>10.4f} {delta:>+9.2f}%  {counts}{marker}")
    print(f"\nbf16 reference ppl = {bf16_ppl:.4f}")
    if knee_idx is not None:
        print(f"knee is at: {labels[knee_idx]} (cost={points[knee_idx][0]:.2f}, "
              f"ppl={points[knee_idx][1]:.4f})")

    # Plot
    plot_pareto(points, labels, bf16_ppl, args.out, knee_idx=knee_idx,
                title=f"DPQ Pareto — wikitext-2 ppl vs cost")

    # Save JSON
    with open(args.save_json, "w") as f:
        json.dump({
            "bf16_ppl": bf16_ppl,
            "runs": manifests,
            "knee_label": labels[knee_idx] if knee_idx is not None else None,
        }, f, indent=2)
    print(f"[pareto] saved data to {args.save_json}")


if __name__ == "__main__":
    main()
