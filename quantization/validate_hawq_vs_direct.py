#!/usr/bin/env python3
"""
Validate HAWQ-V3 sensitivity ranking against direct KL measurement.

For each Linear in the model, compare:
  - Direct: measured KL(L, b) from measure_bit_utility.py
  - HAWQ:   predicted sensitivity from measure_hawq_sensitivity.py

The HAWQ prediction for per-Linear KL at bit b is parametric:
    ΔKL(L, b) ≈ h_w2_sum_L   (bit-independent sensitivity scalar)
or
    ΔKL(L, b) ≈ h_trace_L · w_max² / (12 · (2^(b-1) - 1)²)  (bit-aware)

We compute Spearman rank correlation between HAWQ's per-Linear ranking
and direct measurement's per-Linear ranking at each bit level. If ρ > 0.8
consistently, HAWQ is the scaling path. If not, we fall back to direct
measurement or escalate to Hutchinson-Hessian.
"""
import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple


def spearman(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation, ties broken by average rank."""
    n = len(xs)
    if n < 3:
        return 0.0
    def ranks(vs):
        order = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        for pos, idx in enumerate(order):
            r[idx] = pos + 1
        return r
    rx = ranks(xs)
    ry = ranks(ys)
    mrx = sum(rx) / n
    mry = sum(ry) / n
    vx = sum((r - mrx) ** 2 for r in rx) / n
    vy = sum((r - mry) ** 2 for r in ry) / n
    cov = sum((rx[i] - mrx) * (ry[i] - mry) for i in range(n)) / n
    if vx * vy <= 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
    if vx * vy <= 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", required=True,
                        help="curves.json from measure_bit_utility.py")
    parser.add_argument("--hawq", required=True,
                        help="sensitivity.json from measure_hawq_sensitivity.py")
    args = parser.parse_args()

    with open(args.direct) as f:
        direct = json.load(f)
    with open(args.hawq) as f:
        hawq = json.load(f)

    print(f"direct: {direct['n_linears_measured']} linears, "
          f"{len(direct['bit_levels'])} bit levels, "
          f"{direct['elapsed_sec']:.0f}s")
    print(f"hawq:   {hawq['n_quantizable']} linears, {hawq['elapsed_sec']:.0f}s")

    common_names = set(direct["curves"].keys()) & set(hawq["sensitivity"].keys())
    print(f"common: {len(common_names)} linears")

    # Check whether h_w2_sum is available (older HAWQ outputs had it;
    # newer scalar-only outputs don't)
    sample_entry = next(iter(hawq["sensitivity"].values()))
    has_h_w2_sum = "h_w2_sum" in sample_entry

    if has_h_w2_sum:
        print("\n" + "=" * 70)
        print("Rank correlation: direct-measured KL vs HAWQ h_w2_sum (ranking)")
        print("=" * 70)
        print(f"{'bits':>5} {'spearman':>10} {'pearson':>10} {'mean_kl':>12}")
        print("-" * 45)
        for b_str in [str(b) for b in direct["bit_levels"]]:
            b = int(b_str)
            if b == 16:
                continue
            names = sorted(common_names)
            direct_kls = []
            hawq_scores = []
            for name in names:
                kl = direct["curves"][name]["kl_per_bits"].get(b_str)
                if kl is None:
                    continue
                direct_kls.append(kl)
                hawq_scores.append(hawq["sensitivity"][name]["h_w2_sum"])
            rho = spearman(direct_kls, hawq_scores)
            r = pearson(direct_kls, hawq_scores)
            mkl = sum(direct_kls) / len(direct_kls)
            print(f"{b:>5} {rho:>+10.3f} {r:>+10.3f} {mkl:>12.6f}")

    print("\n" + "=" * 70)
    print("Comparing sensitivity metrics (Spearman ρ only)")
    print("=" * 70)
    metrics = [
        ("h_trace",  lambda s: s["h_trace"]),
        ("h_trace·w_max²", lambda s: s["h_trace"] * s["w_max_abs"] ** 2),
        ("h_trace·mean(w²)", lambda s: s["h_trace"] * s["w_norm_sq"] / s["numel"]),
    ]
    if has_h_w2_sum:
        metrics.insert(0, ("h_w2_sum", lambda s: s["h_w2_sum"]))
    header = f"{'bits':>5} " + " ".join(f"{m[0]:>14}" for m in metrics)
    print(header)
    print("-" * len(header))
    for b_str in [str(b) for b in direct["bit_levels"]]:
        b = int(b_str)
        if b == 16:
            continue
        names = sorted(common_names)
        direct_kls = []
        scores_per_metric = {m[0]: [] for m in metrics}
        for name in names:
            kl = direct["curves"][name]["kl_per_bits"].get(b_str)
            if kl is None:
                continue
            direct_kls.append(kl)
            s = hawq["sensitivity"][name]
            for m_name, m_fn in metrics:
                scores_per_metric[m_name].append(m_fn(s))
        row = f"{b:>5} "
        for m_name, _ in metrics:
            rho = spearman(direct_kls, scores_per_metric[m_name])
            row += f" {rho:>+13.3f} "
        print(row)

    # Top-10 most sensitive by each metric — do the lists overlap?
    score_fn = (lambda n: hawq["sensitivity"][n]["h_w2_sum"]) if has_h_w2_sum else \
               (lambda n: hawq["sensitivity"][n]["h_trace"] *
                          hawq["sensitivity"][n]["w_norm_sq"] /
                          hawq["sensitivity"][n]["numel"])
    label = "h_w2_sum" if has_h_w2_sum else "h_trace·mean(w²)"
    print("\n" + "=" * 70)
    print(f"Top-10 most sensitive linears: direct @4 bits vs HAWQ {label}")
    print("=" * 70)
    by_direct_4 = sorted(
        common_names,
        key=lambda n: -direct["curves"][n]["kl_per_bits"].get("4", 0)
    )[:10]
    by_hawq = sorted(common_names, key=lambda n: -score_fn(n))[:10]
    print(f"{'rank':>5}  {'direct@4bits':<45}  {'hawq h_w2_sum':<45}")
    print("-" * 100)
    for i in range(10):
        d = by_direct_4[i] if i < len(by_direct_4) else ""
        h = by_hawq[i] if i < len(by_hawq) else ""
        overlap = "✓" if d in set(by_hawq) else "."
        print(f"{i+1:>5} {overlap} {d[:43]:<45}  {h[:43]:<45}")

    # Jaccard of top-10
    top_d = set(by_direct_4)
    top_h = set(by_hawq)
    jaccard = len(top_d & top_h) / len(top_d | top_h)
    print(f"\ntop-10 Jaccard overlap: {jaccard:.2f}")


if __name__ == "__main__":
    main()
