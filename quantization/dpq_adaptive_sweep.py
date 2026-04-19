#!/usr/bin/env python3
"""
Adaptive Pareto sweep: bisect around the inflection point instead of
grid-sweeping the full curve.

Algorithm (walk-and-bisect around the knee):

  1. Run 3 anchor efficiency values: {2.0, 0.5, 0.1}
  2. Compute slopes (ΔKL / Δcost) between adjacent points
  3. Identify the "knee interval" — the pair with the steepest slope
  4. Bisect that interval (add a midpoint)
  5. Recompute Kneedle; if knee label changed, bisect the new knee's
     steepest side
  6. Stop when:
     - knee label is stable for 2 consecutive iterations, OR
     - we've used max_points runs, OR
     - slope on both sides of the knee is below min_slope_delta

  Each point reuses the same AutoRound cache, so only the DPQ stage
  runs per iteration (~1-15 min depending on model size).

Usage:
    python3 dpq_adaptive_sweep.py \\
        --model /models/Qwen3.5-27B-bf16 \\
        --cache-dir /tmp/dpq_cache/qwen35-27b-nvfp4-noR \\
        --output-base /tmp/qwen35-27b-adaptive \\
        --max-points 8
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


def find_knee_idx(points: List[Tuple[float, float]]) -> int:
    """Kneedle: return index of the point with max perpendicular distance
    from the chord connecting the first and last points. Points are
    assumed sorted by cost ascending."""
    if len(points) < 3:
        return len(points) - 1
    costs = [p[0] for p in points]
    kls = [p[1] for p in points]
    c_min, c_max = min(costs), max(costs)
    k_min, k_max = min(kls), max(kls)
    c_range = c_max - c_min or 1.0
    k_range = k_max - k_min or 1.0
    norm = [((c - c_min) / c_range, (k - k_min) / k_range) for c, k in zip(costs, kls)]
    x1, y1 = norm[0]
    x2, y2 = norm[-1]
    denom = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) or 1.0
    best_idx, best_dist = 0, -1.0
    for i, (x, y) in enumerate(norm):
        d = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
        if d > best_dist:
            best_dist, best_idx = d, i
    return best_idx


def compute_slopes(points: List[Tuple[float, float]]) -> List[float]:
    """Return slopes (ΔKL / Δcost) between adjacent points. Negative means
    quality improves with cost (which is the expected direction)."""
    slopes = []
    for i in range(len(points) - 1):
        dc = points[i + 1][0] - points[i][0]
        dk = points[i + 1][1] - points[i][1]
        slopes.append(dk / dc if dc else 0.0)
    return slopes


def run_dpq_point(
    eff: float,
    *,
    model: str,
    cache_dir: str,
    output_base: str,
    extra_args: List[str],
    dry_run: bool = False,
) -> dict:
    """Run a single DPQ efficiency point and return its manifest."""
    label = f"eff{eff:05.2f}"
    outdir = Path(output_base) / label
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "dpq_autoround_first.py",
        "--model", model,
        "--output", str(outdir),
        "--cache-dir", cache_dir,
        "--min-efficiency", str(eff),
        "--no-hadamard",
    ] + extra_args
    print(f"[adaptive] running {label}: {' '.join(cmd)}", flush=True)
    if dry_run:
        return {"min_efficiency": eff, "avg_cost_vs_fp4": eff, "final_kl": 1.0 / eff}
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[adaptive] FAILED {label}: {r.stderr[-500:]}", flush=True)
        raise SystemExit(f"DPQ run failed at eff={eff}")
    manifest_path = outdir / "dpq_autoround_first_manifest.json"
    with open(manifest_path) as f:
        m = json.load(f)
    print(f"[adaptive]   {label}: cost={m['avg_cost_vs_fp4']:.3f} KL={m['final_kl']:.5f} "
          f"counts={m['counts']} ({dt:.0f}s)", flush=True)
    return m


def adaptive_sweep(
    model: str,
    cache_dir: str,
    output_base: str,
    extra_args: List[str],
    *,
    initial_effs: List[float] = (2.0, 0.5, 0.1),
    max_points: int = 8,
    min_knee_move_threshold: float = 0.05,
    dry_run: bool = False,
) -> dict:
    """Run adaptive Pareto sweep. Returns the final knee + all points."""
    # pts is a dict keyed on efficiency so we don't re-run the same point
    pts: dict = {}
    for eff in initial_effs:
        m = run_dpq_point(eff, model=model, cache_dir=cache_dir,
                          output_base=output_base, extra_args=extra_args, dry_run=dry_run)
        pts[eff] = m

    def sorted_pts():
        return sorted(pts.values(), key=lambda m: m["avg_cost_vs_fp4"])

    iteration = 0
    last_knee_eff = None
    stable_iters = 0
    while len(pts) < max_points:
        iteration += 1
        rows = sorted_pts()
        curve = [(r["avg_cost_vs_fp4"], r["final_kl"]) for r in rows]
        knee_idx = find_knee_idx(curve)
        knee_row = rows[knee_idx]
        knee_eff = knee_row["min_efficiency"]

        slopes = compute_slopes(curve)
        print(f"\n[adaptive] iter {iteration}: {len(pts)} points, "
              f"knee at eff={knee_eff} (cost={curve[knee_idx][0]:.3f}, "
              f"KL={curve[knee_idx][1]:.5f})", flush=True)
        print(f"[adaptive]   slopes: "
              + ", ".join(f"{s:.4f}" for s in slopes), flush=True)

        # Check stability
        if last_knee_eff is not None and abs(knee_eff - last_knee_eff) < min_knee_move_threshold:
            stable_iters += 1
            if stable_iters >= 2:
                print(f"[adaptive] knee stable for 2 iterations — stopping", flush=True)
                break
        else:
            stable_iters = 0
        last_knee_eff = knee_eff

        # Pick the interval with the steepest (most negative) slope, which
        # is where the curve is bending fastest. Bisect there.
        # First try the interval containing the current knee; if both sides
        # of the knee are already populated with fine-grained points, walk
        # to the STEEPEST side and bisect there.
        steepest_idx = min(range(len(slopes)), key=lambda i: slopes[i])
        left_cost, _ = curve[steepest_idx]
        right_cost, _ = curve[steepest_idx + 1]
        # Bisect via EFFICIENCY (because that's what we control), finding
        # the eff halfway between the two anchor points' effs.
        left_eff = rows[steepest_idx]["min_efficiency"]
        right_eff = rows[steepest_idx + 1]["min_efficiency"]
        # Geometric midpoint feels more natural for efficiency (log-space)
        mid_eff = round(math.sqrt(left_eff * right_eff), 3)
        if mid_eff in pts:
            # already have this — try arithmetic mid
            mid_eff = round((left_eff + right_eff) / 2, 3)
            if mid_eff in pts:
                print(f"[adaptive] bisection target already exists, stopping", flush=True)
                break
        print(f"[adaptive]   bisecting steepest interval "
              f"(eff {left_eff} → {right_eff}) at {mid_eff}", flush=True)
        m = run_dpq_point(mid_eff, model=model, cache_dir=cache_dir,
                          output_base=output_base, extra_args=extra_args, dry_run=dry_run)
        pts[mid_eff] = m

    # Final summary
    rows = sorted_pts()
    curve = [(r["avg_cost_vs_fp4"], r["final_kl"]) for r in rows]
    knee_idx = find_knee_idx(curve)

    print("\n" + "=" * 70)
    print(f"ADAPTIVE SWEEP SUMMARY ({len(pts)} points)")
    print("=" * 70)
    print(f"{'eff':>6} {'cost':>7} {'KL':>10} {'gap':>7} {'fp4/fp8/bf16':>16}  knee")
    print("-" * 70)
    for i, r in enumerate(rows):
        c = r['counts']
        marker = "  <--" if i == knee_idx else ""
        print(f"{r['min_efficiency']:>6.2f} {r['avg_cost_vs_fp4']:>7.3f} {r['final_kl']:>10.5f} "
              f"{r['gap_closure']:>7.3f}  {c['fp4']:>3}/{c['fp8']:>3}/{c['bf16']:>3}{marker}")
    print(f"\nknee: eff={rows[knee_idx]['min_efficiency']}  "
          f"cost={curve[knee_idx][0]:.3f}  KL={curve[knee_idx][1]:.5f}")

    return {
        "points": rows,
        "knee_idx": knee_idx,
        "knee_eff": rows[knee_idx]["min_efficiency"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--max-points", type=int, default=8)
    parser.add_argument("--initial-effs", type=float, nargs="+", default=[2.0, 0.5, 0.1])
    parser.add_argument("--autoround-iters", type=int, default=200)
    parser.add_argument("--autoround-nsamples", type=int, default=128)
    parser.add_argument("--autoround-seqlen", type=int, default=2048)
    parser.add_argument("--autoround-batch-size", type=int, default=4)
    parser.add_argument("--autoround-dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--dpq-steps", type=int, default=150)
    parser.add_argument("--dpq-calib-samples", type=int, default=16)
    parser.add_argument("--dpq-calib-seqlen", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    extra = [
        "--autoround-iters", str(args.autoround_iters),
        "--autoround-nsamples", str(args.autoround_nsamples),
        "--autoround-seqlen", str(args.autoround_seqlen),
        "--autoround-batch-size", str(args.autoround_batch_size),
        "--autoround-dataset", args.autoround_dataset,
        "--dpq-steps", str(args.dpq_steps),
        "--dpq-calib-samples", str(args.dpq_calib_samples),
        "--dpq-calib-seqlen", str(args.dpq_calib_seqlen),
    ]
    adaptive_sweep(
        model=args.model,
        cache_dir=args.cache_dir,
        output_base=args.output_base,
        extra_args=extra,
        initial_effs=args.initial_effs,
        max_points=args.max_points,
        dry_run=args.dry_run,
    )
