#!/usr/bin/env python3
"""bakeoff.py — decide whether a DynaQuant change is worth keeping.

This script compares a candidate run against:
  - the additive baseline calibration
  - an optional interaction-refined recipe
  - an optional oracle local search on the same tiny problem

The goal is not just to print metrics, but to answer:
  "did the new method buy enough quality to justify keeping it?"
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass
class Point:
    label: str
    bits: float
    kl: float


def _load_calibration_point(path: str, selector: str) -> Point:
    with open(path) as f:
        data = json.load(f)
    if selector == "baseline":
        row = min(data["results"], key=lambda r: r["achieved_bits"])
    elif selector == "high":
        row = max(data["results"], key=lambda r: r["achieved_bits"])
    elif selector == "knee":
        rows = sorted(data["results"], key=lambda r: r["achieved_bits"])
        row = rows[1] if len(rows) >= 2 else rows[0]
    else:
        raise ValueError(selector)
    return Point(selector, float(row["achieved_bits"]), float(row["actual_last_token_kl"]))


def _load_refined_point(path: str, calibrated_kl: float) -> Point:
    with open(path) as f:
        data = json.load(f)
    return Point(
        "refined",
        float(data["bits_per_param"]),
        float(calibrated_kl + data["refined_delta_kl_estimate"]),
    )


def _load_oracle_best(path: str) -> Point:
    with open(path) as f:
        data = json.load(f)
    best = data["best"]
    return Point("oracle", float(best["bits_per_param"]), float(best["actual_last_token_kl"]))


def _summarize(candidate: Point, baseline: Point, oracle: Point | None):
    out = {
        "candidate": candidate.__dict__,
        "baseline": baseline.__dict__,
        "delta_kl_vs_baseline": candidate.kl - baseline.kl,
        "delta_bits_vs_baseline": candidate.bits - baseline.bits,
    }
    if oracle is not None:
        out["oracle"] = oracle.__dict__
        out["oracle_gap_abs"] = candidate.kl - oracle.kl
        out["oracle_gap_rel"] = (candidate.kl - oracle.kl) / max(abs(oracle.kl), 1e-12)
    return out


def _decision(summary: dict, max_kl_regression: float, min_kl_gain: float, max_oracle_gap: float | None):
    delta = summary["delta_kl_vs_baseline"]
    if delta > max_kl_regression:
        return "reject", f"KL regressed by {delta:.4e} (> {max_kl_regression:.4e})"
    if delta < -min_kl_gain:
        if max_oracle_gap is not None and "oracle_gap_abs" in summary and summary["oracle_gap_abs"] > max_oracle_gap:
            return "investigate", (
                f"improved vs baseline ({delta:.4e}) but still {summary['oracle_gap_abs']:.4e} "
                f"away from oracle (> {max_oracle_gap:.4e})"
            )
        return "keep", f"improved KL by {-delta:.4e}"
    if max_oracle_gap is not None and "oracle_gap_abs" in summary and summary["oracle_gap_abs"] <= max_oracle_gap:
        return "keep", f"near oracle gap ({summary['oracle_gap_abs']:.4e})"
    return "investigate", "change is neutral; needs broader justification"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--candidate", choices=["baseline", "knee", "high", "refined"], default="knee")
    ap.add_argument("--refined", help="quadratic_refine_allocator output when --candidate refined")
    ap.add_argument("--oracle", help="oracle_search output")
    ap.add_argument("--max-kl-regression", type=float, default=1e-3)
    ap.add_argument("--min-kl-gain", type=float, default=1e-3)
    ap.add_argument("--max-oracle-gap", type=float, default=5e-3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    baseline = _load_calibration_point(args.calibration, "baseline")
    knee = _load_calibration_point(args.calibration, "knee")
    if args.candidate == "refined":
        if not args.refined:
            raise SystemExit("--refined is required for candidate=refined")
        candidate = _load_refined_point(args.refined, knee.kl)
    else:
        candidate = _load_calibration_point(args.calibration, args.candidate)
    oracle = _load_oracle_best(args.oracle) if args.oracle else None

    summary = _summarize(candidate, baseline, oracle)
    decision, reason = _decision(
        summary,
        max_kl_regression=args.max_kl_regression,
        min_kl_gain=args.min_kl_gain,
        max_oracle_gap=args.max_oracle_gap if args.oracle else None,
    )
    summary["decision"] = decision
    summary["reason"] = reason

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
