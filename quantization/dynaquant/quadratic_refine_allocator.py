#!/usr/bin/env python3
"""quadratic_refine_allocator.py — sparse interaction-aware knee refinement."""
from __future__ import annotations

import argparse
import json

from .interaction_refine import RefinementUnit, UnitOption, expand_unit_assignment, sparse_local_refine


def _load_units(payload: dict):
    units = []
    allowed = {}
    for row in payload["selected_units"]:
        options = tuple(
            UnitOption(
                fmt=opt["fmt"],
                bits_total=float(opt["bits_total"]),
                predicted_dloss=float(opt["predicted_dloss"]),
            )
            for opt in row["options"]
        )
        unit = RefinementUnit(
            key=row["key"],
            members=tuple(row["members"]),
            base_fmt=row["base_fmt"],
            options=options,
        )
        units.append(unit)
        allowed[unit.key] = tuple(opt for opt, raw in zip(options, row["options"]) if raw.get("allowed", True))
    return units, allowed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-passes", type=int, default=8)
    args = ap.parse_args()

    with open(args.interactions) as f:
        payload = json.load(f)

    units, allowed = _load_units(payload)
    pairwise = {
        (
            row["left_unit"],
            row["left_fmt"],
            row["right_unit"],
            row["right_fmt"],
        ): float(row["interaction_delta"])
        for row in payload["pairwise"]
    }
    unary = {
        unit_key: {fmt: float(delta) for fmt, delta in fmts.items()}
        for unit_key, fmts in payload["unary"].items()
    }

    result = sparse_local_refine(
        units=units,
        unary=unary,
        pairwise=pairwise,
        target_total_bits=float(payload["target_total_bits"]),
        fixed_bits_total=float(payload["fixed_bits_total"]),
        allowed=allowed,
        max_passes=args.max_passes,
    )
    result["bits_per_param"] = result["bits_total"] / max(float(payload["total_params"]), 1.0)
    refined_assignment = dict(payload["base_assignment"])
    refined_assignment.update(expand_unit_assignment(units, result["choices"]))
    out = {
        "source": args.interactions,
        "base_last_token_kl": payload["base_last_token_kl"],
        "refined_delta_kl_estimate": result["objective_delta"],
        "refined_last_token_kl_estimate": payload["base_last_token_kl"] + result["objective_delta"],
        "bits_total": result["bits_total"],
        "bits_per_param": result["bits_per_param"],
        "selected_choices": result["choices"],
        "refined_assignment": refined_assignment,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[refine] wrote {args.output}")


if __name__ == "__main__":
    main()
