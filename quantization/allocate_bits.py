#!/usr/bin/env python3
"""
allocate_bits.py — water-filling bit allocator over HAWQ sensitivity curves.

Input: a sensitivity JSON from measure_hawq_sensitivity.py (or measure_bit_utility.py
       for direct curves).
Output: a Pareto frontier of (cost, predicted_KL, recipe) triples.

The allocator operates in two modes:

  1. Fine-grained (all bits 1..16): simulates the ideal Pareto frontier if
     we had custom kernels for every bit width. Uses a parametric noise
     model to predict KL at each bit count:
         KL(L, b) ≈ sensitivity_L · noise_var(b)
     where noise_var(b) = 1 / (2^(b-1) - 1)^2 for b >= 2 and ≈ 1 at b=1.

  2. Hardware-aligned (buckets {4, 8, 16}): only allocates from the three
     hardware-native formats. This is what's actually deployable today.

Water-filling algorithm:
  1. Start every Linear at the lowest bit count.
  2. For each possible upgrade (L, b→b+1), compute marginal utility per cost:
         marginal_utility(L, b→b+1) = sensitivity_L · (noise_var(b) - noise_var(b+1))
         marginal_cost(L, b→b+1) = numel_L · (b+1 - b) / 8  (bytes, fine-grained)
                                 = numel_L · (b_bucket+1 - b_bucket) / 8  (hardware)
         score = marginal_utility / marginal_cost
  3. Pop the max-score upgrade, apply it, push the next upgrade for that Linear.
  4. Record state after every upgrade → the Pareto frontier.
  5. Stop when all Linears are at max bit count.

Granularity:
  --granularity linear   (default) allocate per-Linear independently
  --granularity block    allocate per-transformer-block (all Linears in
                         model.layers.N.* share the same bit count).
                         This matches vLLM's fused-layer constraint.

Usage:
    python3 allocate_bits.py \\
        --sensitivity /tmp/curves/qwen35-4b-hawq.json \\
        --mode fine \\
        --output /tmp/pareto/qwen35-4b-fine.json

    python3 allocate_bits.py \\
        --sensitivity /tmp/curves/qwen35-4b-hawq.json \\
        --mode hw --granularity block \\
        --output /tmp/pareto/qwen35-4b-hw-block.json
"""
import argparse
import heapq
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Noise model: KL contribution from quantizing a Linear at b bits
# ---------------------------------------------------------------------------

def noise_var(bits: int) -> float:
    """Relative noise variance for symmetric INT-b per-group quantization.

    For b >= 2: noise_var ∝ 1 / (2^(b-1) - 1)^2  (sqrt(12) factor cancels
    in marginal utility ratios, so we drop it).
    For b == 1: the 1-bit case is degenerate (sign-only); we use the
    relative noise of a 1-bit quantizer compared to the 2-bit baseline,
    which empirically is ~4× worse.
    For b >= 16: ~0 (bf16 reference).
    """
    if bits >= 16:
        return 0.0
    if bits == 1:
        # 1-bit ≈ 4× worse than 2-bit (empirical from measurements)
        return 4.0
    qmax = (2 ** (bits - 1)) - 1
    return 1.0 / (qmax ** 2)


def sensitivity_of(entry: dict) -> float:
    """Extract the best per-Linear sensitivity scalar from a HAWQ entry.

    Using h_trace · mean(w²) = h_trace · w_norm_sq / numel, which had the
    highest correlation (ρ ≈ 0.93 at 4 bits) in our validation sweeps.
    """
    return entry["h_trace"] * entry["w_norm_sq"] / max(1, entry["numel"])


# ---------------------------------------------------------------------------
# Water-filling allocator
# ---------------------------------------------------------------------------

def build_linears(sensitivity_json: dict, mode: str) -> List[dict]:
    """Build the allocator's Linear list from a sensitivity JSON."""
    linears = []
    if "sensitivity" in sensitivity_json:
        # HAWQ format
        for name, entry in sensitivity_json["sensitivity"].items():
            s = sensitivity_of(entry)
            linears.append({
                "name": name,
                "numel": entry["numel"],
                "sensitivity": s,
            })
    elif "layer_scores" in sensitivity_json:
        # Older sensitivity format: {name: {quant_error, params_M, score}}
        for name, entry in sensitivity_json["layer_scores"].items():
            numel = int(entry["params_M"] * 1e6)
            # score = quant_error * params_M ≈ sensitivity * size
            # We want per-element sensitivity, so divide by numel
            s = entry["score"] / max(1, numel / 1e6)
            linears.append({
                "name": name + ".weight" if not name.endswith(".weight") else name,
                "numel": numel,
                "sensitivity": s,
            })
    elif "curves" in sensitivity_json:
        # Direct curves format — use curves directly as KL(L, b) lookup
        for name, entry in sensitivity_json["curves"].items():
            # Convert direct curves to a parametric-compatible form:
            # at 4 bits, we have the measured KL — use that to calibrate
            # a per-Linear constant.
            kl_per_bits = {int(b): kl for b, kl in entry["kl_per_bits"].items()}
            kl4 = kl_per_bits.get(4, 0.0)
            # Back-solve: sensitivity = kl4 / noise_var(4)
            sensitivity = kl4 / noise_var(4) if noise_var(4) > 0 else 0.0
            linears.append({
                "name": name,
                "numel": entry["numel"],
                "sensitivity": sensitivity,
                "measured_kl": kl_per_bits,
            })
    else:
        raise ValueError("unknown sensitivity JSON format")
    return linears


def build_blocks(linears: List[dict]) -> Tuple[List[dict], Dict[str, List[str]]]:
    """Group Linears by transformer block for block-level allocation.

    Groups all Linears matching model.layers.N.* into block N.
    Non-block Linears (embeddings, lm_head, etc.) each become their own
    singleton "block" so they still participate in allocation.

    Per-block sensitivity = max(sensitivity) across projections in the block.
    This is conservative: the most sensitive projection determines the block's
    bit budget (we can't give it fewer bits than the block gets).

    Per-block numel = sum(numel) across projections, since all projections
    in the block will be stored at the block's assigned bit count.

    Returns:
        blocks: list of dicts with name, numel, sensitivity (same shape as linears)
        block_members: dict mapping block name → list of Linear names
    """
    by_block: Dict[str, List[dict]] = defaultdict(list)
    block_order = []

    for L in linears:
        mo = re.match(r'(model\.layers\.(\d+))\.', L["name"])
        if mo:
            block_name = mo.group(1)
        else:
            # Non-block Linear → singleton block
            block_name = L["name"]
        if block_name not in by_block:
            block_order.append(block_name)
        by_block[block_name].append(L)

    blocks = []
    block_members = {}
    for block_name in block_order:
        members = by_block[block_name]
        block_members[block_name] = [m["name"] for m in members]
        blocks.append({
            "name": block_name,
            "numel": sum(m["numel"] for m in members),
            "sensitivity": max(m["sensitivity"] for m in members),
        })

    return blocks, block_members


def get_bit_options(mode: str) -> List[int]:
    """The set of bit counts the allocator can assign."""
    if mode == "fine":
        return list(range(1, 17))       # 1..16
    elif mode == "hw":
        return [4, 8, 16]                # NVFP4, FP8, BF16
    elif mode == "hw-plus-low":
        return [2, 3, 4, 5, 6, 8, 16]    # sub-4 previews
    elif mode == "full":
        return list(range(4, 17))          # 4..16 — full viable range including bf16
    elif mode == "full-plus-sub4":
        return list(range(2, 17))          # 2..16 — includes sub-4 bit widths
    else:
        raise ValueError(f"unknown mode: {mode}")


def allocator_pareto(linears: List[dict], mode: str) -> List[dict]:
    """Water-fill over bit allocations and return the Pareto frontier.

    Each step of the allocator upgrades one Linear to the next available
    bit count, ordered by marginal utility per marginal cost. We record
    the state at every step.
    """
    bit_options = get_bit_options(mode)
    min_bits = bit_options[0]
    max_bits = bit_options[-1]

    # State: current bit count per Linear
    state: Dict[str, int] = {L["name"]: min_bits for L in linears}
    lookup: Dict[str, dict] = {L["name"]: L for L in linears}

    def total_cost() -> float:
        return sum(state[n] * L["numel"] / 8 for n, L in lookup.items())

    def total_kl() -> float:
        return sum(L["sensitivity"] * noise_var(state[n]) for n, L in lookup.items())

    def next_bucket(bits: int) -> int:
        """Next larger bit count in the allowed set, or None if maxed."""
        for b in bit_options:
            if b > bits:
                return b
        return None

    def score(name: str) -> Tuple[float, int, int]:
        """Marginal utility per cost for upgrading this Linear by one step.
        Returns (negative score, current_bits, next_bits) — negative so the
        heap behaves as a max-heap."""
        cur = state[name]
        nxt = next_bucket(cur)
        if nxt is None:
            return (0.0, cur, cur)
        L = lookup[name]
        d_kl = L["sensitivity"] * (noise_var(cur) - noise_var(nxt))
        d_cost = L["numel"] * (nxt - cur) / 8
        if d_cost <= 0:
            return (0.0, cur, nxt)
        return (-d_kl / d_cost, cur, nxt)  # negative for max-heap

    # Build initial heap
    heap: List[Tuple[float, int, int, str]] = []
    for L in linears:
        neg_score, cur, nxt = score(L["name"])
        if cur < nxt:
            heapq.heappush(heap, (neg_score, cur, nxt, L["name"]))

    # Record starting state
    pareto: List[dict] = []
    pareto.append({
        "step": 0,
        "cost_bytes": total_cost(),
        "predicted_kl": total_kl(),
        "bits_histogram": _histogram(state, bit_options),
        "recipe": dict(state),
    })

    sample_every = max(1, len(linears) // 20)
    step = 0
    while heap:
        neg_score, cur, nxt, name = heapq.heappop(heap)
        if state[name] != cur:
            continue
        if neg_score >= 0:
            continue
        state[name] = nxt
        step += 1
        new_neg_score, new_cur, new_nxt = score(name)
        if new_cur < new_nxt:
            heapq.heappush(heap, (new_neg_score, new_cur, new_nxt, name))
        if step % sample_every == 0 or not heap:
            pareto.append({
                "step": step,
                "cost_bytes": total_cost(),
                "predicted_kl": total_kl(),
                "bits_histogram": _histogram(state, bit_options),
                "recipe": dict(state),
            })

    # Final state (may or may not already be recorded)
    if not pareto or pareto[-1]["step"] != step:
        pareto.append({
            "step": step,
            "cost_bytes": total_cost(),
            "predicted_kl": total_kl(),
            "bits_histogram": _histogram(state, bit_options),
            "recipe": dict(state),
        })
    return pareto


def _histogram(state: Dict[str, int], bit_options: List[int]) -> Dict[str, int]:
    h = {str(b): 0 for b in bit_options}
    for bits in state.values():
        h[str(bits)] = h.get(str(bits), 0) + 1
    return h


def find_knee(pareto: List[dict]) -> int:
    """Kneedle: index into pareto of the max-perpendicular-distance point.
    Returns the step index with the best Pareto trade-off."""
    if len(pareto) < 3:
        return len(pareto) - 1
    costs = [p["cost_bytes"] for p in pareto]
    kls = [p["predicted_kl"] for p in pareto]
    c_min, c_max = min(costs), max(costs)
    k_min, k_max = min(kls), max(kls)
    c_r = (c_max - c_min) or 1.0
    k_r = (k_max - k_min) or 1.0
    norm = [((c - c_min) / c_r, (k - k_min) / k_r) for c, k in zip(costs, kls)]
    x1, y1 = norm[0]
    x2, y2 = norm[-1]
    denom = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) or 1.0
    best_i, best_d = 0, -1.0
    for i, (x, y) in enumerate(norm):
        d = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
        if d > best_d:
            best_i, best_d = i, d
    return best_i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", required=True,
                        help="HAWQ sensitivity JSON or direct curves JSON")
    parser.add_argument("--mode", choices=["fine", "hw", "hw-plus-low", "full", "full-plus-sub4"],
                        default="fine",
                        help="fine: all bits 1..16; hw: {4,8,16}; "
                             "hw-plus-low: {2,3,4,5,6,8,16}")
    parser.add_argument("--granularity", choices=["linear", "block"],
                        default="linear",
                        help="linear: per-Linear allocation; "
                             "block: per-transformer-block (vLLM compatible)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.sensitivity) as f:
        sensitivity_json = json.load(f)

    linears = build_linears(sensitivity_json, args.mode)
    print(f"[allocate] {len(linears)} linears")

    total_params = sum(L["numel"] for L in linears)
    print(f"[allocate] total params: {total_params:,}")
    print(f"[allocate] all-bf16 cost: {total_params * 2 / 1e9:.2f} GB")
    print(f"[allocate] all-fp4 cost:  {total_params * 0.5 / 1e9:.2f} GB")
    print(f"[allocate] mode: {args.mode}")
    print(f"[allocate] granularity: {args.granularity}")

    block_members = None
    if args.granularity == "block":
        alloc_units, block_members = build_blocks(linears)
        print(f"[allocate] {len(alloc_units)} blocks "
              f"({len(linears)} linears grouped)")
    else:
        alloc_units = linears

    pareto = allocator_pareto(alloc_units, args.mode)

    # Expand block-level recipes back to per-Linear names
    if block_members is not None:
        for point in pareto:
            expanded = {}
            for block_name, bits in point["recipe"].items():
                if block_name in block_members:
                    for linear_name in block_members[block_name]:
                        expanded[linear_name] = bits
                else:
                    expanded[block_name] = bits
            point["recipe"] = expanded

    knee_idx = find_knee(pareto)

    print(f"\n[allocate] Pareto frontier ({len(pareto)} recorded points)")
    print(f"{'step':>6} {'cost GB':>10} {'pred KL':>12} {'bits histogram':>50}")
    print("-" * 85)
    for i, p in enumerate(pareto[::max(1, len(pareto) // 15)]):
        marker = "  <-- KNEE" if p["step"] == pareto[knee_idx]["step"] else ""
        hist_str = "  ".join(f"{k}:{v}" for k, v in sorted(p["bits_histogram"].items(), key=lambda x: int(x[0])) if v > 0)
        print(f"{p['step']:>6} {p['cost_bytes']/1e9:>9.3f} {p['predicted_kl']:>12.4e}  {hist_str:<50}{marker}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "source": args.sensitivity,
        "mode": args.mode,
        "granularity": args.granularity,
        "n_linears": len(linears),
        "total_params": total_params,
        "knee_step": pareto[knee_idx]["step"],
        "knee_cost_bytes": pareto[knee_idx]["cost_bytes"],
        "knee_predicted_kl": pareto[knee_idx]["predicted_kl"],
        "pareto": pareto,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[allocate] saved to {args.output}")
    print(f"[allocate] knee at step {pareto[knee_idx]['step']}, "
          f"cost {pareto[knee_idx]['cost_bytes']/1e9:.3f} GB, "
          f"predicted KL {pareto[knee_idx]['predicted_kl']:.4e}")


if __name__ == "__main__":
    main()
