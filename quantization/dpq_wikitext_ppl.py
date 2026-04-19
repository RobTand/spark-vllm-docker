#!/usr/bin/env python3
"""
Real wikitext-2 perplexity comparison for DPQ recipes.

Standard sliding-window PPL on the wikitext-2-raw-v1 test split. Uses 2048-token
windows with 1024-token stride (non-overlapping halves), which is the
conventional setup for PPL benchmarks like the one in the GPT-2 paper and
nanoGPT. Results are directly comparable to PPL numbers reported elsewhere.

Loads each model serially (one at a time) so peak GPU memory stays bounded.

Usage:
    python3 dpq_wikitext_ppl.py \\
        --models bf16=/models/Qwen3.5-27B-bf16 \\
                 fp4=/models/Qwen3.5-27B-allfp4-simulated \\
                 dpq=/models/Qwen3.5-27B-dpq-simulated \\
        --max-tokens 32768 \\
        --save /tmp/dpq_wikitext_ppl.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_wikitext_text() -> str:
    """Load wikitext-2 raw test split as a single string."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(row["text"] for row in ds if row["text"].strip())


@torch.no_grad()
def measure_sliding_window_ppl(
    model,
    tokenizer,
    text: str,
    *,
    max_length: int = 2048,
    stride: int = 1024,
    max_tokens: int = None,
    device=None,
) -> dict:
    """
    Sliding-window perplexity following the HF/transformers convention.

    Reference: https://huggingface.co/docs/transformers/perplexity

    Args:
        model: HF causal LM (already on device).
        tokenizer: Matching tokenizer.
        text: Long string of text.
        max_length: Window length (tokens). 2048 is standard.
        stride: Stride between windows. Half of max_length is conventional.
        max_tokens: Optional cap on total tokens evaluated (for speed).
        device: Override device.

    Returns:
        dict with 'perplexity', 'n_tokens_evaluated', 'n_windows', 'wallclock_sec'.
    """
    if device is None:
        device = next(model.parameters()).device

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    seq_len = input_ids.size(1)

    if max_tokens is not None and seq_len > max_tokens:
        input_ids = input_ids[:, :max_tokens]
        seq_len = max_tokens

    print(f"  total tokens to score: {seq_len}", flush=True)

    nlls = []
    n_tokens_seen = 0
    prev_end = 0
    n_windows = 0
    t0 = time.time()

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        # The "target length" — how many tokens this window contributes
        # (avoid double-counting overlapping tokens with the previous window)
        trg_len = end - prev_end
        window = input_ids[:, begin:end]

        target_ids = window.clone()
        target_ids[:, :-trg_len] = -100  # mask out prefix tokens (already counted)

        out = model(window, labels=target_ids)
        # out.loss is averaged over (trg_len - 1) target positions (last is shifted)
        # so total NLL contribution is loss * (trg_len)  — actually just sum
        # the per-token nlls properly
        neg_log_likelihood = out.loss * (trg_len - 1) if trg_len > 1 else out.loss * trg_len
        nlls.append(neg_log_likelihood.item())
        n_tokens_seen += trg_len
        n_windows += 1
        prev_end = end

        if n_windows % 5 == 0 or end == seq_len:
            elapsed = time.time() - t0
            print(f"    window {n_windows}: end={end}/{seq_len}  elapsed={elapsed:.0f}s", flush=True)

        if end == seq_len:
            break

    total_nll = sum(nlls)
    avg_nll = total_nll / max(1, n_tokens_seen)
    ppl = math.exp(avg_nll)

    return {
        "perplexity": ppl,
        "n_tokens_evaluated": n_tokens_seen,
        "n_windows": n_windows,
        "wallclock_sec": time.time() - t0,
    }


def evaluate_model(name: str, path: str, text: str, max_length: int, max_tokens: int) -> dict:
    print(f"\n[ppl] === {name} ({path}) ===", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    print(f"[ppl] {name} loaded in {time.time() - t0:.0f}s", flush=True)

    result = measure_sliding_window_ppl(
        model, tokenizer, text,
        max_length=max_length,
        stride=max_length // 2,
        max_tokens=max_tokens,
    )
    result["name"] = name
    result["path"] = path
    print(f"[ppl] {name} perplexity = {result['perplexity']:.4f} "
          f"(n_tokens={result['n_tokens_evaluated']}, "
          f"wall={result['wallclock_sec']:.0f}s)", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="name=path pairs")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="window length in tokens (default 2048)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="cap total tokens evaluated (default: all of wikitext-2)")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    print(f"[ppl] loading wikitext-2 test split...", flush=True)
    text = get_wikitext_text()
    print(f"[ppl] {len(text)} chars of text loaded", flush=True)

    results = []
    for spec in args.models:
        if "=" not in spec:
            raise SystemExit(f"--models entries must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        result = evaluate_model(name, path, text, args.max_length, args.max_tokens)
        results.append(result)

    print("\n" + "=" * 80)
    print("WIKITEXT-2 PPL SUMMARY")
    print("=" * 80)
    print(f"{'model':<20s} {'PPL':>10s} {'tokens':>10s} {'wall':>8s}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<20s} {r['perplexity']:>10.4f} {r['n_tokens_evaluated']:>10d} {r['wallclock_sec']:>6.0f}s")

    # Compute relative deltas
    if len(results) >= 2:
        baseline = results[0]
        print(f"\nRelative to {baseline['name']}:")
        for r in results[1:]:
            rel = (r["perplexity"] - baseline["perplexity"]) / baseline["perplexity"] * 100
            print(f"  {r['name']:<18s} {rel:+.2f}% PPL")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[ppl] saved to {args.save}")
