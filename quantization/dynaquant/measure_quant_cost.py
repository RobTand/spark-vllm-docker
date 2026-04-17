#!/usr/bin/env python3
"""measure_quant_cost.py — closed-loop per-(layer, format) quantization error.

For each tracked Linear and each candidate format in the registry, we:

  1. Read the Linear's weight tensor from the model
  2. Apply the format's RTN quantize-then-dequantize to get Ŵ
  3. Compute multiple error metrics using the saved input activations X:
       weight_mse    = ‖W - Ŵ‖² / numel(W)
       output_mse    = ‖W·X - Ŵ·X‖² / numel(W·X)
       rel_output_mse = output_mse / ‖W·X‖² / numel(W·X)

  4. Emit a cost matrix of shape [n_linears, n_formats] plus each metric.

The output_mse is the crucial signal: it captures the actual functional
perturbation that the format introduces into this Linear's output, which
is what downstream layers ultimately see.  Combined with per-Linear
sensitivity from sensitivity_probe.py, it gives us an (approximate but
data-grounded) predicted loss delta:

    Δloss_layer ≈ 0.5 · H_trace_layer · output_mse_layer · d_out

Running this tool is much cheaper than the probe itself — no backward
pass, no calibration, just per-Linear matmuls on saved activations.
Typical runtime on a 35B model with ~300 Linears × 8 formats:
~3 minutes on GPU, ~15 minutes on CPU.

Memory: streams one Linear at a time, never holds more than one weight
in VRAM. Safe for 128 GB unified memory on a 35 B model.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import torch
import torch.nn as nn

from . import format_registry as fr


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "__", name)


def _build_weight_index(model_path: str) -> dict[str, str]:
    """Return {tensor_name: safetensors_file} by reading the model's index.

    Handles both sharded (model.safetensors.index.json) and single-file
    layouts, and falls back to scanning shards if no index exists.
    """
    import os
    from safetensors import safe_open

    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        return idx["weight_map"]

    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        with safe_open(single, framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}

    # Scan all shards
    out = {}
    for entry in sorted(os.listdir(model_path)):
        if entry.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, entry), framework="pt") as f:
                for k in f.keys():
                    out[k] = entry
    return out


def stream_weights(model_path: str, device: str, dtype: torch.dtype,
                   linear_names: set[str]):
    """Yield (name, weight_tensor) for each tracked Linear, reading directly
    from safetensors shards. No model instantiation — memory footprint is
    bounded by the largest weight tensor (< 1 GB for any single Linear).

    Handles multimodal wrappers by trying both `<name>.weight` and
    `<name>.weight` with a `model.language_model.` prefix swap, matching
    the convention used elsewhere in the project.
    """
    import os
    from safetensors import safe_open

    # Look up where the actual tensors live. Try staged path first; fall
    # back to the original if the stage is a symlinked tmp.
    staged = _stage_text_only(model_path)
    search_paths = [staged, model_path]

    weight_map = None
    chosen_root = None
    for p in search_paths:
        try:
            weight_map = _build_weight_index(p)
            chosen_root = p
            break
        except Exception:
            continue
    if weight_map is None:
        raise RuntimeError(f"Could not find any safetensors under "
                           f"{search_paths}")

    # Multimodal wrapper support: both `X.weight` and `model.language_model.X.weight`
    def _resolve_key(name: str) -> str | None:
        candidates = [f"{name}.weight"]
        if name.startswith("model.") and "model.language_model." not in name:
            alt = name.replace("model.", "model.language_model.", 1)
            candidates.append(f"{alt}.weight")
        for c in candidates:
            if c in weight_map:
                return c
        return None

    # Group by shard to open each file once
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for name in linear_names:
        key = _resolve_key(name)
        if key is None:
            continue
        by_shard.setdefault(weight_map[key], []).append((name, key))

    device_obj = torch.device(device)
    for shard_name, items in by_shard.items():
        shard_path = os.path.join(chosen_root, shard_name)
        with safe_open(shard_path, framework="pt", device=device) as f:
            for name, key in items:
                W = f.get_tensor(key)
                if W.dtype != dtype:
                    W = W.to(dtype)
                yield name, W
                # Tensor is not cached inside safe_open; it releases
                # when `W` goes out of scope in the caller's loop.


def _stage_text_only(model_path: str) -> str:
    # Minimal staging to drop multimodal config keys without touching files
    # beyond a symlinked tmp dir. Separated out so this script doesn't
    # re-import the probe module just for this helper.
    from .sensitivity_probe import stage_text_only
    return stage_text_only(model_path)


def measure_one(layer_weight: torch.Tensor, activations: torch.Tensor,
                spec: fr.FormatSpec) -> dict:
    """Return {weight_mse, output_mse, rel_output_mse} for one (layer, format)."""
    W = layer_weight.detach()
    W_hat = spec.quantize_dequantize(W.clone())
    weight_mse = float((W - W_hat).pow(2).mean().item())

    # Output error:  ‖x · W.T - x · Ŵ.T‖² / numel
    X = activations.to(W.dtype).to(W.device)
    y_ref = X @ W.T
    y_quant = X @ W_hat.T
    diff = (y_ref - y_quant).float()
    output_mse = float(diff.pow(2).mean().item())
    ref_energy = float(y_ref.float().pow(2).mean().item())
    rel_output_mse = output_mse / max(ref_energy, 1e-12)
    return {
        "weight_mse": weight_mse,
        "output_mse": output_mse,
        "rel_output_mse": rel_output_mse,
    }


def measure_all_formats(layer_weight: torch.Tensor, activations: torch.Tensor,
                        specs: list[fr.FormatSpec]) -> dict:
    """Measure every (format) for one Linear in a single call.

    We pre-compute the reference y_ref = X @ W.T once and reuse it across
    all formats, saving ~len(specs) extra matmuls per Linear.
    """
    W = layer_weight.detach()
    X = activations.to(W.dtype).to(W.device)
    y_ref = X @ W.T
    ref_energy = float(y_ref.float().pow(2).mean().item())
    out = {}
    for spec in specs:
        try:
            W_hat = spec.quantize_dequantize(W.clone())
            weight_mse = float((W - W_hat).pow(2).mean().item())
            y_q = X @ W_hat.T
            diff = (y_ref - y_q).float()
            output_mse = float(diff.pow(2).mean().item())
            out[spec.name] = {
                "weight_mse": weight_mse,
                "output_mse": output_mse,
                "rel_output_mse": output_mse / max(ref_energy, 1e-12),
            }
        except Exception as e:
            out[spec.name] = {"error": str(e)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True,
                    help="Pickle from sensitivity_probe.py")
    ap.add_argument("--activation-cache-dir", required=True,
                    help="Directory with per-Linear input snapshots")
    ap.add_argument("--output", required=True,
                    help="Output pickle with per-(layer, format) cost entries")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--formats", default="",
                    help="Comma-separated format names. Empty = all registered.")
    ap.add_argument("--skip-missing-activations", action="store_true",
                    help="Skip Linears without a cached activation file "
                         "(otherwise error).")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch.set_num_threads for CPU path (0 = default).")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    with open(args.probe, "rb") as f:
        probe = pickle.load(f)
    stats = probe["stats"]
    print(f"[cost] loaded probe stats for {len(stats)} Linears")

    cache = Path(args.activation_cache_dir)
    if not cache.exists():
        raise SystemExit(f"activation cache {cache} does not exist")

    if args.formats:
        chosen_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    else:
        chosen_names = [s.name for s in fr.list_formats()]
    chosen = [fr.get_format(n) for n in chosen_names]
    print(f"[cost] measuring {len(chosen)} formats: "
          f"{[s.name for s in chosen]}")

    # Load all activations into a dict (they're tiny — 256 rows per Linear)
    act_cache: dict[str, torch.Tensor] = {}
    for fp in cache.glob("*.pt"):
        blob = torch.load(fp, map_location="cpu")
        act_cache[blob["name"]] = blob["inputs"]
    print(f"[cost] activation cache: {len(act_cache)} Linears")

    # Respect user-provided thread budget on CPU. With torch's default
    # thread setting, small-matmul RTN work underutilizes the CPU — set
    # it high to saturate. GPU path ignores this.
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    print(f"[cost] torch intra-op threads: {torch.get_num_threads()}")

    results: dict[str, dict[str, dict]] = {}
    missing = []
    tstart = time.time()

    # Stream weights from model one at a time
    target_names = set(stats.keys())
    processed = 0
    for name, weight in stream_weights(args.model, args.device, dtype,
                                        target_names):
        if name not in act_cache:
            if args.skip_missing_activations:
                missing.append(name)
                continue
            raise RuntimeError(f"No activation cache for {name}")
        X = act_cache[name]
        # Measure all formats in one call: reuses the X@W.T reference matmul.
        results[name] = measure_all_formats(weight, X, chosen)
        processed += 1
        if processed % 32 == 0:
            elapsed = time.time() - tstart
            eta = elapsed / processed * (len(target_names) - processed)
            print(f"[cost] {processed}/{len(target_names)} "
                  f"eta={eta:.0f}s", flush=True)

    if missing:
        print(f"[cost] WARNING: {len(missing)} Linears had no activation cache")
    print(f"[cost] done in {time.time()-tstart:.1f}s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "costs": results,
            "formats": chosen_names,
            "meta": {
                "model": args.model,
                "probe": args.probe,
                "n_linears": len(results),
                "missing_activations": missing,
            },
        }, f)
    print(f"[cost] wrote {out_path}")


if __name__ == "__main__":
    main()
