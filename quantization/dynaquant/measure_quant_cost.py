#!/usr/bin/env python3
"""measure_quant_cost.py — per-(Linear, format) RTN quantization error.

Two execution modes:

  1. BATCHED GPU (default when --device cuda): groups Linears by
     (in_features, out_features) signature, stacks each group into a
     single 3D tensor, and runs ONE torch.bmm per (group, format).
     For a 35B MoE model this reduces ~31 000 tiny kernel launches
     down to ~360, which is the difference between 42-hour GPU runtime
     and ~1-3 minute runtime on unified-memory systems.

  2. UNBATCHED CPU (default when --device cpu): streams weights one at
     a time via the live model, processes sequentially. Simpler, slower
     per-item but avoids any memory-packing cost — fine for systems
     where the GPU path is slower than CPU (e.g. GB10 when operating
     on many sub-millisecond matmuls through unified memory).

Output format is identical between modes: a dict keyed by Linear name,
each entry mapping format name to {weight_mse, output_mse,
rel_output_mse}.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import signal
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn

from . import format_registry as fr


# ---------------------------------------------------------------------------
# Activation cache — lazy path index
# ---------------------------------------------------------------------------
class ActivationIndex:
    """Disk-backed activation cache.

    Building the index walks the cache dir once to map Linear name → path,
    but no tensor data is read until `load()` is called for a specific name.
    This keeps resident memory small even when the cache is 20 GB on disk.

    The probe writes files as `re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"`,
    so we apply the same forward transform to each candidate name from the
    probe stats and check for the file. This avoids ambiguity if a name ever
    contained characters that collapse under the substitution.
    """

    _FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")

    def __init__(self, cache_dir: Path, candidate_names):
        self.cache_dir = cache_dir
        self._paths: dict[str, Path] = {}
        for name in candidate_names:
            fname = self._FNAME_SUB.sub("__", name) + ".pt"
            fp = cache_dir / fname
            if fp.is_file():
                self._paths[name] = fp

    def __contains__(self, name: str) -> bool:
        return name in self._paths

    def __len__(self) -> int:
        return len(self._paths)

    def load(self, name: str) -> torch.Tensor:
        blob = torch.load(self._paths[name], map_location="cpu",
                          weights_only=False)
        return blob["inputs"]

    def names(self):
        return self._paths.keys()


# ---------------------------------------------------------------------------
# Memory-pressure watchdog
# ---------------------------------------------------------------------------
def _read_meminfo() -> dict[str, int]:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            parts = v.strip().split()
            if parts:
                info[k] = int(parts[0]) * 1024  # kB → bytes
    return info


def start_mem_watchdog(swap_grow_limit_mb: int = 256,
                       min_mem_available_mb: int = 1024,
                       interval_s: float = 2.0):
    """Background thread that aborts the process if memory pressure rises.

    Triggers a hard abort when either:
      - swap used grows by more than `swap_grow_limit_mb` vs. the baseline
        captured at watchdog start
      - MemAvailable drops below `min_mem_available_mb`

    The abort uses `os._exit(3)` after printing a diagnostic to stderr,
    bypassing any Python-level cleanup that could itself allocate memory.
    """
    baseline = _read_meminfo()
    swap_baseline = baseline.get("SwapTotal", 0) - baseline.get("SwapFree", 0)

    def loop():
        while True:
            try:
                info = _read_meminfo()
                swap_used = info.get("SwapTotal", 0) - info.get("SwapFree", 0)
                mem_avail = info.get("MemAvailable", 0)
                swap_grow_mb = (swap_used - swap_baseline) / (1024 * 1024)
                mem_avail_mb = mem_avail / (1024 * 1024)
                if swap_grow_mb > swap_grow_limit_mb:
                    print(f"\n[watchdog] ABORT: swap grew {swap_grow_mb:.0f} MB "
                          f"(limit {swap_grow_limit_mb} MB). "
                          f"MemAvailable={mem_avail_mb:.0f} MB",
                          flush=True)
                    os._exit(3)
                if mem_avail_mb < min_mem_available_mb:
                    print(f"\n[watchdog] ABORT: MemAvailable={mem_avail_mb:.0f} MB "
                          f"< floor {min_mem_available_mb} MB. "
                          f"swap_grow={swap_grow_mb:.0f} MB",
                          flush=True)
                    os._exit(3)
            except Exception:
                pass
            time.sleep(interval_s)

    t = threading.Thread(target=loop, name="mem-watchdog", daemon=True)
    t.start()
    print(f"[watchdog] armed: swap_grow_limit={swap_grow_limit_mb}MB "
          f"min_mem_avail={min_mem_available_mb}MB interval={interval_s}s  "
          f"baseline swap_used={swap_baseline//(1024*1024)}MB "
          f"mem_avail={(baseline.get('MemAvailable', 0))//(1024*1024)}MB",
          flush=True)
    return t


# ---------------------------------------------------------------------------
# Unbatched CPU path (legacy, robust)
# ---------------------------------------------------------------------------
def _stage_text_only(model_path: str) -> str:
    from .sensitivity_probe import stage_text_only
    return stage_text_only(model_path)


def _load_live_model(model_path: str, device: str, dtype: torch.dtype,
                     unfuse_moe: bool = True) -> nn.Module:
    from transformers import AutoModelForCausalLM

    staged = _stage_text_only(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        staged, torch_dtype=dtype, device_map=device,
        low_cpu_mem_usage=False, trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if unfuse_moe:
        try:
            from auto_round.modeling.fused_moe import prepare_model_for_moe_quantization
            prepare_model_for_moe_quantization(model)
            # unfused linears land on CPU by default — push to target device
            target_dev = None
            for p in model.parameters():
                if p.device.type != "cpu":
                    target_dev = p.device
                    break
            if target_dev is not None:
                for sub in model.modules():
                    if isinstance(sub, nn.Linear) and sub.weight.device != target_dev:
                        sub.to(target_dev)
        except ImportError:
            pass

    return model


def measure_unbatched(model: nn.Module, act_cache: "ActivationIndex",
                     target_names: set[str], specs: list[fr.FormatSpec],
                     device: str, dtype: torch.dtype) -> dict:
    """One-Linear-at-a-time measurement. Simple, safe, slow on small ops
    running through unified memory but robust when batching isn't an option.
    """
    results: dict[str, dict[str, dict]] = {}
    processed = 0
    tstart = time.time()
    target_list = list(target_names)
    n_total = len(target_list)

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or name not in target_names:
            continue
        if name not in act_cache:
            continue
        W = mod.weight.detach()
        X = act_cache.load(name).to(W.dtype).to(W.device)
        y_ref = X @ W.T
        ref_energy = float(y_ref.float().pow(2).mean().item())

        per_format = {}
        for spec in specs:
            try:
                W_hat = spec.quantize_dequantize(W.clone())
                X_hat = spec.activation_quantize_dequantize(X.clone())
                weight_mse = float((W - W_hat).pow(2).mean().item())
                y_q = X_hat @ W_hat.T
                output_mse = float((y_ref - y_q).float().pow(2).mean().item())
                per_format[spec.name] = {
                    "weight_mse": weight_mse,
                    "output_mse": output_mse,
                    "rel_output_mse": output_mse / max(ref_energy, 1e-12),
                }
            except Exception as e:
                per_format[spec.name] = {"error": str(e)}
        results[name] = per_format
        processed += 1
        if processed % 128 == 0:
            elapsed = time.time() - tstart
            eta = elapsed / processed * (n_total - processed)
            print(f"[cost] {processed}/{n_total} eta={eta:.0f}s", flush=True)
    return results


# ---------------------------------------------------------------------------
# Batched GPU path (fast)
# ---------------------------------------------------------------------------
def _group_by_shape(model: nn.Module, target_names: set[str]
                    ) -> dict[tuple[int, int], list[tuple[str, nn.Linear]]]:
    """Group target Linears by (in_features, out_features).

    Expert projections within an MoE share shape exactly — across layers
    too for uniform MoE models — so this groups, say, all 10 240
    gate_proj experts across 40 layers into one bucket for 35B Qwen3.6.
    """
    groups: dict[tuple[int, int], list[tuple[str, nn.Linear]]] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or name not in target_names:
            continue
        key = (mod.in_features, mod.out_features)
        groups.setdefault(key, []).append((name, mod))
    return groups


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _batched_codebook_rtn(stacked_w: torch.Tensor, codebook: torch.Tensor,
                          group_size: int) -> torch.Tensor:
    """Apply the same bucketize-based FP-codebook RTN used by format_registry,
    but on a stacked `(N, out, in)` tensor in one call. No allocation per
    inner Linear."""
    N, out_f, in_f = stacked_w.shape
    w2 = stacked_w.reshape(N, -1, in_f).float()
    if group_size > 0 and group_size < in_f:
        w2 = w2.reshape(N, -1, in_f // group_size, group_size)
    else:
        w2 = w2.unsqueeze(2)

    cb = codebook.to(device=w2.device, dtype=torch.float32).contiguous()
    cmax = float(cb.abs().max().item())
    max_abs = w2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = max_abs / cmax
    x = (w2 / scale).contiguous()

    # Bucketize returns int64 by default; cast to int32 to halve the index
    # tensor footprint. A 256-entry FP8 codebook fits comfortably in int32.
    idx = torch.bucketize(x, cb).to(torch.int32)
    idx_lo = (idx - 1).clamp_min(0)
    idx_hi = idx.clamp_max(cb.numel() - 1)
    del idx
    lo = cb[idx_lo]
    hi = cb[idx_hi]
    del idx_lo, idx_hi
    choose_hi = (hi - x).abs() < (x - lo).abs()
    q = torch.where(choose_hi, hi, lo)
    del lo, hi, choose_hi
    w_rec = q * scale
    del q
    return w_rec.reshape(N, out_f, in_f).to(stacked_w.dtype)


def _batched_int_rtn(stacked_w: torch.Tensor, bits: int, group_size: int,
                     symmetric: bool = True) -> torch.Tensor:
    N, out_f, in_f = stacked_w.shape
    w2 = stacked_w.reshape(N, -1, in_f).float()
    if group_size > 0 and group_size < in_f:
        w2 = w2.reshape(N, -1, in_f // group_size, group_size)
    else:
        w2 = w2.unsqueeze(2)
    max_abs = w2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    if symmetric:
        levels = (1 << (bits - 1)) - 1
        scale = max_abs / levels
        q = torch.round(w2 / scale).clamp(-levels - 1, levels)
        w_rec = q * scale
    else:
        levels = (1 << bits) - 1
        w_min = w2.amin(dim=-1, keepdim=True)
        w_max = w2.amax(dim=-1, keepdim=True)
        scale = (w_max - w_min) / levels
        zp = torch.round(-w_min / scale.clamp_min(1e-8))
        q = torch.round(w2 / scale.clamp_min(1e-8) + zp).clamp(0, levels)
        w_rec = (q - zp) * scale
    return w_rec.reshape(N, out_f, in_f).to(stacked_w.dtype)


# Map FormatSpec to a batched RTN function.  We could have FormatSpec carry
# its own batched op, but hardcoding the two families (codebook vs integer)
# keeps the registry simple.  New formats just declare which family they're
# in via their weight_element_dtype.
_CODEBOOK_NAMES = {
    "fp4_e2m1": "_e2m1",
    "fp6_e3m2": "_e3m2",
    "fp6_e2m3": "_e2m3",
    "fp8_e4m3": "_e4m3",
    "fp8_e5m2": "_e5m2",
}


def _batched_quantize(spec: fr.FormatSpec, stacked_w: torch.Tensor) -> torch.Tensor:
    elt = spec.weight_element_dtype
    if elt in _CODEBOOK_NAMES:
        cb_name = f"_{elt[-4:]}_codebook" if elt.startswith("fp") else None
        # Reuse the registry's codebook tables
        cb = fr._CODEBOOKS[elt]
        return _batched_codebook_rtn(stacked_w, cb, spec.group_size)
    elif elt.startswith("int"):
        return _batched_int_rtn(stacked_w, spec.weight_bits, spec.group_size)
    elif elt == "bfloat16":
        return stacked_w.clone()
    else:
        raise ValueError(f"Unknown weight_element_dtype {elt!r} for "
                         f"batched RTN")


def measure_batched_gpu(model: nn.Module, act_cache: "ActivationIndex",
                       target_names: set[str], specs: list[fr.FormatSpec],
                       device: str, dtype: torch.dtype,
                       chunk_size: int = 256) -> dict:
    """Batched GPU measurement.

    Groups Linears by shape, then within each group processes `chunk_size`
    Linears at a time. Each chunk does one stacked quantize-and-bmm per
    format. This converts the 31k-kernel-launch pathological case into a
    few hundred well-sized kernel launches.

    `chunk_size` trades latency for VRAM. For Qwen3.6-35B MoE experts
    (shape 2048×512) at BF16, one chunk of 256 = 256 MB weights; 256×256
    rows × 2048 = 128 MB activations; 3 formats × (W, Ŵ, Y_ref, Y_q) peak
    ~2 GB. Safe at chunk_size=256 on any GPU with 4+ GB free.
    """
    dev = torch.device(device)
    groups = _group_by_shape(model, target_names)
    total_linears = sum(len(v) for v in groups.values())
    print(f"[cost] batched: {len(groups)} shape groups, "
          f"{total_linears} Linears total", flush=True)

    results: dict[str, dict[str, dict]] = {}
    processed = 0
    tstart = time.time()

    for (in_f, out_f), entries in groups.items():
        entries_with_acts = [(n, m) for n, m in entries if n in act_cache]
        if not entries_with_acts:
            continue

        for chunk in _chunked(entries_with_acts, chunk_size):
            names = [n for n, _ in chunk]
            N = len(chunk)
            # Lazy load activations for this chunk only. Keep the list on
            # CPU briefly so we can pick the uniform row count (min across
            # the chunk), then stack and ship to GPU.
            acts_cpu = [act_cache.load(n) for n in names]
            chunk_min_rows = min(a.size(0) for a in acts_cpu)
            # Stack weights
            W = torch.stack([m.weight.detach().to(device=dev, dtype=dtype)
                             for _, m in chunk], dim=0)   # (N, out, in)
            # Stack activations (truncated to chunk_min_rows)
            X = torch.stack(
                [a[:chunk_min_rows].to(device=dev, dtype=dtype)
                 for a in acts_cpu], dim=0)               # (N, rows, in)
            del acts_cpu
            # Reference output (per-item BMM): shape (N, rows, out)
            y_ref = torch.bmm(X, W.transpose(1, 2))
            ref_energy = y_ref.float().pow(2).mean(dim=(1, 2))   # (N,)

            for spec in specs:
                try:
                    W_hat = _batched_quantize(spec, W)
                    X_hat = spec.activation_quantize_dequantize(X.clone())
                    weight_mse = (W - W_hat).float().pow(2).mean(dim=(1, 2))  # (N,)
                    y_q = torch.bmm(X_hat, W_hat.transpose(1, 2))
                    output_mse = (y_ref - y_q).float().pow(2).mean(dim=(1, 2))  # (N,)
                    rel_mse = output_mse / ref_energy.clamp_min(1e-12)
                    # Unpack per-item into results dict
                    for i, name in enumerate(names):
                        results.setdefault(name, {})[spec.name] = {
                            "weight_mse": float(weight_mse[i].item()),
                            "output_mse": float(output_mse[i].item()),
                            "rel_output_mse": float(rel_mse[i].item()),
                        }
                    del W_hat, X_hat, y_q, weight_mse, output_mse, rel_mse
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    for name in names:
                        results.setdefault(name, {})[spec.name] = {"error": str(e)}

            del W, X, y_ref, ref_energy
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            processed += N
            if processed % (chunk_size * 4) == 0 or processed == total_linears:
                elapsed = time.time() - tstart
                eta = elapsed / processed * (total_linears - processed)
                print(f"[cost] {processed}/{total_linears} "
                      f"eta={eta:.0f}s  ({N} per chunk × {len(specs)} formats)",
                      flush=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--activation-cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--formats", default="",
                    help="Comma-separated format names. Empty = all registered.")
    ap.add_argument("--skip-missing-activations", action="store_true")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch.set_num_threads for CPU path.")
    ap.add_argument("--mode", choices=["auto", "batched", "unbatched"],
                    default="auto",
                    help="'auto' chooses batched on CUDA, unbatched on CPU. "
                         "'batched' groups same-shape Linears for one bmm per "
                         "group. 'unbatched' processes one Linear at a time.")
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="Linears per batched chunk. Trades latency for VRAM.")
    ap.add_argument("--no-unfuse-moe", action="store_false",
                    dest="unfuse_moe", default=True)
    ap.add_argument("--swap-grow-limit-mb", type=int, default=256,
                    help="Abort if swap used grows by more than this "
                         "(MB) versus watchdog baseline. 0 to disable.")
    ap.add_argument("--min-mem-available-mb", type=int, default=2048,
                    help="Abort if MemAvailable drops below this (MB).")
    ap.add_argument("--no-watchdog", action="store_true",
                    help="Disable the memory-pressure watchdog.")
    args = ap.parse_args()

    # Watchdog is armed AFTER model load — the baseline is the post-load
    # steady state, so the limit only flags growth during measurement
    # (e.g. a leak in the activation-loading path), not transient churn
    # from HF weight-loading / MoE unfuse.

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    with open(args.probe, "rb") as f:
        probe = pickle.load(f)
    stats = probe["stats"]
    print(f"[cost] loaded probe stats for {len(stats)} Linears")

    cache = Path(args.activation_cache_dir)
    if not cache.exists():
        raise SystemExit(f"activation cache {cache} does not exist")

    if args.formats:
        fmt_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    else:
        fmt_names = [s.name for s in fr.list_formats()]
    specs = [fr.get_format(n) for n in fmt_names]
    print(f"[cost] measuring {len(specs)} formats: "
          f"{[s.name for s in specs]}")

    act_cache = ActivationIndex(cache, stats.keys())
    print(f"[cost] activation cache (lazy index): "
          f"{len(act_cache)} Linears mapped", flush=True)

    target_names = set(stats.keys())
    missing_act = [n for n in target_names if n not in act_cache]
    if missing_act and not args.skip_missing_activations:
        raise SystemExit(f"{len(missing_act)} Linears missing activation; "
                         f"pass --skip-missing-activations to proceed.")

    t0 = time.time()
    model = _load_live_model(args.model, args.device, dtype,
                             unfuse_moe=args.unfuse_moe)
    print(f"[cost] model loaded in {time.time()-t0:.1f}s", flush=True)

    if not args.no_watchdog:
        start_mem_watchdog(swap_grow_limit_mb=args.swap_grow_limit_mb,
                           min_mem_available_mb=args.min_mem_available_mb)

    mode = args.mode
    if mode == "auto":
        mode = "batched" if args.device.startswith("cuda") else "unbatched"
    print(f"[cost] mode: {mode}")

    if mode == "batched":
        results = measure_batched_gpu(model, act_cache, target_names, specs,
                                      args.device, dtype,
                                      chunk_size=args.chunk_size)
    else:
        results = measure_unbatched(model, act_cache, target_names, specs,
                                    args.device, dtype)

    missing_from_results = [n for n in target_names if n not in results]
    if missing_from_results:
        print(f"[cost] WARNING: {len(missing_from_results)} Linears had no "
              f"measurement output (cache miss or skipped)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "costs": results,
            "formats": fmt_names,
            "meta": {
                "model": args.model,
                "probe": args.probe,
                "n_linears": len(results),
                "missing_activations": missing_act,
                "mode": mode,
            },
        }, f)
    print(f"[cost] wrote {out_path} ({len(results)} Linears)")


if __name__ == "__main__":
    main()
