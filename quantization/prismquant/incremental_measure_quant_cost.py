#!/usr/bin/env python3
"""incremental_measure_quant_cost.py — per-(Linear, format) quantization
error, streamed shard-by-shard on top of `layer_streaming`.

This is the unified cost path. There is no whole-model `from_pretrained`
load anymore: we mirror the Step-2 `incremental_probe` architecture —
one `StreamingContext` built once with the head (embed + norm + lm_head
+ rotary) resident and every decoder layer offloaded to disk / on meta.
Each shard installs its decoder layers on demand, runs
`measure_batched_gpu` / `measure_unbatched` + `_measure_packed_experts`
on the matching Linears, writes a per-shard pickle, then unloads.

MTP is folded in as a built-in shard kind: when `--include-mtp` (default
True), a shard's regex like `^mtp\\.layers\\.0\\.` triggers synthesis
of an `MtpModule` (via `mtp_probe.MtpModule`), loading of `mtp.*`
safetensors (via `_load_into_mtp`), enumeration of targets (via
`_collect_mtp_targets`), then the same measurement pipeline against
the MTP activation cache (the probe writes those activations to the
same `--activation-cache-dir` when its own `--include-mtp` was set).

Small models pay the no-op cost of a cache that can hold every layer
resident; large models drain the cache to disk as needed. The per-shard
pickle format matches `measure_quant_cost.run_cost_pass` — allocator
consumes either unchanged.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

# Must be set before the cuda allocator initializes. On Spark's UMA,
# cuda and cpu share one LPDDR5X pool; without `expandable_segments`
# the caching allocator hoards freed blocks, causing the OS to swap
# while torch's bookkeeping still thinks it has headroom.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

from . import format_registry as fr
from .incremental_probe import (
    build_extended_shard_regexes,
    build_layer_shard_regexes,
    load_num_hidden_layers,
)
from .measure_quant_cost import (
    ActivationIndex,
    HDetailIndex,
    _finalize_results,
    _measure_packed_experts,
    measure_batched_gpu,
    measure_unbatched,
    prepare_cost_context,
    start_mem_watchdog,
)
from .streaming_model import (
    StreamingContext,
    _build_streaming_context,
    _classify_shard,
)


# ---------------------------------------------------------------------------
# Per-shard pickle merge helpers (unchanged public API vs. prior version)
# ---------------------------------------------------------------------------
def merge_cost_pickles(paths: list[Path], output_path: Path):
    merged_costs = {}
    merged_formats = None
    shard_metas = []
    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        costs = data["costs"]
        overlap = set(merged_costs) & set(costs)
        if overlap:
            raise ValueError(f"cost shards overlap on {len(overlap)} entries")
        merged_costs.update(costs)
        if merged_formats is None:
            merged_formats = data.get("formats", [])
        shard_metas.append(data.get("meta", {}))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "costs": merged_costs,
            "formats": merged_formats or [],
            "meta": {
                "incremental": True,
                "n_shards": len(paths),
                "shards": shard_metas,
            },
        }, f)


def _read_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _expected_cost_shard_meta(*,
                              model: str,
                              probe_path: Path,
                              linear_include: str,
                              shard_idx: int,
                              activation_cache_dir: str,
                              mode: str,
                              chunk_size: int,
                              h_detail_dir: str | None,
                              formats: list[str]) -> dict[str, Any]:
    return {
        "model": model,
        "probe": str(probe_path),
        "activation_cache_dir": str(Path(activation_cache_dir)),
        "linear_include": linear_include,
        "mode": mode,
        "chunk_size": chunk_size,
        "h_detail_dir": str(Path(h_detail_dir)) if h_detail_dir else None,
        "shard_idx": shard_idx,
        "formats": list(formats),
    }


def cost_shard_is_reusable(path: Path, expected_meta: dict[str, Any]) -> bool:
    try:
        data = _read_pickle(path)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if "costs" not in data or "meta" not in data:
        return False
    if not isinstance(data["costs"], dict):
        return False
    meta = dict(data.get("meta", {}))
    meta.update(meta.get("incremental_shard", {}))
    actual_formats = data.get("formats", [])
    if list(actual_formats) != list(expected_meta.get("formats", [])):
        return False
    for key, expected in expected_meta.items():
        if key == "formats":
            continue
        if meta.get(key) != expected:
            return False
    return True


def annotate_cost_shard(path: Path, extra_meta: dict[str, Any]) -> None:
    data = _read_pickle(path)
    meta = dict(data.get("meta", {}))
    inc = dict(meta.get("incremental_shard", {}))
    inc.update(extra_meta)
    meta["incremental_shard"] = inc
    data["meta"] = meta
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Shared measurement step: run the configured measure path + packed experts
# against a nn.Module whose relevant Linears have already been materialized.
# Returns the final results dict keyed by canonical Linear name.
# ---------------------------------------------------------------------------
def _run_cost_measurement(
    module: nn.Module,
    *,
    act_cache: ActivationIndex,
    target_names: set[str],
    specs: list[fr.FormatSpec],
    device: str,
    dtype: torch.dtype,
    mode: str,
    chunk_size: int,
    h_detail: "HDetailIndex | None",
    log_prefix: str,
) -> dict:
    chosen_mode = mode
    if chosen_mode == "auto":
        chosen_mode = "batched" if device.startswith("cuda") else "unbatched"
    print(f"{log_prefix} mode={chosen_mode} targets={len(target_names)}",
          flush=True)

    if chosen_mode == "batched":
        results = measure_batched_gpu(
            module, act_cache, target_names, specs, device, dtype,
            chunk_size=chunk_size, h_detail=h_detail)
    else:
        results = measure_unbatched(
            module, act_cache, target_names, specs, device, dtype,
            h_detail=h_detail)

    packed_accum: dict[str, dict] = {}
    _measure_packed_experts(
        module, target_names, specs, device, dtype, packed_accum,
        h_detail=h_detail)
    if packed_accum:
        results.update(_finalize_results(packed_accum))
        print(f"{log_prefix} measured {len(packed_accum)} packed-expert tensors",
              flush=True)

    return results


# ---------------------------------------------------------------------------
# Body / lm_head shard runner — install the decoder layers matched by the
# shard regex, run the cost pipeline on Linears (and packed experts) in
# them, write a shard pickle, unload.
# ---------------------------------------------------------------------------
def _run_body_cost_shard(
    ctx: StreamingContext,
    *,
    linear_include: str,
    shard_kind: str,
    probe_stats: dict[str, dict],
    act_cache: ActivationIndex,
    specs: list[fr.FormatSpec],
    device: str,
    dtype: torch.dtype,
    mode: str,
    chunk_size: int,
    h_detail: "HDetailIndex | None",
    output_path: str,
    model_name: str,
    probe_path: str,
):
    model = ctx.model
    num_layers = ctx.num_layers
    layers_prefix = ctx.layers_prefix

    # Figure out which decoder layers (by index) this shard touches.
    # lm_head / purely-top-level shards don't need any decoder layer
    # installed — their target is a resident module.
    inc = re.compile(linear_include)
    body_layers_needed: set[int] = set()
    if shard_kind == "body":
        for L in range(num_layers):
            probe_key = f"{layers_prefix}{L}."
            # Any probe stat whose key starts with this layer prefix and
            # matches the include regex means we need the layer installed.
            for name in probe_stats:
                if name.startswith(probe_key) and inc.search(name):
                    body_layers_needed.add(L)
                    break

    # Pre-filter the target set to what actually matches this shard.
    target_names: set[str] = {
        n for n in probe_stats if inc.search(n)
    }
    if not target_names:
        print(f"[incremental-cost] shard has no matching Linears "
              f"(include={linear_include!r}); writing empty pickle",
              flush=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "costs": {},
                "formats": [s.name for s in specs],
                "meta": {
                    "model": model_name,
                    "probe": probe_path,
                    "n_linears": 0,
                    "mode": mode,
                    "shard_kind": shard_kind,
                },
            }, f)
        return

    # Install the relevant decoder layers (body path) or none (lm_head
    # shard — resolver has nothing to install at layer granularity;
    # lm_head is already resident from the context setup).
    installed = []
    try:
        t_install = time.time()
        for L in sorted(body_layers_needed):
            ctx.install(L)
            installed.append(L)
        if installed:
            print(f"[incremental-cost] installed layers {installed[0]}..{installed[-1]} "
                  f"({len(installed)}) in {time.time()-t_install:.1f}s",
                  flush=True)

        results = _run_cost_measurement(
            model,
            act_cache=act_cache,
            target_names=target_names,
            specs=specs,
            device=device,
            dtype=dtype,
            mode=mode,
            chunk_size=chunk_size,
            h_detail=h_detail,
            log_prefix=f"[incremental-cost/{shard_kind}]",
        )
    finally:
        for L in installed:
            ctx.unload(L)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    missing_from_results = [n for n in target_names if n not in results]
    if missing_from_results:
        print(f"[incremental-cost] WARNING: {len(missing_from_results)} Linears "
              f"produced no measurement (cache miss or skipped)", flush=True)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "costs": results,
            "formats": [s.name for s in specs],
            "meta": {
                "model": model_name,
                "probe": probe_path,
                "n_linears": len(results),
                "mode": ("batched" if mode == "auto" and device.startswith("cuda")
                         else ("unbatched" if mode == "auto" else mode)),
                "shard_kind": shard_kind,
            },
        }, f)
    print(f"[incremental-cost] wrote {out_path} ({len(results)} entries)",
          flush=True)


# ---------------------------------------------------------------------------
# MTP shard runner — synthesize MtpModule, load `mtp.*` weights from
# safetensors, enumerate targets, run the same measurement pipeline. The
# body context provides the text_config + device/dtype; lm_head is the
# body's resident output embedding.
# ---------------------------------------------------------------------------
def _run_mtp_cost_shard(
    ctx: StreamingContext,
    *,
    model_path: str,
    linear_include: str,
    probe_stats: dict[str, dict],
    act_cache: ActivationIndex,
    specs: list[fr.FormatSpec],
    device: str,
    dtype: torch.dtype,
    mode: str,
    chunk_size: int,
    h_detail: "HDetailIndex | None",
    output_path: str,
    model_name: str,
    probe_path: str,
):
    from .mtp_probe import MtpModule, _load_into_mtp, _load_mtp_state_dict

    inc = re.compile(linear_include)
    # Prune the probe stats to this MTP shard's regex before building
    # anything — if there's nothing to measure, emit an empty pickle.
    shard_targets: set[str] = {n for n in probe_stats if inc.search(n)}
    if not shard_targets:
        print(f"[incremental-cost/mtp] shard has no matching MTP tensors "
              f"(include={linear_include!r}); writing empty pickle",
              flush=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "costs": {},
                "formats": [s.name for s in specs],
                "meta": {
                    "model": model_name,
                    "probe": probe_path,
                    "n_linears": 0,
                    "mode": mode,
                    "shard_kind": "mtp",
                },
            }, f)
        return

    device_t = torch.device(device)
    text_config = ctx.model.config
    inner_mtp = MtpModule(text_config)
    mtp_wrapper = nn.Module()
    mtp_wrapper.add_module("mtp", inner_mtp)
    mtp_wrapper.to(device=device_t, dtype=dtype)
    mtp_wrapper.eval()

    raw = _load_mtp_state_dict(model_path)
    missing, extra = _load_into_mtp(inner_mtp, raw)
    loaded = len(raw) - len(missing)
    print(f"[incremental-cost/mtp] loaded {loaded}/{len(raw)} mtp weights "
          f"(missing={len(missing)}, module_params_unset={len(extra)})",
          flush=True)

    for p in mtp_wrapper.parameters():
        p.requires_grad_(False)

    # Enumerate the MTP module's Linears + packed experts to intersect
    # with the shard's include regex AND with what the probe actually
    # tracked (so we don't measure things the probe never instrumented).
    mtp_linears: set[str] = {
        n for n, m in mtp_wrapper.named_modules()
        if isinstance(m, nn.Linear) and not n.endswith(".mlp.gate")
    }
    for name, module in mtp_wrapper.named_modules():
        if not type(module).__name__.lower().endswith("experts"):
            continue
        for pn, p in module.named_parameters(recurse=False):
            if p.dim() == 3 and pn in {"gate_up_proj", "down_proj"}:
                mtp_linears.add(f"{name}.{pn}")

    target_names = shard_targets & mtp_linears
    if not target_names:
        print(f"[incremental-cost/mtp] shard include regex {linear_include!r} "
              f"matches probe stats but not live MTP module; writing empty pickle",
              flush=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "costs": {},
                "formats": [s.name for s in specs],
                "meta": {
                    "model": model_name,
                    "probe": probe_path,
                    "n_linears": 0,
                    "mode": mode,
                    "shard_kind": "mtp",
                },
            }, f)
        del mtp_wrapper, inner_mtp, raw
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return

    try:
        results = _run_cost_measurement(
            mtp_wrapper,
            act_cache=act_cache,
            target_names=target_names,
            specs=specs,
            device=device,
            dtype=dtype,
            mode=mode,
            chunk_size=chunk_size,
            h_detail=h_detail,
            log_prefix="[incremental-cost/mtp]",
        )
    finally:
        del mtp_wrapper, inner_mtp, raw
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "costs": results,
            "formats": [s.name for s in specs],
            "meta": {
                "model": model_name,
                "probe": probe_path,
                "n_linears": len(results),
                "mode": ("batched" if mode == "auto" and device.startswith("cuda")
                         else ("unbatched" if mode == "auto" else mode)),
                "shard_kind": "mtp",
            },
        }, f)
    print(f"[incremental-cost/mtp] wrote {out_path} ({len(results)} entries)",
          flush=True)


# ---------------------------------------------------------------------------
# Visual shard — text-only staging strips visual blocks, so we emit an
# empty pickle matching the slot so merge layout stays consistent with
# the probe's shard list.
# ---------------------------------------------------------------------------
def _write_empty_cost_shard(
    output_path: str, *, shard_kind: str,
    specs: list[fr.FormatSpec], model_name: str, probe_path: str,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "costs": {},
            "formats": [s.name for s in specs],
            "meta": {
                "model": model_name,
                "probe": probe_path,
                "n_linears": 0,
                "mode": "skipped",
                "shard_kind": shard_kind,
            },
        }, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--activation-cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--formats", default="")
    ap.add_argument("--skip-missing-activations", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mode", choices=["auto", "batched", "unbatched"],
                    default="auto")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--swap-grow-limit-mb", type=int, default=256)
    ap.add_argument("--min-mem-available-mb", type=int, default=2048)
    ap.add_argument("--no-watchdog", action="store_true")
    ap.add_argument("--layers-per-shard", type=int, default=1)
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=None)
    ap.add_argument("--include-mtp", action="store_true", default=True,
                    help="Measure cost for MTP layers (`mtp.layers.X.*`).")
    ap.add_argument("--no-include-mtp", action="store_false", dest="include_mtp")
    ap.add_argument("--include-visual", action="store_true", default=True,
                    help="Measure cost for visual encoder blocks "
                         "(`model.visual.blocks.X.*`) — currently emits "
                         "empty shard pickles since text-only staging "
                         "strips them.")
    ap.add_argument("--no-include-visual", action="store_false",
                    dest="include_visual")
    ap.add_argument("--include-lm-head", action="store_true", default=True,
                    help="Measure cost for lm_head (`^lm_head$`).")
    ap.add_argument("--no-include-lm-head", action="store_false",
                    dest="include_lm_head")
    ap.add_argument("--h-detail-dir", default=None,
                    help="If set, read per-Linear Fisher H diagonal (from "
                         "incremental_probe's --h-detail-dir) and emit "
                         "per-(layer, format) predicted_dloss alongside "
                         "weight_mse in cost.pkl.")
    args = ap.parse_args()

    n_layers = load_num_hidden_layers(args.model)
    start = max(0, args.start_layer)
    end = n_layers if args.end_layer is None else min(args.end_layer, n_layers)
    if start >= end:
        raise SystemExit(f"empty layer range: start={start} end={end}")

    body_regexes = build_layer_shard_regexes(
        n_layers, args.layers_per_shard, layer_prefix="model.layers")
    first_shard = start // args.layers_per_shard
    last_shard = (end + args.layers_per_shard - 1) // args.layers_per_shard
    shard_regexes = body_regexes[first_shard:last_shard]

    extra = build_extended_shard_regexes(
        args.model, args.layers_per_shard,
        include_body=False,
        include_mtp=args.include_mtp,
        include_visual=args.include_visual,
        include_lm_head=args.include_lm_head,
    )
    shard_regexes = shard_regexes + extra
    print(f"[incremental-cost] shard regexes: {len(shard_regexes)} total "
          f"(body={len(body_regexes[first_shard:last_shard])}, extras={len(extra)})",
          flush=True)

    work_dir = Path(args.work_dir)
    shard_dir = work_dir / "shards"
    log_dir = work_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    # Shared probe / activation / spec context — probe pickle stays in
    # memory so each shard can just intersect its regex against the stats.
    _, stats, act_cache, _, missing_act, _, specs = prepare_cost_context(
        probe_path=args.probe,
        activation_cache_dir=args.activation_cache_dir,
        formats_csv=args.formats,
        skip_missing_activations=args.skip_missing_activations,
    )

    # H-detail is optional. Built against the full probe stat set so the
    # intersect-per-shard logic just checks `name in h_detail` when needed.
    h_detail: "HDetailIndex | None" = None
    if args.h_detail_dir:
        detail_path = Path(args.h_detail_dir)
        if detail_path.exists():
            h_detail = HDetailIndex(detail_path, set(stats.keys()))
            print(f"[incremental-cost] h-detail cache: {len(h_detail)} Linears "
                  "→ full per-weight Δloss cost model", flush=True)
        else:
            print(f"[incremental-cost] WARN: h-detail dir {detail_path} not "
                  f"found; falling back to scalar proxy", flush=True)

    # Compute per-shard paths + expected metas up front so we can decide
    # whether to even bother loading the streaming context.
    shard_paths = [shard_dir / f"cost_shard_{i:03d}.pkl"
                   for i in range(len(shard_regexes))]
    expected_metas = [
        _expected_cost_shard_meta(
            model=args.model,
            probe_path=args.probe,
            linear_include=linear_include,
            shard_idx=i,
            activation_cache_dir=args.activation_cache_dir,
            mode=args.mode,
            chunk_size=args.chunk_size,
            h_detail_dir=args.h_detail_dir,
            formats=[s.name for s in specs],
        )
        for i, linear_include in enumerate(shard_regexes)
    ]
    all_reusable = all(
        shard_paths[i].exists()
        and cost_shard_is_reusable(shard_paths[i], expected_metas[i])
        for i in range(len(shard_regexes))
    )

    ctx: StreamingContext | None = None

    def _ensure_ready():
        nonlocal ctx
        if ctx is not None:
            return
        offload_folder = str(work_dir / "streaming_offload")
        ctx_local = _build_streaming_context(
            args.model,
            device=torch.device(args.device),
            dtype=dtype,
            offload_folder=offload_folder,
            log_prefix="[incremental-cost]",
        )
        # Watchdog is armed AFTER model skeleton is up — baseline is the
        # post-context state, so the trigger reflects measurement-phase
        # growth only (e.g. a leak in the activation cache path), not
        # transient churn from skeleton build / layer install.
        if not args.no_watchdog:
            start_mem_watchdog(
                swap_grow_limit_mb=args.swap_grow_limit_mb,
                min_mem_available_mb=args.min_mem_available_mb,
            )
        ctx = ctx_local

    try:
        if not all_reusable:
            _ensure_ready()

        for shard_idx, linear_include in enumerate(shard_regexes):
            shard_path = shard_paths[shard_idx]
            expected_meta = expected_metas[shard_idx]
            if shard_path.exists() and cost_shard_is_reusable(shard_path, expected_meta):
                print(f"[incremental-cost] reuse shard {shard_idx}: {shard_path}",
                      flush=True)
                continue
            if shard_path.exists():
                print(f"[incremental-cost] stale shard {shard_idx}: "
                      f"recomputing {shard_path}", flush=True)

            kind = _classify_shard(linear_include)
            print(f"[incremental-cost] shard {shard_idx} ({kind}): "
                  f"include={linear_include!r}", flush=True)
            _ensure_ready()

            if kind in ("body", "lm_head"):
                _run_body_cost_shard(
                    ctx,
                    linear_include=linear_include,
                    shard_kind=kind,
                    probe_stats=stats,
                    act_cache=act_cache,
                    specs=specs,
                    device=args.device,
                    dtype=dtype,
                    mode=args.mode,
                    chunk_size=args.chunk_size,
                    h_detail=h_detail,
                    output_path=str(shard_path),
                    model_name=args.model,
                    probe_path=args.probe,
                )
            elif kind == "mtp":
                _run_mtp_cost_shard(
                    ctx,
                    model_path=args.model,
                    linear_include=linear_include,
                    probe_stats=stats,
                    act_cache=act_cache,
                    specs=specs,
                    device=args.device,
                    dtype=dtype,
                    mode=args.mode,
                    chunk_size=args.chunk_size,
                    h_detail=h_detail,
                    output_path=str(shard_path),
                    model_name=args.model,
                    probe_path=args.probe,
                )
            else:
                # visual blocks: text-only staging strips them; emit an
                # empty pickle slot so the merged output mirrors the
                # probe shard layout.
                print(f"[incremental-cost] skip shard {shard_idx} ({kind}): "
                      f"not supported in streaming path", flush=True)
                _write_empty_cost_shard(
                    str(shard_path), shard_kind=kind, specs=specs,
                    model_name=args.model, probe_path=args.probe,
                )

            annotate_cost_shard(shard_path, expected_meta)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        if ctx is not None:
            ctx.shutdown()

    merge_cost_pickles(shard_paths, Path(args.output))
    print(f"[incremental-cost] wrote merged cost to {args.output}", flush=True)


if __name__ == "__main__":
    main()
