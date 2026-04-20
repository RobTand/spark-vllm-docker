#!/usr/bin/env python3
"""incremental_probe.py — PrismQuant sensitivity probe, streamed shard-by-shard.

This is the unified probe path. There is no separate "whole model fits in
RAM" branch: the model is always loaded via the layer-streaming primitives
in `layer_streaming`, with the head (embed + norm + lm_head + rotary)
resident and decoder layers offloaded to disk and streamed in on demand.
Small models just pay the no-op cost of a LayerCache that can hold every
layer resident; large models drain the cache to disk as needed.

Each shard (body layer range, MTP, lm_head) runs one streaming pass: the
exact phase-1 / phase-2 / phase-3 flow from `streaming_probe.run_streaming_probe`,
specialized to Fisher-instrument only the Linears matching that shard's
regex. MTP is a built-in shard kind: after the body forward we synthesize
a `MtpModule`, load `mtp.*` weights directly from safetensors, and run
its own forward+backward for Fisher collection. The per-shard pickle
output format matches `sensitivity_probe.run_probe_pass` / `streaming_probe`
unchanged — the allocator consumes either.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Must be set before the cuda allocator initializes. On Spark's UMA,
# cuda and cpu share one LPDDR5X pool; without `expandable_segments`
# the caching allocator hoards freed blocks, causing the OS to swap
# while torch's bookkeeping still thinks it has headroom.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_streaming import (
    _call_layer,
    _compute_position_embeddings,
    _make_causal_mask,
)
from .sensitivity_probe import (
    FisherAccumulator,
    RouterTracker,
    discover_moe_structure,
    install_packed_expert_hooks,
    load_calibration,
    per_token_ce,
    read_top_k,
    stage_text_only,
)
from .streaming_model import (
    StreamingContext,
    _build_streaming_context,
    _classify_shard,
)


# ---------------------------------------------------------------------------
# Shard regex builders (unchanged public API)
# ---------------------------------------------------------------------------
def build_layer_shard_regexes(num_hidden_layers: int,
                              layers_per_shard: int,
                              layer_prefix: str = "model.layers") -> list[str]:
    regexes: list[str] = []
    for start in range(0, num_hidden_layers, layers_per_shard):
        end = min(start + layers_per_shard, num_hidden_layers)
        if end - start == 1:
            body = rf"{re.escape(layer_prefix)}\.{start}\."
        else:
            idxs = "|".join(str(i) for i in range(start, end))
            body = rf"{re.escape(layer_prefix)}\.(?:{idxs})\."
        regexes.append(body)
    return regexes


def build_extended_shard_regexes(
    model_path: str,
    layers_per_shard: int,
    *,
    include_body: bool = True,
    include_mtp: bool = True,
    include_visual: bool = True,
    include_lm_head: bool = True,
) -> list[str]:
    """Extended shard list covering everything quantizable in a
    multimodal Qwen3.5/3.6 checkpoint:

      - body transformer    (`model.layers.X.*`)         — N shards
      - MTP block(s)        (`mtp.layers.X.*`)           — 1 shard typically
      - visual ViT blocks   (`model.visual.blocks.X.*`)  — depth/N shards
      - lm_head             (`^lm_head$`)                — 1 shard
    """
    src_cfg_path = Path(model_path) / "config.json"
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    vis_cfg = cfg.get("vision_config", {})

    regexes: list[str] = []

    if include_body:
        n_body = int(text_cfg.get("num_hidden_layers", cfg.get("num_hidden_layers", 0)))
        regexes.extend(build_layer_shard_regexes(
            n_body, layers_per_shard, layer_prefix="model.layers"))

    if include_mtp:
        n_mtp = int(
            text_cfg.get("num_nextn_predict_layers")
            or cfg.get("num_nextn_predict_layers")
            or text_cfg.get("num_mtp_layers")
            or cfg.get("num_mtp_layers")
            or _count_mtp_layers_from_safetensors(model_path)
            or 0
        )
        if n_mtp > 0:
            regexes.extend(build_layer_shard_regexes(
                n_mtp, layers_per_shard, layer_prefix="mtp.layers"))

    if include_visual:
        n_vis = int(vis_cfg.get("depth") or vis_cfg.get("num_hidden_layers") or 0)
        if n_vis > 0:
            vis_per_shard = max(layers_per_shard, 4)
            regexes.extend(build_layer_shard_regexes(
                n_vis, vis_per_shard, layer_prefix="model.visual.blocks"))

    if include_lm_head:
        regexes.append(r"^lm_head$")

    return regexes


def _count_mtp_layers_from_safetensors(model_path: str) -> int:
    """Fallback for when the config doesn't carry an MTP layer count:
    scan the source safetensors index and count `mtp.layers.<N>.` paths."""
    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    if not idx_path.exists():
        try:
            from safetensors.torch import safe_open
            mtp_indices: set[int] = set()
            for f in os.listdir(src):
                if not f.endswith(".safetensors"):
                    continue
                with safe_open(str(src / f), framework="pt") as sf:
                    for k in sf.keys():
                        m = re.match(r"^mtp\.layers\.(\d+)\.", k)
                        if m:
                            mtp_indices.add(int(m.group(1)))
            return max(mtp_indices) + 1 if mtp_indices else 0
        except Exception:
            return 0
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]
    mtp_indices = set()
    for k in wm:
        m = re.match(r"^mtp\.layers\.(\d+)\.", k)
        if m:
            mtp_indices.add(int(m.group(1)))
    return max(mtp_indices) + 1 if mtp_indices else 0


# ---------------------------------------------------------------------------
# Per-shard pickle merge helpers (unchanged)
# ---------------------------------------------------------------------------
def _merge_nested_counts(dst: dict, src: dict):
    for key, sub in src.items():
        tgt = dst.setdefault(key, {})
        for sk, sv in sub.items():
            tgt[sk] = tgt.get(sk, 0.0) + float(sv)


def _read_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _expected_probe_shard_meta(args, *,
                               linear_include: str,
                               shard_idx: int,
                               activation_cache_dir: str) -> dict[str, Any]:
    return {
        "model": args.model,
        "dataset": args.dataset,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "dtype": args.dtype,
        "requested_device": args.device,
        "requested_device_map": str(args.device_map),
        "importance_weighting": args.importance_weighting,
        "activation_cache_dir": str(Path(activation_cache_dir)),
        "linear_include": linear_include,
        "linear_exclude": (
            r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|block_sparse_moe\.gate$)"
        ),
        "h_detail_dir": str(Path(args.h_detail_dir)) if args.h_detail_dir else None,
        "shard_idx": shard_idx,
    }


def probe_shard_is_reusable(path: Path, expected_meta: dict[str, Any]) -> bool:
    try:
        data = _read_pickle(path)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if "stats" not in data or "meta" not in data:
        return False
    if not isinstance(data["stats"], dict):
        return False
    meta = data.get("meta") or {}
    probe_meta = dict(meta)
    probe_meta.update(meta.get("incremental_shard", {}))
    for key, expected in expected_meta.items():
        if probe_meta.get(key) != expected:
            return False
    return True


def annotate_probe_shard(path: Path, extra_meta: dict[str, Any]) -> None:
    data = _read_pickle(path)
    meta = dict(data.get("meta", {}))
    inc = dict(meta.get("incremental_shard", {}))
    inc.update(extra_meta)
    meta["incremental_shard"] = inc
    data["meta"] = meta
    with open(path, "wb") as f:
        pickle.dump(data, f)


def merge_probe_pickles(paths: list[Path], output_path: Path):
    merged = None
    merged_stats = {}
    merged_router_counts = {}
    merged_router_totals = defaultdict(int)
    merged_expert_info = {}
    shard_metas = []

    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if merged is None:
            merged = data
        overlap = set(merged_stats) & set(data["stats"])
        if overlap:
            raise ValueError(f"probe shards overlap on {len(overlap)} stats entries")
        merged_stats.update(data["stats"])
        _merge_nested_counts(merged_router_counts, data.get("router_counts", {}))
        for rk, rv in data.get("router_totals", {}).items():
            merged_router_totals[rk] += int(rv)
        merged_expert_info.update(data.get("expert_info", {}))
        shard_metas.append(data.get("meta", {}))

    if merged is None:
        raise ValueError("no probe shards to merge")

    merged["stats"] = merged_stats
    merged["router_counts"] = dict(merged_router_counts)
    merged["router_totals"] = dict(merged_router_totals)
    merged["expert_info"] = merged_expert_info
    merged["meta"] = {
        **merged.get("meta", {}),
        "incremental": True,
        "n_shards": len(paths),
        "shards": shard_metas,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(merged, f)


def load_num_hidden_layers(model_path: str) -> int:
    staged = stage_text_only(model_path)
    cfg_path = Path(staged) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    n = cfg.get("num_hidden_layers")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Could not infer num_hidden_layers from {cfg_path}")
    return n


# Streaming infrastructure — `StreamingContext`, `_build_streaming_context`,
# and `_classify_shard` live in `streaming_model` so both the probe and
# the cost measurement share one implementation.


# ---------------------------------------------------------------------------
# Per-shard body runner — phase-1 / phase-2 / phase-3 of streaming_probe,
# scoped to the Linears matching this shard's regex. The non-matching
# layers are forwarded without Fisher instrumentation so hidden-state
# chains stay intact.
# ---------------------------------------------------------------------------
def _run_body_streaming_shard(
    ctx: StreamingContext,
    *,
    calib: torch.Tensor,
    linear_include: str,
    linear_exclude: str,
    importance_weighting: bool,
    activation_cache_dir: str | None,
    h_detail_dir: str | None,
    output_path: str,
    dataset_name: str,
    dtype_name: str,
    seqlen: int,
    model_path: str,
    prefetch_lookahead: int = 3,
):
    device = ctx.device
    dtype = ctx.dtype
    model = ctx.model
    base_model = ctx.base_model
    layers = ctx.layers
    num_layers = ctx.num_layers
    layers_prefix = ctx.layers_prefix

    inc = re.compile(linear_include)
    exc = re.compile(linear_exclude)
    all_linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    all_tracked = [n for n in all_linears
                   if inc.search(n) and not exc.search(n)]
    layer_linear_names: list[list[str]] = []
    for L in range(num_layers):
        pref = f"{layers_prefix}{L}."
        layer_linear_names.append([n for n in all_tracked if n.startswith(pref)])
    total_tracked = sum(len(x) for x in layer_linear_names)
    if total_tracked == 0:
        print(f"[incremental] shard has no Linears matching "
              f"{linear_include!r} under {layers_prefix}*; writing empty pickle",
              flush=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "stats": {},
                "router_counts": {},
                "router_totals": {},
                "expert_info": {},
                "meta": {
                    "model": model_path,
                    "dataset": dataset_name,
                    "nsamples": int(calib.size(0)),
                    "seqlen": seqlen,
                    "dtype": dtype_name,
                    "device_map": "streaming-layerwise",
                    "execution_device": str(device),
                    "top_k": read_top_k(model, default=2),
                    "importance_weighting": importance_weighting,
                    "activation_cache_dir": activation_cache_dir,
                    "linear_include": linear_include,
                    "linear_exclude": linear_exclude,
                },
            }, f)
        return
    print(f"[incremental] body shard: tracking {total_tracked} Linears "
          f"across {sum(1 for x in layer_linear_names if x)} layers "
          f"(include={linear_include!r})", flush=True)

    top_k = read_top_k(model, default=2)

    merged_stats: dict[str, dict] = {}
    merged_h_full: dict[str, torch.Tensor] = {}

    tokens_in_sample = calib.size(-1)
    batch_size = calib.size(0)

    ids = calib.to(device)
    position_ids = torch.arange(tokens_in_sample, device=device).unsqueeze(0)

    prefetch_depth = prefetch_lookahead

    # ---- Phase 1: streaming forward, cache activations on CPU ----
    t_phase = time.time()
    with torch.no_grad():
        hidden = base_model.embed_tokens(ids).to(dtype)
    causal_mask = _make_causal_mask(tokens_in_sample, device, dtype)
    position_embeddings = _compute_position_embeddings(
        base_model, hidden, position_ids)
    print(f"[incremental] phase-1 N={batch_size} T={tokens_in_sample} "
          f"hidden={tuple(hidden.shape)}", flush=True)

    for d in range(prefetch_depth):
        ctx.schedule_prefetch(d)
    activations_cpu: list[torch.Tensor] = [hidden.detach().cpu()]
    for L in range(num_layers):
        load_t0 = time.time()
        src = ctx.install(L)
        ctx.schedule_prefetch(L + prefetch_depth)
        load_s = time.time() - load_t0
        fwd_t0 = time.time()
        with torch.no_grad():
            out = _call_layer(
                layers[L], hidden,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
        fwd_s = time.time() - fwd_t0
        hidden = out
        activations_cpu.append(hidden.detach().cpu())
        ctx.unload(L)
        if L % 8 == 0 or L == num_layers - 1:
            print(f"[incremental] fwd L{L:02d}  src={src}  load={load_s:.2f}s  "
                  f"fwd={fwd_s:.2f}s", flush=True)
    print(f"[incremental] phase-1 forward: {time.time()-t_phase:.1f}s  "
          f"{ctx.layer_cache.summary()}", flush=True)

    # ---- Phase 2: final norm + lm_head + loss; grad at final hidden ----
    ctx.layer_cache.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t_phase = time.time()
    final_hidden = activations_cpu[-1].to(device).to(dtype).requires_grad_(True)
    norm_out = base_model.norm(final_hidden)
    norm_out_d = norm_out.detach().requires_grad_(True)
    grad_buf = torch.zeros_like(norm_out_d)
    chunk_T = 256
    N, T, _ = norm_out_d.shape
    if importance_weighting:
        total_ce, total_count = 0.0, 0
        for start in range(0, T - 1, chunk_T):
            end = min(start + chunk_T, T)
            with torch.no_grad():
                preds = model.lm_head(norm_out_d[:, start:end, :]).float()
                cut = end - 1 - start if end >= T else end - start
                if cut <= 0:
                    continue
                preds = preds[:, :cut, :]
                tgt = ids[:, start + 1:start + 1 + cut]
                lp_c = F.log_softmax(preds.reshape(-1, preds.size(-1)), dim=-1)
                tok_ce = -lp_c.gather(1, tgt.reshape(-1, 1)).squeeze(1)
                total_ce += float(tok_ce.sum().item())
                total_count += int(tok_ce.numel())
        ce_mean = total_ce / max(total_count, 1)
    else:
        ce_mean = None

    for start in range(0, T - 1, chunk_T):
        end = min(start + chunk_T, T)
        cut = end - 1 - start if end >= T else end - start
        if cut <= 0:
            continue
        preds = model.lm_head(norm_out_d[:, start:end, :]).float()[:, :cut, :]
        tgt = ids[:, start + 1:start + 1 + cut]
        lp_c = F.log_softmax(preds.reshape(-1, preds.size(-1)), dim=-1)
        tok_ce = -lp_c.gather(1, tgt.reshape(-1, 1)).squeeze(1)
        if importance_weighting:
            with torch.no_grad():
                w = (tok_ce.detach() / max(ce_mean, 1e-6)).clamp(0.25, 4.0)
            chunk_loss = (tok_ce * w).sum()
        else:
            chunk_loss = tok_ce.sum()
        g, = torch.autograd.grad(chunk_loss, norm_out_d, retain_graph=False)
        grad_buf.add_(g)
        del preds, lp_c, tok_ce, chunk_loss, g
    norm_out.backward(grad_buf)
    grad_at_tail = final_hidden.grad.detach().clone()
    del grad_buf, norm_out, norm_out_d, final_hidden
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[incremental] phase-2 loss+head bwd: {time.time()-t_phase:.1f}s",
          flush=True)

    # ---- Phase 3: reverse sweep, Fisher collection only on tracked Linears ----
    t_phase = time.time()
    grad_out = grad_at_tail
    for d in range(prefetch_depth):
        ctx.schedule_prefetch(num_layers - 1 - d)

    activation_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
    activation_rows: dict[str, int] = defaultdict(int)
    input_rows_limit = 256
    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    packed_act_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
    packed_act_rows: dict[str, int] = defaultdict(int)

    for L in reversed(range(num_layers)):
        load_t0 = time.time()
        src = ctx.install(L)
        ctx.schedule_prefetch(L - prefetch_depth)
        load_s = time.time() - load_t0

        tracked_here = layer_linear_names[L]
        acc_h_full: dict[str, torch.Tensor] = {}
        acc_stats: dict[str, dict] = {}
        saved_inputs: dict[str, torch.Tensor] = {}
        handles: list = []

        def make_fwd(name: str):
            def hook(module, inp, out):
                x = inp[0] if isinstance(inp, tuple) else inp
                saved_inputs[name] = x.detach()
                if cache_dir is not None:
                    need = input_rows_limit - activation_rows[name]
                    if need > 0:
                        flat = x.detach().reshape(-1, x.size(-1))
                        if flat.size(0) > need:
                            idx = torch.randperm(flat.size(0), device=flat.device)[:need]
                            flat = flat.index_select(0, idx)
                        activation_snaps[name].append(flat.to("cpu"))
                        activation_rows[name] += flat.size(0)
            return hook

        def make_bwd(name: str, mod_ref: nn.Linear):
            def hook(module, grad_input, grad_output):
                gy = grad_output[0]
                x = saved_inputs.pop(name, None)
                if x is None or gy is None:
                    return
                gy2 = gy.reshape(-1, gy.size(-1))
                x2 = x.reshape(-1, x.size(-1))
                grad_w = gy2.t() @ x2
                grad_w_sq = grad_w.pow(2)
                acc = acc_h_full.get(name)
                if acc is None:
                    acc = torch.zeros(
                        grad_w.shape[0], grad_w.shape[1],
                        dtype=torch.float32, device="cpu")
                    acc_h_full[name] = acc
                acc.add_(grad_w_sq.float().to("cpu"))
                acc_stats[name]["h_trace_raw"] += float(grad_w_sq.sum().item())
                w = mod_ref.weight
                if w is not None and not w.is_meta:
                    acc_stats[name]["h_w2_sum_raw"] += float(
                        (grad_w_sq * w.detach().pow(2)).sum().item())
                acc_stats[name]["n_tokens_seen"] += x2.size(0)
            return hook

        for fqn in tracked_here:
            mod = model.get_submodule(fqn)
            if not isinstance(mod, nn.Linear):
                continue
            w = mod.weight
            if w.is_meta:
                continue
            acc_stats[fqn] = {
                "h_trace_raw": 0.0,
                "h_w2_sum_raw": 0.0,
                "w_max_abs": float(w.detach().abs().max().item()),
                "w_norm_sq": float(w.detach().pow(2).sum().item()),
                "n_params": int(w.numel()),
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "n_tokens_seen": 0,
                "route_prob": None,
                "router_path": None,
                "expert_id": None,
            }
            for p in mod.parameters():
                p.requires_grad_(True)
            handles.append(mod.register_forward_hook(make_fwd(fqn)))
            handles.append(mod.register_full_backward_hook(make_bwd(fqn, mod)))

        packed_grad_acc: dict[str, float] = {}
        packed_full_acc: dict[str, torch.Tensor] | None = (
            {} if h_detail_dir is not None else None)
        # Reverse-sweep visits every layer (gradient chain-rule needs
        # all of them), but Fisher stats should only be recorded for
        # layers in this shard's scope. Skip the packed-expert install
        # + stats merge when L is out-of-scope; backward still flows.
        layer_in_scope = bool(tracked_here) or bool(
            inc.search(f"{layers_prefix}{L}."))
        packed_meta = install_packed_expert_hooks(
            layers[L], accumulator=packed_grad_acc,
            full_accumulator=packed_full_acc,
        ) if layer_in_scope else {}
        layer_prefix = f"{layers_prefix}{L}."
        layer_packed_handles: list = []
        for key, md in packed_meta.items():
            full_key = f"{layer_prefix}{key}"
            experts_qname_rel = md["_packed_experts_module"]
            md["_packed_experts_module"] = f"{layer_prefix}{experts_qname_rel}"
            acc_stats[full_key] = md
            # Capture activations for the packed-experts module so the
            # allocator can use the same input cache as nn.Linear entries.
            if cache_dir is not None:
                try:
                    experts_mod = layers[L].get_submodule(experts_qname_rel)
                except AttributeError:
                    experts_mod = None
                if experts_mod is not None:
                    experts_full = f"{layer_prefix}{experts_qname_rel}"

                    def _exp_fwd(_mod, inp, _out,
                                 _q=experts_full, _rows=packed_act_rows,
                                 _snaps=packed_act_snaps,
                                 _lim=input_rows_limit):
                        x = inp[0] if isinstance(inp, tuple) else inp
                        if isinstance(x, torch.Tensor):
                            need = _lim - _rows[_q]
                            if need > 0:
                                flat = x.detach().reshape(-1, x.size(-1))
                                if flat.size(0) > need:
                                    idx = torch.randperm(flat.size(0), device=flat.device)[:need]
                                    flat = flat.index_select(0, idx)
                                _snaps[_q].append(flat.to("cpu"))
                                _rows[_q] += flat.size(0)

                    layer_packed_handles.append(
                        experts_mod.register_forward_hook(_exp_fwd))

        # Forward + backward for this layer with the full batch.
        x_in = activations_cpu[L].to(device).to(dtype).detach().requires_grad_(True)
        bwd_t0 = time.time()
        out = _call_layer(
            layers[L], x_in,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
        )
        out.backward(grad_out.to(device))
        bwd_s = time.time() - bwd_t0

        for local_key, raw in packed_grad_acc.items():
            full_key = f"{layer_prefix}{local_key}"
            if full_key in acc_stats:
                acc_stats[full_key]["h_trace_raw"] += float(raw)
                acc_stats[full_key]["n_tokens_seen"] = \
                    acc_stats[full_key].get("n_tokens_seen", 0) + x_in.size(0) * x_in.size(1)

        grad_out = x_in.grad.detach().clone().cpu()

        for h in handles:
            h.remove()
        for h in layer_packed_handles:
            h.remove()
        for fqn, s in acc_stats.items():
            prev = merged_stats.get(fqn)
            if prev is None:
                merged_stats[fqn] = dict(s)
            else:
                prev["h_trace_raw"] += s.get("h_trace_raw", 0.0)
                prev["h_w2_sum_raw"] += s.get("h_w2_sum_raw", 0.0)
                prev["n_tokens_seen"] += s.get("n_tokens_seen", 0)
        for fqn, h in acc_h_full.items():
            if fqn in merged_h_full:
                merged_h_full[fqn].add_(h)
            else:
                merged_h_full[fqn] = h.clone()
        if packed_full_acc:
            detail_dir = Path(h_detail_dir)
            detail_dir.mkdir(parents=True, exist_ok=True)
            for local_key, tensor in packed_full_acc.items():
                full_key = f"{layer_prefix}{local_key}"
                fname = re.sub(r"[^A-Za-z0-9_-]", "__", full_key) + ".pt"
                torch.save({"H": tensor, "name": full_key},
                           detail_dir / fname)
            packed_full_acc.clear()

        ctx.unload(L)
        del x_in, out, saved_inputs, acc_stats, acc_h_full, handles
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if L % 8 == 0 or L == 0 or L == num_layers - 1:
            print(f"[incremental] bwd L{L:02d}  src={src}  load={load_s:.2f}s  "
                  f"bwd={bwd_s:.2f}s", flush=True)

    print(f"[incremental] phase-3 reverse sweep: {time.time()-t_phase:.1f}s  "
          f"{ctx.layer_cache.summary()}", flush=True)

    del activations_cpu, grad_at_tail, grad_out

    # ---- Finalize ----
    for s in merged_stats.values():
        tokens = max(s.get("n_tokens_seen", 1), 1)
        s["h_trace"] = s.get("h_trace_raw", 0.0) / tokens
        s["h_w2_sum"] = s.get("h_w2_sum_raw", 0.0) / tokens

    detail_dir = Path(h_detail_dir) if h_detail_dir else None
    if detail_dir is not None:
        detail_dir.mkdir(parents=True, exist_ok=True)
        for fqn, h in merged_h_full.items():
            fname = re.sub(r"[^A-Za-z0-9_-]", "__", fqn) + ".pt"
            torch.save({"H": h, "name": fqn}, detail_dir / fname)

    # Flush activation snapshots.
    if cache_dir is not None:
        sub = re.compile(r"[^A-Za-z0-9_-]")
        for name, snaps in activation_snaps.items():
            if not snaps:
                continue
            X = torch.cat(snaps, dim=0).to(torch.bfloat16).contiguous()
            fname = sub.sub("__", name) + ".pt"
            torch.save({"inputs": X, "name": name}, cache_dir / fname)
        for experts_qname, snaps in packed_act_snaps.items():
            if not snaps:
                continue
            X = torch.cat(snaps, dim=0).to(torch.bfloat16).contiguous()
            fname = sub.sub("__", experts_qname) + ".pt"
            torch.save({"inputs": X, "name": experts_qname}, cache_dir / fname)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "stats": merged_stats,
            "router_counts": {},
            "router_totals": {},
            "expert_info": {},
            "meta": {
                "model": model_path,
                "dataset": dataset_name,
                "nsamples": int(calib.size(0)),
                "seqlen": seqlen,
                "dtype": dtype_name,
                "device_map": "streaming-layerwise",
                "execution_device": str(device),
                "top_k": top_k,
                "importance_weighting": importance_weighting,
                "activation_cache_dir": activation_cache_dir,
                "linear_include": linear_include,
                "linear_exclude": linear_exclude,
            },
        }, f)
    print(f"[incremental] wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# MTP shard runner — synthesize MtpModule, load `mtp.*` weights from
# safetensors, run forward+backward, collect Fisher. The body model has
# to be forwarded once (streaming phase-1) to produce final hidden states;
# no phase-3 reverse over body is needed since MTP gradients don't propagate
# back into the body.
# ---------------------------------------------------------------------------
def _run_mtp_streaming_shard(
    ctx: StreamingContext,
    *,
    calib: torch.Tensor,
    linear_include: str,
    linear_exclude: str,
    importance_weighting: bool,
    activation_cache_dir: str | None,
    output_path: str,
    dataset_name: str,
    dtype_name: str,
    seqlen: int,
    model_path: str,
    prefetch_lookahead: int = 3,
):
    # Lazy import to avoid depending on transformers subpath at module load.
    from .mtp_module import MtpModule, _load_into_mtp, _load_mtp_state_dict

    device = ctx.device
    dtype = ctx.dtype
    model = ctx.model
    base_model = ctx.base_model
    layers = ctx.layers
    num_layers = ctx.num_layers

    # --- Body forward to final hidden states, streaming layer-by-layer ---
    tokens_in_sample = calib.size(-1)
    batch_size = calib.size(0)
    ids = calib.to(device)
    position_ids = torch.arange(tokens_in_sample, device=device).unsqueeze(0)
    causal_mask = _make_causal_mask(tokens_in_sample, device, dtype)

    t_phase = time.time()
    with torch.no_grad():
        inputs_embeds = base_model.embed_tokens(ids).to(dtype)
        hidden = inputs_embeds
    position_embeddings = _compute_position_embeddings(
        base_model, hidden, position_ids)

    prefetch_depth = prefetch_lookahead
    for d in range(prefetch_depth):
        ctx.schedule_prefetch(d)
    for L in range(num_layers):
        load_t0 = time.time()
        src = ctx.install(L)
        ctx.schedule_prefetch(L + prefetch_depth)
        load_s = time.time() - load_t0
        with torch.no_grad():
            hidden = _call_layer(
                layers[L], hidden,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
        ctx.unload(L)
        if L % 8 == 0 or L == num_layers - 1:
            print(f"[incremental/mtp] body fwd L{L:02d}  src={src}  "
                  f"load={load_s:.2f}s", flush=True)
    with torch.no_grad():
        body_final = base_model.norm(hidden)
    print(f"[incremental/mtp] body forward: {time.time()-t_phase:.1f}s",
          flush=True)

    # Hold these on CPU between body and MTP so we free GPU memory
    # before allocating the MTP module.
    inputs_embeds_cpu = inputs_embeds.detach().cpu()
    body_final_cpu = body_final.detach().cpu()
    del inputs_embeds, hidden, body_final
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Synthesize MTP module, load its weights from safetensors ---
    text_config = model.config
    inner_mtp = MtpModule(text_config)
    mtp_wrapper = nn.Module()
    mtp_wrapper.add_module("mtp", inner_mtp)
    mtp_wrapper.to(device=device, dtype=dtype)
    mtp_wrapper.eval()

    raw = _load_mtp_state_dict(model_path)
    missing, extra = _load_into_mtp(inner_mtp, raw)
    loaded = len(raw) - len(missing)
    print(f"[incremental/mtp] loaded {loaded}/{len(raw)} mtp weights "
          f"(missing={len(missing)}, module_params_unset={len(extra)})",
          flush=True)
    if missing:
        print(f"[incremental/mtp] unmatched checkpoint keys (first 5): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)

    # Freeze every leaf; Fisher hooks capture ||grad_w||² without
    # retaining leaf .grads.
    for p in mtp_wrapper.parameters():
        p.requires_grad_(False)

    tracked = [n for n, m in mtp_wrapper.named_modules()
               if isinstance(m, nn.Linear) and not re.search(r"mlp\.gate$", n)]
    print(f"[incremental/mtp] tracking {len(tracked)} MTP Linears", flush=True)

    expert_info_all = discover_moe_structure(mtp_wrapper)
    expert_info = {k: v for k, v in expert_info_all.items() if k in tracked}
    top_k = read_top_k(mtp_wrapper, default=2)

    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    acc = FisherAccumulator(mtp_wrapper, tracked, expert_info, cache_dir)

    # lm_head lives on the body model (resident).
    lm_head = model.get_output_embeddings()
    assert isinstance(lm_head, nn.Linear), "lm_head must be Linear for MTP CE"

    from transformers.masking_utils import create_causal_mask

    t_fwd = t_bwd = 0.0
    for i in range(calib.size(0)):
        ids_i = calib[i:i + 1].to(device)
        t0 = time.time()
        embed_i = inputs_embeds_cpu[i:i + 1].to(device, dtype=dtype)
        body_hidden_i = body_final_cpu[i:i + 1].to(device, dtype=dtype)

        shifted_embed = embed_i[:, 1:-1, :].contiguous()
        shifted_hidden = body_hidden_i[:, :-2, :].contiguous()
        target_ids = ids_i[:, 2:].contiguous()
        B, T2, _ = shifted_embed.shape
        trimmed_pos_ids = torch.arange(T2, device=device).view(1, T2).expand(B, T2)
        causal_mask_t2 = create_causal_mask(
            config=text_config,
            inputs_embeds=shifted_embed,
            attention_mask=None,
            past_key_values=None,
            position_ids=trimmed_pos_ids,
        )
        rot_pos = trimmed_pos_ids.view(1, B, T2).expand(3, B, T2)
        pos_emb_t2 = base_model.rotary_emb(shifted_embed, rot_pos)

        shifted_hidden = shifted_hidden.detach().requires_grad_(True)
        shifted_embed = shifted_embed.detach().requires_grad_(True)

        inner_mtp.train()
        out_hidden = inner_mtp(
            inputs_embeds=shifted_embed,
            body_hidden_states=shifted_hidden,
            position_embeddings=pos_emb_t2,
            causal_mask=causal_mask_t2,
            position_ids=trimmed_pos_ids,
        )
        logits = lm_head(out_hidden)
        t_fwd += time.time() - t0

        t0 = time.time()
        lp = F.log_softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
        gather = -lp.gather(1, target_ids.reshape(-1, 1)).squeeze(1)
        if importance_weighting:
            with torch.no_grad():
                mean = float(gather.mean().item())
            w = (gather.detach() / max(mean, 1e-6)).clamp(0.25, 4.0)
            loss = (gather * w).sum()
        else:
            loss = gather.sum()
        loss.backward()
        t_bwd += time.time() - t0

        n_tok = max(int(gather.numel()), 1)
        mean_loss = float(loss.detach().item()) / n_tok
        print(f"[incremental/mtp] sample {i+1}/{calib.size(0)} "
              f"loss={mean_loss:.3f} fwd_avg={t_fwd/(i+1):.2f}s "
              f"bwd_avg={t_bwd/(i+1):.2f}s", flush=True)

        del out_hidden, logits, loss, gather
        acc._saved_inputs.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    acc.finalize(tracker=None)
    acc.remove_hooks()

    renamed = dict(acc.stats)
    expert_info_renamed = dict(expert_info)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "stats": renamed,
            "router_counts": {},
            "router_totals": {},
            "expert_info": expert_info_renamed,
            "meta": {
                "model": model_path,
                "dataset": dataset_name,
                "nsamples": int(calib.size(0)),
                "seqlen": seqlen,
                "dtype": dtype_name,
                "device_map": "streaming-layerwise",
                "execution_device": str(device),
                "top_k": top_k,
                "importance_weighting": importance_weighting,
                "activation_cache_dir": activation_cache_dir,
                "linear_include": linear_include,
                "linear_exclude": linear_exclude,
                "mtp_probe": True,
                "mtp_objective": "CE(lm_head(MTP(embed_{t+1}, body_hidden_t)), ids_{t+2})",
            },
        }, f)
    print(f"[incremental/mtp] wrote {output_path}", flush=True)

    # Free MTP before the next shard.
    del mtp_wrapper, inner_mtp, acc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ultrachat_200k")
    ap.add_argument("--nsamples", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output", required=True)
    ap.add_argument("--activation-cache-dir", required=True)
    ap.add_argument("--work-dir", required=True,
                    help="Stores shard logs/pickles; safe to resume.")
    ap.add_argument("--layers-per-shard", type=int, default=1)
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=None)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", action="store_false",
                    dest="gradient_checkpointing")
    ap.add_argument("--importance-weighting", action="store_true", default=True)
    ap.add_argument("--no-importance-weighting", action="store_false",
                    dest="importance_weighting")
    ap.add_argument("--include-mtp", action="store_true", default=True,
                    help="Probe MTP layers (`mtp.layers.X.*`).")
    ap.add_argument("--no-include-mtp", action="store_false", dest="include_mtp")
    ap.add_argument("--include-visual", action="store_true", default=True,
                    help="Probe visual encoder blocks (`model.visual.blocks.X.*`).")
    ap.add_argument("--no-include-visual", action="store_false", dest="include_visual")
    ap.add_argument("--include-lm-head", action="store_true", default=True,
                    help="Probe lm_head (`^lm_head$`).")
    ap.add_argument("--no-include-lm-head", action="store_false", dest="include_lm_head")
    ap.add_argument("--h-detail-dir", default=None,
                    help="If set, write per-Linear full Fisher diagonal "
                         "(shape [out, in]) and per-packed-expert Fisher "
                         "(shape [E, M]) as .pt files in this directory. "
                         "measure_quant_cost reads them to compute the full "
                         "per-weight delta loss = 0.5 * <H, MSE_W> instead "
                         "of the scalar proxy. Omit to keep the legacy "
                         "scalar path.")
    ap.add_argument("--prefetch-lookahead", type=int, default=3,
                    help="Number of layers to queue ahead in the disk "
                         "prefetch pool. Bump up when per-layer compute "
                         "time >> per-layer disk read time (e.g. batch>=32).")
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
    print(f"[incremental] shard regexes: {len(shard_regexes)} total "
          f"(body={len(body_regexes[first_shard:last_shard])}, extras={len(extra)})",
          flush=True)

    work_dir = Path(args.work_dir)
    shard_dir = work_dir / "shards"
    log_dir = work_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.activation_cache_dir).mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    exec_device = device  # streaming path runs on the requested device directly

    # Skip setup + calibration if every shard is reusable. Loading the
    # model+tokenizer is expensive — if the run is a no-op we want to
    # avoid paying that cost.
    shard_paths = [shard_dir / f"probe_shard_{i:03d}.pkl" for i in range(len(shard_regexes))]
    expected_metas = [
        _expected_probe_shard_meta(
            args,
            linear_include=linear_include,
            shard_idx=i,
            activation_cache_dir=args.activation_cache_dir,
        )
        for i, linear_include in enumerate(shard_regexes)
    ]
    all_reusable = all(
        shard_paths[i].exists()
        and probe_shard_is_reusable(shard_paths[i], expected_metas[i])
        for i in range(len(shard_regexes))
    )

    ctx: StreamingContext | None = None
    tokenizer = None
    calib: torch.Tensor | None = None

    def _ensure_ready():
        nonlocal ctx, tokenizer, calib
        if ctx is not None:
            return
        from transformers import AutoTokenizer
        staged = stage_text_only(args.model)
        tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
        calib = load_calibration(tokenizer, args.dataset, args.nsamples, args.seqlen)
        print(f"[incremental] calibration ready: {tuple(calib.shape)}", flush=True)
        offload_folder = str(work_dir / "streaming_offload")
        ctx = _build_streaming_context(
            args.model,
            device=device,
            dtype=dtype,
            offload_folder=offload_folder,
            log_prefix="[incremental]",
        )

    try:
        if not all_reusable:
            _ensure_ready()

        for shard_idx, linear_include in enumerate(shard_regexes):
            shard_path = shard_paths[shard_idx]
            expected_meta = expected_metas[shard_idx]
            if shard_path.exists() and probe_shard_is_reusable(shard_path, expected_meta):
                print(f"[incremental] reuse shard {shard_idx}: {shard_path}",
                      flush=True)
                continue
            if shard_path.exists():
                print(f"[incremental] stale shard {shard_idx}: "
                      f"recomputing {shard_path}", flush=True)
            kind = _classify_shard(linear_include)
            print(f"[incremental] shard {shard_idx} ({kind}): "
                  f"include={linear_include!r}", flush=True)
            _ensure_ready()

            if kind == "body":
                _run_body_streaming_shard(
                    ctx,
                    calib=calib,
                    linear_include=linear_include,
                    linear_exclude=(
                        r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|"
                        r"block_sparse_moe\.gate$)"
                    ),
                    importance_weighting=args.importance_weighting,
                    activation_cache_dir=args.activation_cache_dir,
                    h_detail_dir=args.h_detail_dir,
                    output_path=str(shard_path),
                    dataset_name=args.dataset,
                    dtype_name=args.dtype,
                    seqlen=args.seqlen,
                    model_path=args.model,
                    prefetch_lookahead=args.prefetch_lookahead,
                )
            elif kind == "mtp":
                _run_mtp_streaming_shard(
                    ctx,
                    calib=calib,
                    linear_include=linear_include,
                    linear_exclude=(
                        r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|"
                        r"block_sparse_moe\.gate$)"
                    ),
                    importance_weighting=args.importance_weighting,
                    activation_cache_dir=args.activation_cache_dir,
                    output_path=str(shard_path),
                    dataset_name=args.dataset,
                    dtype_name=args.dtype,
                    seqlen=args.seqlen,
                    model_path=args.model,
                    prefetch_lookahead=args.prefetch_lookahead,
                )
            elif kind == "lm_head":
                # The lm_head Fisher is collected naturally as a body
                # shard with the regex pinned to `^lm_head$`: phase-2 runs
                # lm_head's forward+backward, and we install a Fisher hook
                # on it. Reuse the body runner for that.
                _run_body_streaming_shard(
                    ctx,
                    calib=calib,
                    linear_include=linear_include,
                    linear_exclude=(
                        r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|"
                        r"block_sparse_moe\.gate$)"
                    ),
                    importance_weighting=args.importance_weighting,
                    activation_cache_dir=args.activation_cache_dir,
                    h_detail_dir=args.h_detail_dir,
                    output_path=str(shard_path),
                    dataset_name=args.dataset,
                    dtype_name=args.dtype,
                    seqlen=args.seqlen,
                    model_path=args.model,
                    prefetch_lookahead=args.prefetch_lookahead,
                )
            else:
                # visual blocks are stripped by text-only staging, so the
                # streaming body never installs them. Emit an empty pickle
                # so the shard slot stays in the merged output with matching
                # metadata (and future visual-aware runs can fill it in).
                print(f"[incremental] skip shard {shard_idx} ({kind}): "
                      f"not supported in streaming path", flush=True)
                Path(shard_path).parent.mkdir(parents=True, exist_ok=True)
                with open(shard_path, "wb") as f:
                    pickle.dump({
                        "stats": {},
                        "router_counts": {},
                        "router_totals": {},
                        "expert_info": {},
                        "meta": {
                            "model": args.model,
                            "dataset": args.dataset,
                            "nsamples": args.nsamples,
                            "seqlen": args.seqlen,
                            "dtype": args.dtype,
                            "device_map": "streaming-layerwise",
                            "execution_device": str(device),
                            "importance_weighting": args.importance_weighting,
                            "activation_cache_dir": args.activation_cache_dir,
                            "linear_include": linear_include,
                            "linear_exclude": (
                                r"(?:mlp\.gate$|mlp\..*gate$|"
                                r"\.router(?:$|\.)|block_sparse_moe\.gate$)"
                            ),
                            "shard_kind": kind,
                        },
                    }, f)
            annotate_probe_shard(shard_path, expected_meta)
            if exec_device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if ctx is not None:
            ctx.shutdown()

    merge_probe_pickles(shard_paths, Path(args.output))
    print(f"[incremental] wrote merged probe to {args.output}", flush=True)


if __name__ == "__main__":
    main()
