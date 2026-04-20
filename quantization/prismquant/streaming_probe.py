"""Layer-wise streaming probe for models that don't fit in RAM.

Preserves the exact Fisher-diagonal math from `sensitivity_probe.py`:

    s(w) = E[(∂L/∂w)²]

but chain-rules the backward one decoder layer at a time, carrying the
gradient at each layer boundary forward to the previous step. The
per-layer H_full accumulation, packed-expert grad-norm capture, and
stats schema are identical to the monolithic probe — downstream cost
measurement / allocator consume the outputs unchanged.

Orchestration (one sample at a time):
  1. Materialize embed + norm + lm_head + rotary once (tiny).
  2. Forward sweep: for L in [0..N): load layer L, forward (no_grad),
     cache output activation on CPU, unload layer L.
  3. Loss + head backward: compute CE loss; backward through lm_head /
     norm to get `grad_at_final_hidden`.
  4. Backward sweep: for L in reversed(range(N)): load L, install
     Fisher hooks, forward + backward for this layer only,
     carry `grad` → `grad_at_input` for the next step, unload L.

Memory on Spark (121 GB total):
  - Always resident: embed + norm + lm_head (~3 GB for Qwen3.5-122B).
  - Current layer: ~6 GB BF16 weights + ~1 GB activations/grads.
  - Linux page cache absorbs 30–80 GB of safetensors reads so the
    backward sweep's reverse reads often hit RAM, not disk.
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import re
import time
from pathlib import Path

# Must be set before the cuda allocator initializes. On Spark's UMA,
# cuda and cpu share one LPDDR5X pool; without `expandable_segments`
# the caching allocator hoards freed blocks, causing the OS to swap
# while torch's bookkeeping still thinks it has headroom.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights

from .layer_streaming import (
    LayerCache,
    _build_install_resolver,
    _build_weight_map,
    _call_layer,
    _compute_position_embeddings,
    _embed_prefix,
    _fast_install,
    _get_layer_list,
    _get_rotary,
    _head_prefixes,
    _install_cached_tensors,
    _make_causal_mask,
    _materialize,
    _read_layer_to_cpu,
    _read_layer_to_device,
    _resolve_base_prefix,
    _unload,
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


def run_streaming_probe(
    model_path: str,
    calib: torch.Tensor,
    *,
    output_path: str,
    dataset_name: str,
    dtype: torch.dtype,
    dtype_name: str,
    device: torch.device,
    linear_include: str,
    linear_exclude: str,
    importance_weighting: bool,
    activation_cache_dir: str | None,
    h_detail_dir: str | None,
    seqlen: int,
    prefetch_lookahead: int = 3,
):
    """Run the streaming probe. Writes a pickle in the same schema as
    `sensitivity_probe.run_probe_pass`."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate.hooks import remove_hook_from_module

    staged = stage_text_only(model_path)
    config = AutoConfig.from_pretrained(staged, trust_remote_code=True)

    # Discover the layer count + naming convention WITHOUT loading any
    # weights, so we can build an explicit device_map before calling
    # `from_pretrained`. Using `init_empty_weights` costs nothing on meta.
    with init_empty_weights():
        skeleton = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True)
    skel_base, skel_layers = _get_layer_list(skeleton)
    base_prefix = _resolve_base_prefix(skeleton, skel_base)
    num_layers = len(skel_layers)
    del skeleton, skel_base, skel_layers

    layers_prefix = f"{base_prefix}.layers." if base_prefix else "layers."
    head_prefixes = _head_prefixes(None, base_prefix)

    # Explicit device_map: head/embed/norm/rotary resident on the exec
    # device; every decoder layer to `disk` (forces accelerate to write
    # the layer weights into `offload_folder` at load time). We then
    # detach accelerate's auto-load hooks from layers and stream them
    # back in manually from the original safetensors via `_materialize`.
    # `offload_folder` must exist and have space for ~half the model.
    base = base_prefix if base_prefix else ""
    device_map: dict[str, object] = {}
    resident_device = 0 if device.type == "cuda" else "cpu"
    for pfx in (f"{base}.embed_tokens" if base else "embed_tokens",
                f"{base}.norm" if base else "norm",
                f"{base}.rotary_emb" if base else "rotary_emb",
                "lm_head"):
        device_map[pfx] = resident_device
    for L in range(num_layers):
        device_map[f"{base}.layers.{L}" if base else f"layers.{L}"] = "disk"

    offload_folder = os.path.join(
        os.path.dirname(os.path.abspath(output_path)), "streaming_offload")
    os.makedirs(offload_folder, exist_ok=True)

    t0 = time.time()
    print(f"[streaming] base_prefix={base_prefix!r}  layers={num_layers}  "
          f"head_prefixes={head_prefixes}", flush=True)
    print(f"[streaming] loading head resident on {resident_device}, "
          f"{num_layers} layers to disk offload={offload_folder} ...",
          flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        staged,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder=offload_folder,
        offload_buffers=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_model, layers = _get_layer_list(model)

    # Drop accelerate's auto-load hooks from decoder layers so WE
    # control when each layer is materialized. Resident head/embed/
    # norm/rotary keep their hooks (no harm; they're fully loaded).
    for L in range(num_layers):
        remove_hook_from_module(layers[L], recurse=True)
    # Collapse layer params/buffers back to meta so `_materialize` can
    # cleanly repopulate them on demand. After `remove_hook_from_module`,
    # layer tensors are empty shells pointing at offload files;
    # `set_module_tensor_to_device(..., "meta")` resets the slot so a
    # subsequent fresh tensor can be written without dtype/device clash.
    for L in range(num_layers):
        layer_pref = f"{layers_prefix}{L}."
        _unload(model, [layer_pref])

    weight_shard, weight_ckpt = _build_weight_map(model_path)
    print(f"[streaming] model ready in {time.time()-t0:.1f}s", flush=True)

    # Per-layer fast-install resolver. Built once; used for every
    # Phase-1 + Phase-3 install. See `_build_install_resolver`.
    print(f"[streaming] building install resolvers for {num_layers} layers ...",
          flush=True)
    t_res = time.time()
    install_resolvers = [
        _build_install_resolver(model, f"{layers_prefix}{L}".rstrip("."))
        for L in range(num_layers)
    ]
    total_resolved = sum(len(r) for r in install_resolvers)
    print(f"[streaming] resolvers built: {total_resolved} tensors across "
          f"{num_layers} layers in {time.time()-t_res:.1f}s", flush=True)

    # Per-layer hook-spec planning: tracked Linear names under each layer.
    inc = re.compile(linear_include)
    exc = re.compile(linear_exclude)
    layer_linear_names: list[list[str]] = []
    # Walk the model once (everything is on meta but named_modules works
    # because the graph structure is initialized).
    all_linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    all_tracked = [n for n in all_linears
                   if inc.search(n) and not exc.search(n)]
    for L in range(num_layers):
        pref = f"{layers_prefix}{L}."
        layer_linear_names.append([n for n in all_tracked if n.startswith(pref)])
    total_tracked = sum(len(x) for x in layer_linear_names)
    print(f"[streaming] tracking {total_tracked} Linears "
          f"({total_tracked/num_layers:.1f} avg per layer)", flush=True)

    expert_info = {}  # populated opportunistically per layer
    top_k = read_top_k(model, default=2)

    # Merged outputs across layers/samples — match the stats schema of
    # sensitivity_probe's FisherAccumulator so downstream tooling works.
    merged_stats: dict[str, dict] = {}
    merged_h_full: dict[str, torch.Tensor] = {}

    tokens_in_sample = calib.size(-1)
    batch_size = calib.size(0)

    # ---- Single-batch flow. Load each layer exactly ONCE for Phase 1
    #      (forward) and ONCE for Phase 3 (backward). Fisher diagonal
    #      accumulates identically to the per-sample flow because
    #      sum-reduction CE is linear across the batch. ----
    ids = calib.to(device)  # [N, T]
    position_ids = torch.arange(tokens_in_sample, device=device).unsqueeze(0)  # [1, T]

    # ---- Layer cache + background disk prefetcher ----
    # Budget: free RAM minus headroom for:
    #   (a) Phase-1 activation stash `activations_cpu`
    #       (N * T * hidden * 2 bytes/layer): ~19 GB at N=32 T=2048
    #       hidden=3072 for 48 layers, ~10 GB at N=16.
    #   (b) transient prefetch working tensors (~6 GB at lookahead=3);
    #   (c) cuda working set for the current layer's fwd/bwd
    #       (~7-15 GB depending on batch);
    #   (d) Phase-3 `merged_h_full` accumulators, ~20 GB CPU fp32 for
    #       a 122B-scale MoE: this grows monotonically as Phase 3 walks
    #       layers and is NOT freed between layers;
    #   (e) process + Python + cuda kernel scratch.
    # 75 GB covers all of that with slack. Smaller values crash Phase 3.
    import psutil
    free_bytes = psutil.virtual_memory().available
    cache_bytes = max(int(free_bytes) - 75 * 1024 ** 3, 8 * 1024 ** 3)
    layer_cache = LayerCache(max_bytes=cache_bytes)
    print(f"[streaming] layer cache budget={cache_bytes/(1024**3):.1f} GB "
          f"(free={free_bytes/(1024**3):.1f} GB)", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    import threading
    # One worker keeps a sequential disk queue (the NVMe services reads
    # fastest when we don't contend on the same bytes). The `depth`
    # parameter below controls how many layers we queue ahead of main.
    prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
    inflight: dict[int, "Future"] = {}
    inflight_lock = threading.Lock()

    def _prefetch_worker(L: int):
        prefix = f"{layers_prefix}{L}."
        tensors = _read_layer_to_device(
            prefix, weight_shard, weight_ckpt, dtype, device)
        layer_cache.put(L, tensors)
        with inflight_lock:
            inflight.pop(L, None)
        return tensors

    def _schedule_prefetch(L: int):
        """Queue layer L for async read if it's not already cached or
        already in flight. Returns the future (or None)."""
        if L < 0 or L >= num_layers:
            return None
        if layer_cache.peek(L):
            return None
        with inflight_lock:
            if L in inflight:
                return inflight[L]
            fut = prefetch_pool.submit(_prefetch_worker, L)
            inflight[L] = fut
            return fut

    def _ensure_loaded(L: int) -> tuple[dict[str, torch.Tensor], str]:
        """Return the cached tensor dict for layer L. Hot-path: cache hit.
        Warm path: wait on an in-flight prefetch future (prefetch still
        reading when main arrives). Cold path: synchronous read."""
        cached = layer_cache.get(L)
        if cached is not None:
            return cached, "hot"
        with inflight_lock:
            fut = inflight.get(L)
        if fut is not None:
            fut.result()
            cached = layer_cache.get(L)
            if cached is not None:
                return cached, "wait"
        prefix = f"{layers_prefix}{L}."
        tensors = _read_layer_to_device(
            prefix, weight_shard, weight_ckpt, dtype, device)
        layer_cache.put(L, tensors)
        return tensors, "cold"

    # How many layers we try to have in flight / cached ahead of main.
    # At batch=32, compute/layer ~= 8s and disk/layer ~= 2s → we can fit
    # ~3-4 reads inside one compute step.
    prefetch_depth = prefetch_lookahead

    # ---- Phase 1: streaming forward, cache activations on CPU ----
    t_phase = time.time()
    with torch.no_grad():
        embed_mod = base_model.embed_tokens
        hidden = embed_mod(ids).to(dtype)  # [N, T, H]
    causal_mask = _make_causal_mask(tokens_in_sample, device, dtype)
    position_embeddings = _compute_position_embeddings(
        base_model, hidden, position_ids)
    print(f"[streaming] batched probe: N={batch_size}  T={tokens_in_sample}  "
          f"initial hidden={tuple(hidden.shape)} dtype={hidden.dtype}",
          flush=True)

    # Warm the prefetch queue `prefetch_depth` layers ahead before the
    # main loop starts, so the worker is already busy when we arrive at
    # L=0.
    for d in range(prefetch_depth):
        _schedule_prefetch(d)
    activations_cpu: list[torch.Tensor] = [hidden.detach().cpu()]
    for L in range(num_layers):
        load_t0 = time.time()
        tensors, src = _ensure_loaded(L)
        _fast_install(install_resolvers[L], tensors, device, model=model)
        # Maintain depth-deep lookahead.
        _schedule_prefetch(L + prefetch_depth)
        load_s = time.time() - load_t0
        if L == 0:
            ln_w = layers[L].input_layernorm.weight
            child_kinds = [k for k, _ in layers[L].named_children()]
            print(f"[DEBUG] after install L0: src={src} "
                  f"input_layernorm.weight.device={ln_w.device} "
                  f"is_meta={ln_w.is_meta}  children={child_kinds}",
                  flush=True)
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
        _unload(model, [f"{layers_prefix}{L}."])   # params on meta; CPU copy stays in cache
        if L % 8 == 0 or L == num_layers - 1:
            print(f"[streaming] fwd L{L:02d}  src={src}  load={load_s:.2f}s  "
                  f"fwd={fwd_s:.2f}s", flush=True)
    print(f"[streaming] phase-1 forward: {time.time()-t_phase:.1f}s  "
          f"{layer_cache.summary()}", flush=True)

    # ---- Phase 2: final norm + lm_head + loss; grad at last hidden ----
    # Chunk along the sequence dim so we never materialize the full
    # `[N, T, vocab]` logits tensor. At N=32 T=2048 vocab=150k that's
    # ~20 GB per copy — full CE would peak at 60-80 GB (logits +
    # shift_logits + log_softmax + backward), colliding with the cache
    # on UMA. With chunk_T=256 the per-chunk footprint is ~2.4 GB.
    # Also flush the LayerCache between phases: Phase 3 runs in reverse
    # so the retained top-of-stack layers are the first we'll need, but
    # the head-compute peak would OOM before we get there. Prefetch
    # lookahead fills Phase 3 back up within a couple of layers.
    layer_cache.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t_phase = time.time()
    final_hidden = activations_cpu[-1].to(device).to(dtype).requires_grad_(True)
    norm_out = base_model.norm(final_hidden)
    # Detach norm_out and give the detached copy requires_grad so we can
    # call autograd.grad against it per chunk. Accumulating the grad
    # chunk-by-chunk lets each chunk's logits/log_softmax intermediates
    # be freed immediately — that's where the memory saving comes from.
    norm_out_d = norm_out.detach().requires_grad_(True)
    grad_buf = torch.zeros_like(norm_out_d)
    chunk_T = 256
    N, T, _ = norm_out_d.shape
    # Two passes if importance-weighted: first pass builds the global
    # per-token CE mean (needed for the weight denominator), second
    # pass does the weighted backward. Non-IW path runs one pass.
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
    print(f"[streaming] phase-2 loss+head bwd: {time.time()-t_phase:.1f}s",
          flush=True)

    # ---- Phase 3: reverse sweep, per-layer Fisher collection ----
    t_phase = time.time()
    grad_out = grad_at_tail
    # Warm the reverse prefetch queue `prefetch_depth` layers ahead.
    for d in range(prefetch_depth):
        _schedule_prefetch(num_layers - 1 - d)
    for L in reversed(range(num_layers)):
        load_t0 = time.time()
        tensors, src = _ensure_loaded(L)
        _fast_install(install_resolvers[L], tensors, device, model=model)
        _schedule_prefetch(L - prefetch_depth)
        load_s = time.time() - load_t0

        # Install Fisher hooks on just this layer's tracked Linears.
        tracked_here = layer_linear_names[L]
        acc_h_full: dict[str, torch.Tensor] = {}
        acc_stats: dict[str, dict] = {}
        saved_inputs: dict[str, torch.Tensor] = {}
        handles: list = []

        def make_fwd(name):
            def hook(module, inp, out):
                x = inp[0] if isinstance(inp, tuple) else inp
                saved_inputs[name] = x.detach()
            return hook

        def make_bwd(name, mod_ref):
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
        # Full per-weight Fisher accumulator for packed-expert params.
        # Key → `[E, out, in]` fp32 CPU tensor. At 122B MoE this is ~5 GB
        # per packed param (two per layer), released after we flush it to
        # disk below. For larger models (e.g. 397B-A17B), the same
        # streaming pattern holds — we only ever keep one layer's worth
        # of packed-expert Fishers resident.
        packed_full_acc: dict[str, torch.Tensor] | None = (
            {} if h_detail_dir is not None else None)
        packed_meta = install_packed_expert_hooks(
            layers[L], accumulator=packed_grad_acc,
            full_accumulator=packed_full_acc)
        layer_prefix = f"{layers_prefix}{L}."
        for key, md in packed_meta.items():
            full_key = f"{layer_prefix}{key}"
            md["_packed_experts_module"] = f"{layer_prefix}{md['_packed_experts_module']}"
            acc_stats[full_key] = md

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
        # Flush packed-expert full Fisher to disk under the same mangled
        # naming the allocator / cost scripts already use, then free CPU
        # memory. We write per-layer so peak CPU footprint is one layer,
        # not the whole model — essential for models beyond 122B.
        if packed_full_acc:
            detail_dir = Path(h_detail_dir)
            detail_dir.mkdir(parents=True, exist_ok=True)
            for local_key, tensor in packed_full_acc.items():
                full_key = f"{layer_prefix}{local_key}"
                fname = re.sub(r"[^A-Za-z0-9_-]", "__", full_key) + ".pt"
                torch.save({"H": tensor, "name": full_key},
                           detail_dir / fname)
            packed_full_acc.clear()

        _unload(model, [f"{layers_prefix}{L}."])
        del x_in, out, saved_inputs, acc_stats, acc_h_full, handles
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if L % 8 == 0 or L == 0 or L == num_layers - 1:
            print(f"[streaming] bwd L{L:02d}  src={src}  load={load_s:.2f}s  "
                  f"bwd={bwd_s:.2f}s", flush=True)

    print(f"[streaming] phase-3 reverse sweep: {time.time()-t_phase:.1f}s  "
          f"{layer_cache.summary()}", flush=True)
    prefetch_pool.shutdown(wait=True)
    del activations_cpu, grad_at_tail, grad_out

    # ---- Finalize: divide by tokens, write pickle + H-detail files ----
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
    print(f"[streaming] wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ultrachat_200k")
    ap.add_argument("--nsamples", type=int, default=2)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output", required=True)
    ap.add_argument("--activation-cache-dir", default=None)
    ap.add_argument("--h-detail-dir", default=None)
    ap.add_argument("--linear-include", default=".*")
    ap.add_argument("--linear-exclude",
                    default=r"(?:mlp\.gate$|mlp\..*gate$|"
                            r"\.router(?:$|\.)|"
                            r"block_sparse_moe\.gate$)")
    ap.add_argument("--importance-weighting", action="store_true", default=True)
    ap.add_argument("--no-importance-weighting",
                    action="store_false", dest="importance_weighting")
    ap.add_argument("--prefetch-lookahead", type=int, default=3,
                    help="Number of layers to queue ahead in the disk "
                         "prefetch pool. Bump up when per-layer compute "
                         "time >> per-layer disk read time (e.g. batch≥32).")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    # Calibration needs a tokenizer — load it via the staged text-only dir.
    from transformers import AutoTokenizer
    staged = stage_text_only(args.model)
    tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
    calib = load_calibration(tokenizer, args.dataset, args.nsamples, args.seqlen)

    run_streaming_probe(
        model_path=args.model,
        calib=calib,
        output_path=args.output,
        dataset_name=args.dataset,
        dtype=dtype,
        dtype_name=args.dtype,
        device=device,
        linear_include=args.linear_include,
        linear_exclude=args.linear_exclude,
        importance_weighting=args.importance_weighting,
        activation_cache_dir=args.activation_cache_dir,
        h_detail_dir=args.h_detail_dir,
        seqlen=args.seqlen,
        prefetch_lookahead=args.prefetch_lookahead,
    )


if __name__ == "__main__":
    main()
