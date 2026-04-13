#!/usr/bin/env python3
"""
measure_hawq_sensitivity.py — compute per-Linear HAWQ-V3 sensitivity via
Fisher-diagonal approximation.

For each calibration sample, runs forward+backward on the next-token
prediction loss, squares the resulting gradients, and accumulates:
    H_ii ≈ (1/N) * sum_n (∂L_n / ∂w_i)²

This is the Fisher-information approximation to the diagonal of the loss
Hessian. Per HAWQ-V3 (Yao et al. 2021), this diagonal drives the per-Linear
sensitivity scalar used for mixed-precision allocation.

Why Fisher and not true Hessian: HAWQ-V3 uses Hutchinson's estimator on
the full Hessian, which requires second-order backprop (H @ z matvec per
sample). That's more faithful but costs ~5-10× more compute. Fisher equals
Hessian at the loss optimum; for a pretrained model on held-out calibration
data we're near (but not at) optimum, so Fisher is a usable first-order
proxy. We validate the proxy against direct measurement on 0.5B/1.5B and
escalate to Hutchinson-Hessian if correlation is poor.

Per-Linear sensitivity scalars (recorded):
    h_trace: sum(H_ii)           — Fisher trace. Primary HAWQ sensitivity.
    h_w2_sum: sum(H_ii * w_i²)   — expected loss from zeroing each weight.
    w_max_abs: max|w_i|          — dynamic range, needed for noise model.
    w_norm_sq: sum(w_i²)         — weight energy.

The bit-utility curve is computed from these scalars in the allocator:
    ΔKL(L, b) ≈ c · H_trace_L · (w_max_L / (2^(b-1) - 1))² / 12
for uniform symmetric quantization. The curve is analytical, not measured.

Memory: peak ≈ 2×model_size (weights + gradients) + activation memory.
Fits on Spark for models up to ~30B bf16. For larger, use gradient
checkpointing or chunked backward (future work).

Usage:
    python3 measure_hawq_sensitivity.py \\
        --model /path/to/bf16 \\
        --output /tmp/curves/hawq.json \\
        --n-calib-samples 8 --calib-seqlen 256
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from build_rtn_cache import (
    stage_multimodal,
    load_wikitext_calibration,
    iter_quantizable_tensors,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-calib-samples", type=int, default=8)
    parser.add_argument("--calib-seqlen", type=int, default=256)
    parser.add_argument("--skip-small", type=int, default=1000)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to halve activation "
                             "memory during backward. Required for 27B+.")
    parser.add_argument("--cpu", action="store_true",
                        help="Run entirely on CPU. Slow but works for models "
                             "that don't fit on GPU. Uses all available cores.")
    parser.add_argument("--offload-dir", type=str, default=None,
                        help="Directory for disk offloading with accelerate. "
                             "Use for models that don't fit in RAM at bf16 "
                             "(e.g. large fp8 MoE models).")
    parser.add_argument("--cpu-fisher", action="store_true",
                        help="Keep Fisher accumulators on CPU rather than GPU. "
                             "On unified memory this is free and saves ~2× weight "
                             "size of GPU-resident fp32 buffers. Required for 27B+.")
    parser.add_argument("--linear-chunks", type=int, default=1,
                        help="Process Linears in N chunks per sample. Higher N "
                             "reduces peak gradient memory but multiplies backward "
                             "pass count. Default 1 (all at once).")
    args = parser.parse_args()

    t_start = time.time()

    staged, cleanup = stage_multimodal(args.model)
    try:
        from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

        # Check if the original model is actually VL (has vision/text configs
        # AND has language_model prefix in tensor names)
        with open(Path(args.model) / "config.json") as _f:
            _orig_cfg = json.load(_f)
        is_vl = ("vision_config" in _orig_cfg or "text_config" in _orig_cfg) and \
                 any("language_model" in k for k in
                     (json.load(open(Path(args.model) / "model.safetensors.index.json"))
                      ["weight_map"] if (Path(args.model) / "model.safetensors.index.json").exists()
                      else {}))
        device_map = "cpu" if args.cpu else "cuda"

        # For disk offloading (large fp8 MoE models), use accelerate auto
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        if args.offload_dir:
            import os
            os.makedirs(args.offload_dir, exist_ok=True)
            load_kwargs["device_map"] = "auto"
            load_kwargs["offload_folder"] = args.offload_dir
            load_kwargs["offload_state_dict"] = True

        print(f"[hawq] loading {args.model} (VL={is_vl}, device={device_map}, "
              f"offload={args.offload_dir is not None})", flush=True)

        if is_vl:
            # VL models: load full model and extract language_model submodule.
            full_model = AutoModel.from_pretrained(args.model, **load_kwargs)
            # Extract language model
            if hasattr(full_model, 'language_model'):
                model = full_model.language_model
            elif hasattr(full_model, 'model') and hasattr(full_model.model, 'language_model'):
                model = full_model.model.language_model
            else:
                model = full_model
            # Free vision tower to reclaim memory
            for attr in ('visual', 'vision_tower', 'vision_model',
                         'embed_vision', 'multi_modal_projector'):
                for obj in (full_model, getattr(full_model, 'model', None)):
                    if obj is not None and hasattr(obj, attr):
                        delattr(obj, attr)
            gc.collect()
            if not args.cpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[hawq] extracted language_model from VL model", flush=True)
        else:
            # Text-only models: load directly as CausalLM
            model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # Cast fp8 parameters to bf16 (needed for gradient computation).
        # Models loaded without their quantization_config may have fp8 tensors.
        n_cast = 0
        for p in model.parameters():
            if p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                           torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
                p.data = p.data.to(torch.bfloat16)
                n_cast += 1
        if n_cast:
            print(f"[hawq] cast {n_cast} fp8 parameters to bf16", flush=True)

        device = next(model.parameters()).device
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[hawq]   {n_params:,} params on {device}", flush=True)

        # Enable gradients only on quantizable tensors. This halves the
        # gradient-memory footprint vs gradients-on-all (no grads for
        # embeddings, norms, lm_head).
        for p in model.parameters():
            p.requires_grad_(False)
        quantizable = []
        for full_name, mod, attr in iter_quantizable_tensors(model):
            param = getattr(mod, attr)
            if param.numel() < args.skip_small:
                continue
            # Don't enable here — do it per chunk below
            quantizable.append((full_name, mod, attr, tuple(param.shape), param.numel()))
        print(f"[hawq] {len(quantizable)} quantizable tensors will be measured",
              flush=True)

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            print(f"[hawq] gradient checkpointing enabled", flush=True)

        # Scalar accumulators per Linear. We only need h_trace (sum of g²)
        # per Linear, not the full per-element Fisher. This cuts memory from
        # O(params) to O(num_linears) — from ~108 GB (27B fp32 Fisher) to
        # a few KB. Enables 27B and 122B on 128 GB unified memory.
        h_trace_sum: Dict[str, float] = {name: 0.0 for name, _, _, _, _ in quantizable}
        print(f"[hawq] scalar h_trace accumulators (n={len(h_trace_sum)})",
              flush=True)

        # Split Linears into chunks to reduce peak gradient memory.
        # On CPU with large MoE models, auto-chunk to keep gradient memory
        # under ~30 GB (each chunk's gradients are freed before the next).
        n_chunks = max(1, args.linear_chunks)
        if n_chunks == 1 and args.cpu and n_params > 50_000_000_000:
            # Auto-chunk: ~50 linears per chunk for MoE, keeps grads small
            n_chunks = max(1, len(quantizable) // 50)
            print(f"[hawq] auto-chunking for large CPU model: {n_chunks} chunks",
                  flush=True)
        chunk_size = (len(quantizable) + n_chunks - 1) // n_chunks
        chunks = [quantizable[i:i+chunk_size]
                  for i in range(0, len(quantizable), chunk_size)]
        print(f"[hawq] processing in {len(chunks)} chunk(s) of "
              f"~{chunk_size} linears each", flush=True)

        # Calibration data
        calib_ids = load_wikitext_calibration(
            tokenizer, args.n_calib_samples, args.calib_seqlen)
        print(f"[hawq] running {args.n_calib_samples} × {len(chunks)} "
              f"forward+backward passes", flush=True)

        model.train()
        total_passes = args.n_calib_samples * len(chunks)
        pass_idx = 0
        for i in range(args.n_calib_samples):
            for chunk_idx, chunk in enumerate(chunks):
                pass_idx += 1
                t0 = time.time()
                # Toggle requires_grad for this chunk only
                for name, mod, attr, _, _ in quantizable:
                    getattr(mod, attr).requires_grad_(False)
                for name, mod, attr, _, _ in chunk:
                    getattr(mod, attr).requires_grad_(True)

                model.zero_grad(set_to_none=True)
                batch = calib_ids[i:i+1].to(device)
                out = model(batch, labels=batch)
                if hasattr(out, 'loss') and out.loss is not None:
                    loss = out.loss
                else:
                    # Base model (e.g. extracted language_model from VL model)
                    # — compute causal LM loss manually from hidden states
                    hidden = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
                    # Need lm_head — check model or parent
                    lm_head = getattr(model, 'lm_head', None)
                    embed_w = getattr(model, 'embed_tokens', None)
                    if embed_w is not None:
                        embed_w = embed_w.weight
                    if lm_head is not None:
                        logits = lm_head(hidden)
                    elif embed_w is not None:
                        # Tied embeddings: logits = hidden @ embed.T
                        logits = torch.nn.functional.linear(hidden, embed_w)
                    else:
                        raise RuntimeError("Cannot compute loss: no lm_head or tied embeddings")
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = batch[:, 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                loss.backward()
                with torch.no_grad():
                    for name, mod, attr, _, _ in chunk:
                        g = getattr(mod, attr).grad
                        if g is not None:
                            # Aggregate to a scalar — never materialize the
                            # full per-element Fisher tensor.
                            h_trace_sum[name] += g.float().pow(2).sum().item()
                # Drop grads immediately
                for name, mod, attr, _, _ in chunk:
                    getattr(mod, attr).grad = None
                del out, loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[hawq]   pass {pass_idx}/{total_passes} "
                      f"(sample {i+1}, chunk {chunk_idx+1}/{len(chunks)}) "
                      f"({time.time()-t0:.1f}s)", flush=True)

        # Normalize h_trace sums
        for name in h_trace_sum:
            h_trace_sum[name] /= args.n_calib_samples

        # Aggregate per-Linear sensitivity scalars. Weight stats are computed
        # directly from the parameter tensors (no Fisher needed).
        sensitivity: Dict[str, dict] = {}
        for name, mod, attr, shape, numel in quantizable:
            param = getattr(mod, attr)
            # Handle meta/offloaded tensors — weight stats are secondary
            # to h_trace; use defaults if tensor is on meta device.
            if param.device.type == "meta" or not param.is_floating_point():
                w_max = 0.0
                w_norm = 0.0
            else:
                w = param.data.float()
                w_max = w.abs().max().item()
                w_norm = w.pow(2).sum().item()
                del w
            sensitivity[name] = {
                "shape": list(shape),
                "numel": numel,
                "h_trace": h_trace_sum[name],
                "w_max_abs": w_max,
                "w_norm_sq": w_norm,
            }
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output = {
            "source_model": args.model,
            "method": "fisher_diagonal",
            "n_calib_samples": args.n_calib_samples,
            "calib_seqlen": args.calib_seqlen,
            "n_params": n_params,
            "n_quantizable": len(quantizable),
            "elapsed_sec": time.time() - t_start,
            "sensitivity": sensitivity,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[hawq] done in {time.time() - t_start:.0f}s, saved to {args.output}",
              flush=True)
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
