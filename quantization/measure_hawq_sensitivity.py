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

        is_vl = (cleanup is not None)  # stage_multimodal detected a VL model

        print(f"[hawq] loading {args.model} (VL={is_vl})", flush=True)

        if is_vl:
            # VL models: load full model and extract language_model submodule.
            # This avoids name-mapping issues (safetensors keys keep their
            # original model.language_model.* prefix).
            full_model = AutoModel.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map="cuda",
                trust_remote_code=True,
            )
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
            gc.collect(); torch.cuda.empty_cache()
            print(f"[hawq] extracted language_model from VL model", flush=True)
        else:
            # Text-only models: load directly as CausalLM
            model = AutoModelForCausalLM.from_pretrained(
                staged, torch_dtype=torch.bfloat16, device_map="cuda",
                trust_remote_code=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        device = next(model.parameters()).device
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[hawq]   {n_params:,} params", flush=True)

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

        # Split Linears into chunks if requested (reduces peak grad memory)
        n_chunks = max(1, args.linear_chunks)
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
            w = getattr(mod, attr).data.float()
            sensitivity[name] = {
                "shape": list(shape),
                "numel": numel,
                "h_trace": h_trace_sum[name],
                # h_w2_sum dropped — requires per-element Fisher. Was worse
                # than h_trace × mean(w²) anyway in our validation.
                "w_max_abs": w.abs().max().item(),
                "w_norm_sq": w.pow(2).sum().item(),
            }
            del w
        gc.collect()
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
