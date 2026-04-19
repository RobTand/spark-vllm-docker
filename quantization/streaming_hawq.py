#!/usr/bin/env python3
"""streaming_hawq.py — Fisher-trace HAWQ without full gradient materialization.

The standard HAWQ script (measure_hawq_sensitivity.py) runs loss.backward()
and reads param.grad for every Linear. That requires O(2×|weights|) memory
(weights + gradient tensors), which for Qwen3.6-35B is ~140GB — won't fit
on the Spark's 128GB unified memory.

This script accomplishes the same Fisher estimate using backward hooks that
compute per-Linear gradient statistics on-the-fly, reducing to scalars
BEFORE any gradient tensor outlives its enclosing backward step.

Per Linear we record:
  - h_trace   : sum_i sum_n (∂L_n / ∂w_i)²           Fisher trace
  - h_w2_sum  : sum_i H_ii · w_i²                    expected zero-out loss
  - w_max_abs : max_i |w_i|                          dynamic range
  - w_norm_sq : sum_i w_i²                           weight energy
  - n_params  : weight.numel()

For each Linear we register a full-backward-hook. It fires once per sample,
receives the output gradient tensor grad_y, reconstructs grad_w = grad_y^T @ x
via the input activation saved by a forward hook, reduces to
sum(grad_w²) as a scalar, and drops everything. No gradient tensor larger
than O(in_features × out_features) ever exists beyond one hook call, and
only if we have to reconstruct grad_w (we can skip it via trick below).

Key efficiency:
  sum(grad_w²) where grad_w = x^T @ grad_y with shapes (B,S,I) × (B,S,O)
  = sum over (i,o) of (sum_{b,s} x_{b,s,i} * grad_y_{b,s,o})²
  Which Einsums into a 2D (I, O) matrix, then sum of squares.

We still need to allocate that (I, O) matrix but only transiently per
hook call, so peak memory stays bounded at O(largest_linear).

Forward inputs are retained via gradient checkpointing to bound activation
memory to O(√L) of full model.

Output: pickled dict keyed by "layer_name" -> {h_trace, h_w2_sum, w_max_abs,
                                                w_norm_sq, n_params}
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Multimodal staging — same as the non-streaming script + preprocessor strip
# ---------------------------------------------------------------------------
def stage_text_only(model_path: str) -> str:
    src = Path(model_path)
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        return str(src)
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "vision_config" not in cfg and "text_config" not in cfg:
        return str(src)

    import shutil, tempfile
    for k in ["vision_config", "image_token_id", "video_token_id",
              "vision_start_token_id", "vision_end_token_id"]:
        cfg.pop(k, None)
    if "text_config" in cfg:
        tc = cfg.pop("text_config")
        for k, v in tc.items():
            if k not in cfg:
                cfg[k] = v
        if "model_type" in tc:
            cfg["model_type"] = tc["model_type"]
    archs = cfg.get("architectures", [])
    if archs:
        cfg["architectures"] = [
            a.replace("ForConditionalGeneration", "ForCausalLM") for a in archs
        ]

    staged = Path(tempfile.mkdtemp(prefix="streaming_hawq_stage_"))
    skip = {"config.json", "preprocessor_config.json",
            "video_preprocessor_config.json", "processor_config.json"}
    for p in src.iterdir():
        if p.name in skip:
            continue
        (staged / p.name).symlink_to(p.resolve())
    with open(staged / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return str(staged)


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------
class FisherAccumulator:
    """Tracks Fisher-trace statistics per-Linear without materializing grads."""

    def __init__(self, model: nn.Module, linear_filter):
        self.model = model
        self.linear_filter = linear_filter
        self.stats = {}  # name -> dict of accumulators
        self._saved_inputs = {}  # name -> last forward input
        self._fwd_handles = []
        self._bwd_handles = []

        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if not linear_filter(name):
                continue
            # Initialize stats with zero scalars (live on CPU to save VRAM)
            weight = mod.weight
            self.stats[name] = {
                "h_trace": 0.0,
                "h_w2_sum": 0.0,
                "w_max_abs": float(weight.detach().abs().max().item()),
                "w_norm_sq": float(weight.detach().pow(2).sum().item()),
                "n_params": int(weight.numel()),
                "in_features": mod.in_features,
                "out_features": mod.out_features,
            }
            # Forward hook: save input for use in backward
            def make_fwd(n):
                def h(module, inp, out):
                    # Detach input so it stays alive for backward hook.
                    # Single tensor input assumed (Linear).
                    x = inp[0] if isinstance(inp, tuple) else inp
                    self._saved_inputs[n] = x.detach()
                return h
            self._fwd_handles.append(mod.register_forward_hook(make_fwd(name)))

            # Full backward hook: receives grad_output; reconstructs grad_w
            def make_bwd(n, mod_ref):
                def h(module, grad_input, grad_output):
                    grad_y = grad_output[0]
                    x = self._saved_inputs.pop(n, None)
                    if x is None or grad_y is None:
                        return
                    # grad_w[o,i] = sum_{b,s} grad_y[b,s,o] * x[b,s,i]
                    # Shape: (out, in)
                    # Reshape to 2D collapsing batch/seq:
                    gy2 = grad_y.reshape(-1, grad_y.size(-1))        # [N, O]
                    x2  = x.reshape(-1, x.size(-1))                  # [N, I]
                    grad_w = gy2.t() @ x2                            # [O, I]
                    # Fisher: sum over all (o,i) of grad_w²
                    h_trace_contrib = float(grad_w.pow(2).sum().item())
                    w = mod_ref.weight.detach()
                    h_w2_contrib = float((grad_w.pow(2) * w.pow(2)).sum().item())
                    self.stats[n]["h_trace"] += h_trace_contrib
                    self.stats[n]["h_w2_sum"] += h_w2_contrib
                    # Release temporaries
                    del gy2, x2, grad_w
                return h
            self._bwd_handles.append(
                mod.register_full_backward_hook(make_bwd(name, mod))
            )

    def finalize(self, n_samples: int):
        for name, s in self.stats.items():
            # Normalize to per-sample average (matches non-streaming script)
            s["h_trace"] /= max(n_samples, 1)
            s["h_w2_sum"] /= max(n_samples, 1)

    def remove_hooks(self):
        for h in self._fwd_handles + self._bwd_handles:
            h.remove()
        self._fwd_handles.clear()
        self._bwd_handles.clear()


# ---------------------------------------------------------------------------
# Calibration loop
# ---------------------------------------------------------------------------
def load_calibration(tokenizer, dataset_name: str, n_samples: int, seqlen: int):
    from datasets import load_dataset
    if dataset_name == "ultrachat_200k":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                          split="train_sft", streaming=True)
        texts = []
        for row in ds:
            msgs = row.get("messages", [])
            if not msgs:
                continue
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
            texts.append(text)
            if len(texts) >= n_samples * 4:  # oversample; many will be short
                break
    elif dataset_name.endswith(".jsonl") or dataset_name.endswith(".json"):
        with open(dataset_name) as f:
            texts = [json.loads(l)["text"] if l.strip().startswith("{")
                     else l.strip() for l in f if l.strip()]
    else:
        # Fallback: pile-10k
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        texts = [row["text"] for row in ds]

    # Tokenize and keep only samples >= seqlen
    input_ids = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=False).input_ids
        if ids.size(1) < seqlen:
            continue
        # Take a random seqlen window
        import random
        start = random.randint(0, ids.size(1) - seqlen)
        input_ids.append(ids[0, start:start + seqlen])
        if len(input_ids) >= n_samples:
            break

    if len(input_ids) < n_samples:
        print(f"[hawq] warning: only got {len(input_ids)} samples of target "
              f"{n_samples} (not enough long-enough sequences)")
    return torch.stack(input_ids, dim=0)  # [N, S]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ultrachat_200k")
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output", required=True,
                    help="Output pickle with per-Linear Fisher stats")
    ap.add_argument("--linear-include", default=".*",
                    help="Regex: only include Linear names matching this")
    ap.add_argument("--linear-exclude",
                    default=r"(?:^lm_head$|\.lm_head$|mlp\.gate$|mlp\..*gate$)",
                    help="Regex: exclude Linear names matching this")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    default=True)
    ap.add_argument("--no-gradient-checkpointing", action="store_false",
                    dest="gradient_checkpointing")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    staged = stage_text_only(args.model)
    print(f"[hawq] staged: {staged}", flush=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)

    print(f"[hawq] loading model ({args.model}, dtype={args.dtype}, "
          f"device={args.device}, gc={args.gradient_checkpointing})",
          flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        staged,
        torch_dtype=dtype,
        device_map=args.device,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )
    model.eval()
    # CRITICAL: keep requires_grad=False on params to avoid allocating
    # .grad tensors alongside each weight (that would double memory).
    # Backward hooks still fire because gradients flow through input/activations
    # even when params are frozen.  We reconstruct grad_w = grad_output^T @ input
    # inside the hook using the saved forward input — never touching param.grad.
    for p in model.parameters():
        p.requires_grad_(False)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"[hawq] model loaded in {time.time()-t0:.1f}s", flush=True)

    # Set up Fisher accumulator
    inc = re.compile(args.linear_include)
    exc = re.compile(args.linear_exclude)
    def linear_filter(name: str) -> bool:
        return bool(inc.search(name)) and not bool(exc.search(name))

    acc = FisherAccumulator(model, linear_filter)
    print(f"[hawq] tracking {len(acc.stats)} Linear layers", flush=True)

    # Load calibration
    calib_ids = load_calibration(tokenizer, args.dataset,
                                 args.nsamples, args.seqlen)
    print(f"[hawq] calibration: {calib_ids.shape}", flush=True)

    # Forward+backward per sample.
    # With params frozen (requires_grad=False), autograd would not build a
    # graph — so we inject a requires_grad=True scalar tap at the input-
    # embedding step. Everything downstream then tracks gradients w.r.t.
    # that tap, flowing through every Linear's grad_output (which our hooks
    # consume).  No param.grad is ever allocated.
    #
    # Concretely: get the embedding output for each sample, clone it with
    # requires_grad=True, then run the rest of the model forward starting
    # from that point. If the model exposes `inputs_embeds`, we use that.
    model.train()  # keep for gradient-checkpointing semantics; no params update
    t_fwd = 0.0
    t_bwd = 0.0
    for i in range(calib_ids.size(0)):
        ids = calib_ids[i:i+1].to(args.device)
        t0 = time.time()
        # Embed first so we can attach requires_grad at the embedding output.
        with torch.no_grad():
            embed = model.get_input_embeddings()(ids)
        embed.requires_grad_(True)
        out = model(inputs_embeds=embed, labels=ids)
        loss = out.loss
        t_fwd += time.time() - t0
        t0 = time.time()
        loss.backward()
        t_bwd += time.time() - t0
        if (i + 1) % 4 == 0 or i == 0:
            print(f"[hawq] sample {i+1}/{calib_ids.size(0)} loss={loss.item():.3f} "
                  f"fwd_avg={t_fwd/(i+1):.2f}s bwd_avg={t_bwd/(i+1):.2f}s",
                  flush=True)
        # aggressive cleanup
        del out, loss, ids, embed
        # clear saved inputs dict just in case
        acc._saved_inputs.clear()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    acc.finalize(calib_ids.size(0))
    acc.remove_hooks()

    # Dump stats
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "stats": acc.stats,
            "meta": {
                "model": args.model,
                "dataset": args.dataset,
                "nsamples": calib_ids.size(0),
                "seqlen": args.seqlen,
                "dtype": args.dtype,
                "device": args.device,
            },
        }, f)
    print(f"[hawq] wrote {out_path} with {len(acc.stats)} layer stats",
          flush=True)


if __name__ == "__main__":
    main()
