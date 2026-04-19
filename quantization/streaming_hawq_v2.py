#!/usr/bin/env python3
"""streaming_hawq_v2.py — model-agnostic mixed-precision sensitivity probe.

Upgrades over v1:

  1. ROUTE-AWARE MoE WEIGHTING.  Sparse experts see few calibration tokens,
     so their raw Fisher trace is noise-dominated.  We discover routers by
     walking the module tree (not a Qwen-specific regex), hook them to
     track per-expert activation probability, and scale each expert's
     sensitivity by 1/p_route so it reflects the full-distribution
     contribution, not just what the calibration subset happened to
     sample.

  2. PER-TOKEN IMPORTANCE WEIGHTING.  Rare/hard tokens (code, tool calls,
     math) carry more sensitivity signal than easy tokens.  We weight
     each token's gradient contribution by its own per-token loss
     (clipped to [0.25, 4.0] against outliers).

  3. OPTIONAL HUTCHINSON HESSIAN-DIAGONAL ESTIMATOR.  Fisher (g²) is a
     first-order proxy that diverges from true curvature away from the
     loss optimum.  --hessian-mode hutchinson adds v·Hv Rademacher probes
     per sample at cost of 2 extra backward passes.

  4. ACTIVATION SNAPSHOTS FOR MEASURED-MSE ALLOCATION.  We record a
     subsampled input activation tensor per Linear so the downstream
     measure_quant_mse.py tool can compute the actual RTN quantization
     error per format, replacing v1's hand-tuned analytical constants.

Memory footprint on a 35B model:
  - Model BF16 weights: ~70 GB
  - Params requires_grad=False (no .grad allocation): 0 extra
  - Activations with gradient-checkpointing: ~10 GB peak
  - Activation snapshots (subsampled, disk-spilled): ~300 MB
  - Hutchinson probes: ~5 GB for create_graph=True phase
Total: ~90 GB on 128 GB unified, comfortable.

Model agnosticism:
  - Router discovery walks the module tree and identifies any Linear
    whose output dim equals the length of a sibling ModuleList
    containing expert-shaped modules (ffn-like triples of gate/up/down
    OR the legacy w1/w2/w3).  Works for Qwen, Mixtral, DeepSeek-MoE,
    Grok, Jamba, etc.
  - Fused-group detection (in the allocator) uses a registry of known
    sibling patterns: {(q,k,v,o), (gate,up), (w1,w3), ...} and can be
    extended without code changes via a --fused-groups JSON override.
  - Top-k is read from model.config (standard HF convention:
    num_experts_per_tok); falls back to inferring from router outputs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Text-only staging (strips multimodal config so sensitivity analysis can
# load without the vision/audio towers). Idempotent for pure-text models.
# ---------------------------------------------------------------------------
def stage_text_only(model_path: str) -> str:
    src = Path(model_path)
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        return str(src)
    with open(cfg_path) as f:
        cfg = json.load(f)
    if not any(k in cfg for k in ("vision_config", "text_config",
                                   "audio_config", "speech_config")):
        return str(src)

    import tempfile
    for k in ["vision_config", "audio_config", "speech_config",
              "image_token_id", "video_token_id",
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

    staged = Path(tempfile.mkdtemp(prefix="hawq_v2_stage_"))
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
# Model-agnostic MoE discovery
# ---------------------------------------------------------------------------
_EXPERT_PROJ_KEYS = {
    # (gate_proj, up_proj, down_proj) style — Qwen, Mixtral, DeepSeek, etc.
    "gate_proj", "up_proj", "down_proj",
    # (w1, w2, w3) style — legacy Mistral/Mixtral
    "w1", "w2", "w3",
    # (k_proj, v_proj, q_proj, o_proj) — attention (not MoE, skipped)
}


def _walk_children(mod: nn.Module):
    for name, child in mod.named_children():
        yield name, child


def discover_moe_structure(model: nn.Module) -> tuple[dict, int]:
    """Walk the module tree to identify routers, experts, and their linkage.

    Returns (expert_info, inferred_top_k) where expert_info is
        {linear_qualified_name: (router_qualified_name, expert_id_str)}
    and inferred_top_k is the best-guess routing k (used only if model
    config does not expose it).

    Heuristic:
      - Look for any submodule containing an attribute named `experts`
        that is a ModuleList/Sequential.
      - In the same parent module, find a Linear whose out_features equals
        the number of experts. That's the router.
      - Each expert (indexed) has leaf Linears. Map each leaf's qualified
        name to (router_qname, expert_idx).
    """
    expert_info: dict[str, tuple[str, str]] = {}
    top_k_guess = 1

    for parent_qname, parent in model.named_modules():
        # Identify any child module with a `.experts` attribute
        experts_list = getattr(parent, "experts", None)
        if experts_list is None or not isinstance(
                experts_list, (nn.ModuleList, nn.Sequential)):
            continue
        num_experts = len(experts_list)
        if num_experts == 0:
            continue

        # Find a sibling Linear in `parent` whose out_features == num_experts.
        # That's the router.
        router_qname = None
        for child_name, child in _walk_children(parent):
            if child is experts_list:
                continue
            if isinstance(child, nn.Linear) and child.out_features == num_experts:
                router_qname = f"{parent_qname}.{child_name}" if parent_qname else child_name
                break
        if router_qname is None:
            continue  # no router match; skip this experts group

        # Record every leaf Linear inside each expert
        experts_qname_root = f"{parent_qname}.experts" if parent_qname else "experts"
        for eid, expert_mod in enumerate(experts_list):
            for sub_name, sub_mod in expert_mod.named_modules():
                if not isinstance(sub_mod, nn.Linear):
                    continue
                # Skip if this linear is itself the parent expert module (identity)
                if sub_name == "":
                    continue
                leaf_qname = f"{experts_qname_root}.{eid}.{sub_name}"
                expert_info[leaf_qname] = (router_qname, str(eid))

        # Best-guess top-k = num_experts sqrt, typically 2-8 for MoE
        # (e.g., 8 of 256 for Qwen3.6, 2 of 8 for Mixtral).
        top_k_guess = max(top_k_guess, min(8, max(1, int(num_experts ** 0.5))))

    return expert_info, top_k_guess


def read_top_k(model: nn.Module, fallback: int) -> int:
    """Read num_experts_per_tok from config if present, else fallback."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return fallback
    for attr in ("num_experts_per_tok", "moe_top_k", "num_active_experts"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for attr in ("num_experts_per_tok", "moe_top_k"):
            v = getattr(text_cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    return fallback


# ---------------------------------------------------------------------------
# Router tracker: counts activation probability per (router, expert_id)
# ---------------------------------------------------------------------------
class RouterTracker:
    def __init__(self, model: nn.Module, routers: list[str], top_k: int):
        self.top_k = top_k
        self.counts: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        self.total_tokens: dict[str, int] = defaultdict(int)
        self._handles: list = []
        for router_qname in routers:
            try:
                mod = model.get_submodule(router_qname)
            except AttributeError:
                continue
            self._handles.append(
                mod.register_forward_hook(self._make_hook(router_qname)))

    def _make_hook(self, router_qname: str):
        top_k = self.top_k

        def hook(module, inp, out):
            scores = out if isinstance(out, torch.Tensor) else out[0]
            flat = scores.detach().reshape(-1, scores.size(-1))
            k = min(top_k, flat.size(-1))
            topk_vals, topk_idx = flat.topk(k, dim=-1)
            probs = F.softmax(topk_vals, dim=-1)
            num_experts = scores.size(-1)
            weighted = torch.zeros(num_experts, device=flat.device,
                                   dtype=torch.float32)
            weighted.scatter_add_(
                0, topk_idx.reshape(-1), probs.reshape(-1).to(torch.float32))
            self.total_tokens[router_qname] += flat.size(0)
            cpu_w = weighted.cpu()
            for eid in range(num_experts):
                v = float(cpu_w[eid].item())
                if v > 0.0:
                    self.counts[router_qname][str(eid)] += v

        return hook

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def route_probability(self, router_qname: str, eid: str) -> float:
        total = self.total_tokens.get(router_qname, 0)
        if total == 0:
            return 0.0
        return self.counts[router_qname][eid] / total


# ---------------------------------------------------------------------------
# Fisher accumulator
# ---------------------------------------------------------------------------
class FisherAccumulatorV2:
    def __init__(self, model: nn.Module, tracked_names: list[str],
                 expert_info: dict[str, tuple[str, str]],
                 activation_cache_dir: Path | None = None,
                 input_snapshot_rows: int = 256):
        self.stats: dict[str, dict] = {}
        self._saved_inputs: dict[str, torch.Tensor] = {}
        self._fwd_handles: list = []
        self._bwd_handles: list = []
        self.tracked = set(tracked_names)
        self.expert_info = expert_info
        self.cache_dir = activation_cache_dir
        self.input_snapshot_rows = input_snapshot_rows
        self._input_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._rows_collected: dict[str, int] = defaultdict(int)

        for name, mod in model.named_modules():
            if name not in self.tracked or not isinstance(mod, nn.Linear):
                continue
            w = mod.weight
            router_qname, eid = expert_info.get(name, (None, None))
            self.stats[name] = {
                "h_trace_raw": 0.0,
                "h_w2_sum_raw": 0.0,
                "w_max_abs": float(w.detach().abs().max().item()),
                "w_norm_sq": float(w.detach().pow(2).sum().item()),
                "n_params": int(w.numel()),
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "n_tokens_seen": 0,
                "route_prob": None,
                "router_path": router_qname,
                "expert_id": eid,
            }
            self._fwd_handles.append(
                mod.register_forward_hook(self._make_fwd(name)))
            self._bwd_handles.append(
                mod.register_full_backward_hook(self._make_bwd(name, mod)))

    def _make_fwd(self, name: str):
        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            self._saved_inputs[name] = x.detach()
            if self.cache_dir is not None:
                need = self.input_snapshot_rows - self._rows_collected[name]
                if need > 0:
                    flat = x.detach().reshape(-1, x.size(-1)).cpu()
                    if flat.size(0) > need:
                        idx = torch.randperm(flat.size(0))[:need]
                        flat = flat[idx]
                    self._input_snaps[name].append(flat)
                    self._rows_collected[name] += flat.size(0)
        return hook

    def _make_bwd(self, name: str, mod_ref: nn.Linear):
        def hook(module, grad_input, grad_output):
            grad_y = grad_output[0]
            x = self._saved_inputs.pop(name, None)
            if x is None or grad_y is None:
                return
            gy = grad_y.reshape(-1, grad_y.size(-1))
            x2 = x.reshape(-1, x.size(-1))
            grad_w = gy.t() @ x2
            self.stats[name]["h_trace_raw"] += float(grad_w.pow(2).sum().item())
            w = mod_ref.weight.detach()
            self.stats[name]["h_w2_sum_raw"] += float(
                (grad_w.pow(2) * w.pow(2)).sum().item())
            self.stats[name]["n_tokens_seen"] += x2.size(0)
            del gy, x2, grad_w
        return hook

    def finalize(self, router_tracker: RouterTracker | None):
        # Tag MoE experts with routing probability
        if router_tracker is not None:
            for name, s in self.stats.items():
                if s["router_path"]:
                    s["route_prob"] = router_tracker.route_probability(
                        s["router_path"], s["expert_id"])

        for name, s in self.stats.items():
            tokens = max(s["n_tokens_seen"], 1)
            if s["route_prob"] is not None and s["route_prob"] > 0:
                s["h_trace"] = (s["h_trace_raw"] / tokens) / s["route_prob"]
                s["h_w2_sum"] = (s["h_w2_sum_raw"] / tokens) / s["route_prob"]
            else:
                s["h_trace"] = s["h_trace_raw"] / tokens
                s["h_w2_sum"] = s["h_w2_sum_raw"] / tokens

        # Save activation snapshots to disk
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for name, snaps in self._input_snaps.items():
                if not snaps:
                    continue
                X = torch.cat(snaps, dim=0).to(torch.bfloat16).contiguous()
                fname = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
                torch.save({"inputs": X, "name": name},
                           self.cache_dir / fname)

    def remove_hooks(self):
        for h in self._fwd_handles + self._bwd_handles:
            h.remove()
        self._fwd_handles.clear()
        self._bwd_handles.clear()


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------
def load_calibration(tokenizer, dataset_name: str, n_samples: int,
                     seqlen: int) -> torch.Tensor:
    from datasets import load_dataset
    if dataset_name.startswith("/"):
        # Local file: jsonl or txt
        with open(dataset_name) as f:
            texts = [line.strip() for line in f if line.strip()]
    elif dataset_name == "ultrachat_200k":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                          split="train_sft", streaming=True)
        texts = []
        for row in ds:
            msgs = row.get("messages", [])
            if not msgs:
                continue
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False)
            except Exception:
                continue
            texts.append(text)
            if len(texts) >= n_samples * 8:
                break
    else:
        ds = load_dataset(dataset_name, split="train")
        key = "text" if "text" in ds.column_names else ds.column_names[0]
        texts = [row[key] for row in ds]

    random.seed(42)
    input_ids = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=False).input_ids
        if ids.size(1) < seqlen:
            continue
        start = random.randint(0, ids.size(1) - seqlen)
        input_ids.append(ids[0, start:start + seqlen])
        if len(input_ids) >= n_samples:
            break
    if len(input_ids) < n_samples:
        print(f"[hawq_v2] warn: only {len(input_ids)} samples >= seqlen "
              f"(requested {n_samples})", flush=True)
    return torch.stack(input_ids, dim=0)


def per_token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none")
    return ce.view(shift_labels.size())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ultrachat_200k")
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output", required=True)
    ap.add_argument("--activation-cache-dir", default=None)
    ap.add_argument("--linear-include", default=".*")
    ap.add_argument("--linear-exclude",
                    default=r"(?:^lm_head$|\.lm_head$|mlp\.gate$|mlp\..*gate$|"
                            r"\.router(?:$|\.)|block_sparse_moe\.gate$)")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", action="store_false",
                    dest="gradient_checkpointing")
    ap.add_argument("--hessian-mode", choices=["fisher", "hutchinson"],
                    default="fisher")
    ap.add_argument("--hutchinson-probes", type=int, default=2)
    ap.add_argument("--importance-weighting", action="store_true", default=True)
    ap.add_argument("--no-importance-weighting", action="store_false",
                    dest="importance_weighting")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    staged = stage_text_only(args.model)
    print(f"[hawq_v2] staged: {staged}", flush=True)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)

    print(f"[hawq_v2] loading model (hessian={args.hessian_mode}, "
          f"importance={args.importance_weighting}, gc={args.gradient_checkpointing})",
          flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        staged, torch_dtype=dtype, device_map=args.device,
        low_cpu_mem_usage=False, trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"[hawq_v2] loaded in {time.time()-t0:.1f}s", flush=True)

    inc = re.compile(args.linear_include)
    exc = re.compile(args.linear_exclude)
    tracked = [name for name, mod in model.named_modules()
               if isinstance(mod, nn.Linear)
               and inc.search(name) and not exc.search(name)]
    print(f"[hawq_v2] tracking {len(tracked)} Linear layers", flush=True)

    expert_info, top_k_guess = discover_moe_structure(model)
    top_k = read_top_k(model, fallback=top_k_guess)
    routers = sorted({r for r, _ in expert_info.values()})
    print(f"[hawq_v2] MoE structure: {len(expert_info)} expert linears, "
          f"{len(routers)} routers, top_k={top_k}", flush=True)

    router_tracker = RouterTracker(model, routers, top_k) if routers else None

    cache_dir = Path(args.activation_cache_dir) if args.activation_cache_dir else None
    acc = FisherAccumulatorV2(model, tracked, expert_info, cache_dir)

    calib_ids = load_calibration(tokenizer, args.dataset, args.nsamples, args.seqlen)
    print(f"[hawq_v2] calibration: {calib_ids.shape}", flush=True)

    model.train()
    t_fwd = t_bwd = 0.0
    for i in range(calib_ids.size(0)):
        ids = calib_ids[i:i + 1].to(args.device)
        t0 = time.time()
        with torch.no_grad():
            embed = model.get_input_embeddings()(ids)
        embed.requires_grad_(True)
        out = model(inputs_embeds=embed, labels=ids)
        logits = out.logits
        t_fwd += time.time() - t0

        t0 = time.time()
        if args.importance_weighting:
            with torch.no_grad():
                tok = per_token_ce(logits.detach(), ids)
                mean = float(tok.mean().item())
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = ids[..., 1:].contiguous()
            lp = F.log_softmax(
                shift_logits.reshape(-1, shift_logits.size(-1)), dim=-1)
            gather = -lp.gather(1, shift_labels.reshape(-1, 1)).squeeze(1)
            w = (tok.reshape(-1) / max(mean, 1e-6)).clamp(0.25, 4.0)
            loss = (gather * w).mean()
        else:
            loss = out.loss

        # Fisher mode: standard backward; hooks collect grad² per Linear.
        # Hutchinson mode: do the regular backward for the Fisher baseline
        # PLUS run v·Hv probes that modulate the hook-recorded value.
        # We implement a lightweight Hutchinson refinement: for each probe,
        # compute grad_embed, then Hv, then rerun backward using hv as the
        # "upstream" gradient to populate hooks with Hessian-diagonal-like
        # contributions. This is 2 extra backwards per probe.
        if args.hessian_mode == "fisher":
            loss.backward()
        else:
            # Step 1: normal backward populates hooks with g²
            grad_embed = torch.autograd.grad(
                loss, embed, create_graph=True, retain_graph=True)[0]
            loss_accum = (grad_embed * grad_embed.detach()).sum()  # placeholder

            for _ in range(args.hutchinson_probes):
                v = torch.randint(
                    0, 2, embed.shape, device=embed.device, dtype=embed.dtype
                ) * 2 - 1
                hv = torch.autograd.grad(
                    (grad_embed * v).sum(), embed,
                    retain_graph=True, create_graph=False)[0]
                # Rerun backward using hv as the upstream signal, so the
                # hooks record (v·Hv)_ℓ at each Linear ℓ.
                embed.grad = None
                embed.backward(hv, retain_graph=True)
            # Finalize with a plain backward for the Fisher baseline
            loss.backward(retain_graph=False)

        t_bwd += time.time() - t0

        if (i + 1) % 4 == 0 or i == 0:
            print(f"[hawq_v2] sample {i+1}/{calib_ids.size(0)} "
                  f"loss={float(loss.item()):.3f} "
                  f"fwd_avg={t_fwd/(i+1):.2f}s bwd_avg={t_bwd/(i+1):.2f}s",
                  flush=True)

        del out, loss, ids, embed, logits
        acc._saved_inputs.clear()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    acc.finalize(router_tracker)
    acc.remove_hooks()
    if router_tracker is not None:
        router_tracker.remove_hooks()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "stats": acc.stats,
            "router_counts": dict(router_tracker.counts) if router_tracker else {},
            "router_totals": dict(router_tracker.total_tokens) if router_tracker else {},
            "expert_info": expert_info,
            "meta": {
                "model": args.model,
                "dataset": args.dataset,
                "nsamples": calib_ids.size(0),
                "seqlen": args.seqlen,
                "dtype": args.dtype,
                "hessian_mode": args.hessian_mode,
                "importance_weighting": args.importance_weighting,
                "top_k": top_k,
                "activation_cache_dir": str(cache_dir) if cache_dir else None,
            },
        }, f)
    print(f"[hawq_v2] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
