#!/usr/bin/env python3
"""sensitivity_probe.py — per-Linear curvature measurement.

Cleaned rewrite of streaming_hawq_v2.py, now inside the dynaquant package.
Removes the broken Hutchinson stub; Fisher trace stays the default. Adds:

  - route-aware MoE scaling (discover routers by walking module tree)
  - per-token importance weighting (harder tokens count more)
  - activation snapshot cache for measure_quant_cost.py

Memory:
  - params requires_grad_(False)   → no gradient tensor storage
  - gradient checkpointing on      → activations are recomputed during backward
  - backward hooks reduce grad_w to a scalar inline and drop it
Result: peak ≈ model weights + one-block activation, fits in 128 GB for 35 B.

Model-agnostic:
  - Router discovered via module walk (any Linear whose out_features equals a
    sibling ModuleList named experts, gates, etc.)
  - Top-k read from model.config (num_experts_per_tok)
  - Dense models just skip RouterTracker
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Text-only staging
# ---------------------------------------------------------------------------
def stage_text_only(model_path: str) -> str:
    src = Path(model_path)
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        return str(src)
    with open(cfg_path) as f:
        cfg = json.load(f)
    if not any(k in cfg for k in
               ("vision_config", "text_config", "audio_config", "speech_config")):
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

    staged = Path(tempfile.mkdtemp(prefix="dynaquant_stage_"))
    skip = {"config.json", "preprocessor_config.json",
            "video_preprocessor_config.json", "processor_config.json"}
    for p in src.iterdir():
        if p.name in skip:
            continue
        (staged / p.name).symlink_to(p.resolve())
    with open(staged / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return str(staged)


def prepare_model_for_moe_linears(model: nn.Module) -> str | None:
    """Unfuse supported packed MoE expert tensors into per-expert nn.Linear.

    Returns the target device string used for the unfused linears when any
    change was made, else None.

    This prefers AutoRound's public helper, but falls back to the lower-level
    expert-interface unfuser with `check_decorator=False` for model families
    like Qwen3.6 whose experts implement the same packed shape conventions
    yet do not pass the public decorator gate cleanly.
    """
    try:
        from auto_round.modeling.fused_moe import prepare_model_for_moe_quantization
        from auto_round.modeling.fused_moe.moe_experts_interface import (
            LINEAR_LOOP_IMPL,
            _unfuse_experts_weights_inplace,
            register_linear_loop_experts,
        )
    except ImportError:
        return None

    target_dev = None
    for p in model.parameters():
        target_dev = p.device
        if p.device.type != "cpu":
            break
    if target_dev is None:
        target_dev = torch.device("cpu")

    unfused_modules: list[str] = []
    try:
        unfused_modules = prepare_model_for_moe_quantization(model) or []
    except Exception:
        unfused_modules = []

    # Fallback: forcibly unfuse any experts module carrying packed 3D
    # parameters (e.g. Qwen3.6's gate_up_proj/down_proj) even when the
    # public helper declines due to decorator checks.
    forced_modules: list[str] = []
    if not unfused_modules:
        try:
            register_linear_loop_experts()
        except Exception:
            pass
        for name, module in model.named_modules():
            try:
                changed = _unfuse_experts_weights_inplace(module, check_decorator=False)
            except Exception:
                changed = False
            if changed:
                forced_modules.append(name)
        unfused_modules = forced_modules

    if not unfused_modules:
        return None

    if hasattr(model, "config"):
        model.config._experts_implementation = LINEAR_LOOP_IMPL

    for sub in model.modules():
        if isinstance(sub, nn.Linear) and sub.weight.device != target_dev:
            sub.to(target_dev)

    return str(target_dev)


def resolve_execution_device(model: nn.Module, requested_device: str) -> torch.device:
    """Choose the device used for input ids / embeddings during probing.

    When `device_map="auto"` is used for model load, the model can be sharded
    across CPU and GPU. In that case we want to feed tokens to the device that
    owns the input embedding weights rather than assuming a single global
    `cuda`/`cpu` target.
    """
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device(requested_device)


# ---------------------------------------------------------------------------
# Model-agnostic MoE discovery
# ---------------------------------------------------------------------------
def discover_moe_structure(model: nn.Module) -> dict[str, tuple[str, str]]:
    """Return {expert_linear_qname: (router_qname, expert_id_str)}.

    Walk the module tree.  For any module that has a child attribute named
    `experts` or `block_sparse_moe_experts` that is a ModuleList, find a
    sibling Linear in the same parent whose out_features equals len(experts).
    That Linear is the router.
    """
    def _router_matches_num_experts(child: nn.Module, num_experts: int) -> bool:
        if isinstance(child, nn.Linear) and child.out_features == num_experts:
            return True
        weight = getattr(child, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
            return int(weight.shape[0]) == num_experts
        return False

    expert_info: dict[str, tuple[str, str]] = {}
    for parent_qname, parent in model.named_modules():
        candidates = []
        for attr in ("experts", "block_sparse_moe_experts",
                     "moe_experts", "expert_layer"):
            experts_container = getattr(parent, attr, None)
            if experts_container is None or not isinstance(experts_container, nn.Module):
                continue
            # Two possible layouts:
            #   A) experts_container IS the list (nn.ModuleList / nn.Sequential /
            #      AutoRound's SequentialQwen3_5MoeExperts which subclasses ModuleList)
            #   B) experts_container is a plain nn.Module with numbered children
            #      (e.g. Qwen3_5MoeExperts after in-place unfuse: children are
            #      named "0", "1", ..., each holding per-expert Linears).
            #
            # Both layouts are detected by looking at child names that are
            # consecutive integer strings starting from 0.
            child_dict = dict(experts_container.named_children())
            numeric_keys = sorted(
                [k for k in child_dict if k.isdigit()],
                key=int,
            )
            if numeric_keys:
                # Require the numeric children to be 0..N-1 (no gaps)
                if [int(k) for k in numeric_keys] != list(range(len(numeric_keys))):
                    continue
                if not all(isinstance(child_dict[k], nn.Module) for k in numeric_keys):
                    continue
                candidates.append((attr, experts_container, "nested", numeric_keys))
                continue

            # Linear-loop layout after MoE unfuse: experts container itself
            # remains a module, but its packed projections become ModuleLists:
            #   experts.gate_up_proj.<expert_idx>
            #   experts.down_proj.<expert_idx>
            projection_lists = {}
            for proj_name in ("gate_up_proj", "down_proj", "w1", "w2", "w3"):
                proj = getattr(experts_container, proj_name, None)
                if proj is None or not isinstance(proj, nn.Module):
                    continue
                proj_children = dict(proj.named_children())
                proj_numeric = sorted([k for k in proj_children if k.isdigit()], key=int)
                if not proj_numeric:
                    continue
                if [int(k) for k in proj_numeric] != list(range(len(proj_numeric))):
                    continue
                if not all(isinstance(proj_children[k], nn.Module) for k in proj_numeric):
                    continue
                projection_lists[proj_name] = proj_numeric
            if projection_lists:
                # Require a consistent expert count across projections.
                expert_lists = list(projection_lists.values())
                if all(v == expert_lists[0] for v in expert_lists[1:]):
                    candidates.append((attr, experts_container, "linear_loop", expert_lists[0]))
        if not candidates:
            continue
        attr_name, experts_container, layout, numeric_keys = candidates[0]
        num_experts = len(numeric_keys)

        # Find sibling Linear (or any module whose output feature dim
        # equals num_experts) that acts as the router.
        router_qname = None
        for child_name, child in parent.named_children():
            if child is experts_container:
                continue
            if _router_matches_num_experts(child, num_experts):
                router_qname = (f"{parent_qname}.{child_name}"
                                if parent_qname else child_name)
                break
        if router_qname is None:
            continue

        experts_root = (f"{parent_qname}.{attr_name}"
                        if parent_qname else attr_name)
        if layout == "nested":
            for eid_str in numeric_keys:
                expert_mod = child_dict[eid_str]
                for sub_name, sub_mod in expert_mod.named_modules():
                    if not isinstance(sub_mod, nn.Linear) or sub_name == "":
                        continue
                    leaf = f"{experts_root}.{eid_str}.{sub_name}"
                    expert_info[leaf] = (router_qname, eid_str)
        else:
            for proj_name in ("gate_up_proj", "down_proj", "w1", "w2", "w3"):
                proj = getattr(experts_container, proj_name, None)
                if proj is None or not isinstance(proj, nn.Module):
                    continue
                proj_children = dict(proj.named_children())
                for eid_str in numeric_keys:
                    sub_mod = proj_children.get(eid_str)
                    if not isinstance(sub_mod, nn.Linear):
                        continue
                    leaf = f"{experts_root}.{proj_name}.{eid_str}"
                    expert_info[leaf] = (router_qname, eid_str)

    return expert_info


def read_top_k(model: nn.Module, default: int = 2) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return default
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
    return default


# ---------------------------------------------------------------------------
# Router tracker: per-(router, expert) activation probability
# ---------------------------------------------------------------------------
class RouterTracker:
    def __init__(self, model: nn.Module, routers: list[str], top_k: int):
        self.top_k = top_k
        self.counts: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        self.total_tokens: dict[str, int] = defaultdict(int)
        self._handles = []
        for rq in routers:
            try:
                mod = model.get_submodule(rq)
            except AttributeError:
                continue
            self._handles.append(mod.register_forward_hook(self._make_hook(rq)))

    def _make_hook(self, router_qname: str):
        def hook(module, inp, out):
            scores = out if isinstance(out, torch.Tensor) else out[0]
            flat = scores.detach().reshape(-1, scores.size(-1))
            k = min(self.top_k, flat.size(-1))
            topk_v, topk_i = flat.topk(k, dim=-1)
            probs = F.softmax(topk_v, dim=-1)
            n_experts = scores.size(-1)
            weighted = torch.zeros(n_experts, device=flat.device,
                                   dtype=torch.float32)
            weighted.scatter_add_(
                0, topk_i.reshape(-1), probs.reshape(-1).to(torch.float32))
            self.total_tokens[router_qname] += flat.size(0)
            cpu_w = weighted.cpu()
            for eid in range(n_experts):
                v = float(cpu_w[eid].item())
                if v > 0.0:
                    self.counts[router_qname][str(eid)] += v
        return hook

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def prob(self, router_qname: str, eid: str) -> float:
        total = self.total_tokens.get(router_qname, 0)
        if total == 0:
            return 0.0
        return self.counts[router_qname][eid] / total


# ---------------------------------------------------------------------------
# Fisher accumulator with activation snapshot cache
# ---------------------------------------------------------------------------
class FisherAccumulator:
    def __init__(self, model: nn.Module, tracked: list[str],
                 expert_info: dict[str, tuple[str, str]],
                 act_cache_dir: Path | None = None,
                 input_rows: int = 256):
        self.stats: dict[str, dict] = {}
        self._saved_inputs: dict[str, torch.Tensor] = {}
        self._fwd_handles, self._bwd_handles = [], []
        self.tracked = set(tracked)
        self.expert_info = expert_info
        self.cache_dir = act_cache_dir
        self.input_rows = input_rows
        self._input_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._rows_got: dict[str, int] = defaultdict(int)

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
                need = self.input_rows - self._rows_got[name]
                if need > 0:
                    flat = x.detach().reshape(-1, x.size(-1)).cpu()
                    if flat.size(0) > need:
                        idx = torch.randperm(flat.size(0))[:need]
                        flat = flat[idx]
                    self._input_snaps[name].append(flat)
                    self._rows_got[name] += flat.size(0)
        return hook

    def _make_bwd(self, name: str, mod_ref: nn.Linear):
        def hook(module, grad_input, grad_output):
            gy = grad_output[0]
            x = self._saved_inputs.pop(name, None)
            if x is None or gy is None:
                return
            gy2 = gy.reshape(-1, gy.size(-1))
            x2 = x.reshape(-1, x.size(-1))
            grad_w = gy2.t() @ x2
            self.stats[name]["h_trace_raw"] += float(grad_w.pow(2).sum().item())
            w = mod_ref.weight.detach()
            self.stats[name]["h_w2_sum_raw"] += float(
                (grad_w.pow(2) * w.pow(2)).sum().item())
            self.stats[name]["n_tokens_seen"] += x2.size(0)
        return hook

    def finalize(self, tracker: RouterTracker | None):
        if tracker is not None:
            for name, s in self.stats.items():
                if s["router_path"]:
                    s["route_prob"] = tracker.prob(
                        s["router_path"], s["expert_id"])

        for s in self.stats.values():
            tokens = max(s["n_tokens_seen"], 1)
            if s["route_prob"] is not None and s["route_prob"] > 0:
                s["h_trace"] = (s["h_trace_raw"] / tokens) / s["route_prob"]
                s["h_w2_sum"] = (s["h_w2_sum_raw"] / tokens) / s["route_prob"]
            else:
                s["h_trace"] = s["h_trace_raw"] / tokens
                s["h_w2_sum"] = s["h_w2_sum_raw"] / tokens

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
def load_calibration(tokenizer, source: str, n_samples: int,
                     seqlen: int) -> torch.Tensor:
    """Load calibration from a HuggingFace dataset id, a local .jsonl, or
    a local .txt file. JSONL rows can have either {"text": ...} or
    {"messages": [...]} for chat-style data.
    """
    import os
    from datasets import load_dataset

    texts: list[str] = []
    if source.endswith(".jsonl") and os.path.exists(source):
        with open(source) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "messages" in obj:
                    try:
                        texts.append(tokenizer.apply_chat_template(
                            obj["messages"], tokenize=False))
                    except Exception:
                        continue
                elif "text" in obj:
                    texts.append(obj["text"])
    elif source.endswith(".txt") and os.path.exists(source):
        with open(source) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    elif source == "ultrachat_200k":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                          split="train_sft", streaming=True)
        for row in ds:
            msgs = row.get("messages", [])
            if not msgs:
                continue
            try:
                texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
            except Exception:
                continue
            if len(texts) >= n_samples * 8:
                break
    else:
        # Generic HF dataset loader. Handles three common schemas:
        #   1. {"text": "..."} — raw text corpora (pile, wikitext, etc.)
        #   2. {"messages": [...]} — chat-format SFT (ultrachat, tulu-3, etc.)
        #   3. anything else — falls back to first string column
        # Streaming when possible so we don't download the full dataset for
        # just 32 samples.
        try:
            ds = load_dataset(source, split="train", streaming=True)
            stream = True
        except Exception:
            ds = load_dataset(source, split="train")
            stream = False

        # Probe one row to detect schema
        iterator = iter(ds) if stream else ds
        first = next(iterator) if stream else (ds[0] if len(ds) else {})
        schema = None
        if "messages" in first:
            schema = "messages"
        elif "text" in first:
            schema = "text"
        else:
            # pick first string-valued column
            for k, v in first.items():
                if isinstance(v, str):
                    schema = k
                    break
        if schema is None:
            raise ValueError(f"Could not find text or messages field in {source}")
        print(f"[probe] {source} schema: {schema}", flush=True)

        # Re-iterate (we consumed the first row)
        if stream:
            ds = load_dataset(source, split="train", streaming=True)
            iterator = iter(ds)
        else:
            iterator = iter(ds)

        for row in iterator:
            if schema == "messages":
                msgs = row.get("messages") or row.get("conversations") or []
                if not msgs:
                    continue
                try:
                    texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
                except Exception:
                    continue
            else:
                v = row.get(schema)
                if isinstance(v, str) and v.strip():
                    texts.append(v)
            if len(texts) >= n_samples * 8:
                break

    # Two-pass sampling:
    #   1) first pass picks any sample already >= seqlen tokens
    #   2) fallback packs multiple short samples together (separated by
    #      EOS) to reach seqlen. This makes SFT/chat datasets with short
    #      turns (tulu-3, glaive) usable without lowering seqlen.
    random.seed(42)
    samples = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=False).input_ids
        if ids.size(1) < seqlen:
            continue
        start = random.randint(0, ids.size(1) - seqlen)
        samples.append(ids[0, start:start + seqlen])
        if len(samples) >= n_samples:
            break

    if len(samples) < n_samples:
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        # Pack short samples by concatenating with EOS separator
        buf: list[int] = []
        for t in texts:
            ids = tokenizer(t, return_tensors="pt", truncation=False).input_ids[0].tolist()
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= seqlen and len(samples) < n_samples:
                samples.append(torch.tensor(buf[:seqlen], dtype=torch.long))
                buf = buf[seqlen:]
            if len(samples) >= n_samples:
                break

    if len(samples) < n_samples:
        print(f"[probe] warning: only got {len(samples)}/{n_samples} samples "
              f"(even with packing). Consider wider corpus.",
              flush=True)
    return torch.stack(samples[:n_samples], dim=0)


def per_token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1), reduction="none")
    return ce.view(shift_labels.size())


def load_probe_model_and_tokenizer(model_path: str,
                                   requested_device: str,
                                   dtype: torch.dtype,
                                   device_map: str | None = None,
                                   unfuse_moe: bool = True,
                                   gradient_checkpointing: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    staged = stage_text_only(model_path)
    tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
    load_device_map = device_map if device_map is not None else requested_device
    model = AutoModelForCausalLM.from_pretrained(
        staged, torch_dtype=dtype, device_map=load_device_map,
        low_cpu_mem_usage=False, trust_remote_code=True,
    )
    model.eval()

    if unfuse_moe:
        try:
            target_dev = prepare_model_for_moe_linears(model)
            if target_dev is not None:
                print(f"[probe] unfused MoE experts into per-expert linears "
                      f"(all on {target_dev})", flush=True)
            else:
                print("[probe] MoE unfuse made no changes; continuing with "
                      "packed experts.", flush=True)
        except ImportError:
            print("[probe] AutoRound not available; skipping MoE unfuse. "
                  "Per-expert sensitivity will not be measured.", flush=True)
        except Exception as e:
            print(f"[probe] MoE unfuse failed ({e}); continuing with "
                  "fused experts.", flush=True)

    exec_device = resolve_execution_device(model, requested_device)
    print(f"[probe] execution device: {exec_device} "
          f"(load device_map={load_device_map})", flush=True)

    for p in model.parameters():
        p.requires_grad_(False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    return staged, tokenizer, model, exec_device, load_device_map


def run_probe_pass(model: nn.Module,
                   tokenizer,
                   calib: torch.Tensor,
                   model_name: str,
                   dataset_name: str,
                   seqlen: int,
                   dtype_name: str,
                   requested_device: str,
                   load_device_map,
                   exec_device: torch.device,
                   linear_include: str,
                   linear_exclude: str,
                   importance_weighting: bool,
                   activation_cache_dir: str | None,
                   output_path: str):
    inc = re.compile(linear_include)
    exc = re.compile(linear_exclude)
    tracked = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear)
               and inc.search(n) and not exc.search(n)]
    print(f"[probe] tracking {len(tracked)} Linear layers", flush=True)

    expert_info_all = discover_moe_structure(model)
    expert_info = {k: v for k, v in expert_info_all.items() if k in tracked}
    top_k = read_top_k(model, default=2)
    routers = sorted({r for r, _ in expert_info.values()})
    print(f"[probe] MoE: {len(expert_info)} expert linears, "
          f"{len(routers)} routers, top_k={top_k}", flush=True)
    if len(expert_info) == 0:
        diag_count = 0
        for pname, pmod in model.named_modules():
            for attr in ("experts", "block_sparse_moe_experts",
                         "moe_experts", "expert_layer"):
                child = getattr(pmod, attr, None)
                if child is None or not isinstance(child, nn.Module):
                    continue
                kids = list(child.named_children())
                numkids = [k for k, _ in kids if k.isdigit()]
                print(f"[probe/diag] parent={pname!r} attr={attr!r} "
                      f"container_cls={type(child).__name__} "
                      f"n_children={len(kids)} n_numeric_children={len(numkids)}"
                      f" first_children={[k for k,_ in kids[:5]]}",
                      flush=True)
                diag_count += 1
                if diag_count >= 3:
                    break
            if diag_count >= 3:
                break

    tracker = RouterTracker(model, routers, top_k) if routers else None
    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    acc = FisherAccumulator(model, tracked, expert_info, cache_dir)

    print(f"[probe] calibration shape: {calib.shape}", flush=True)

    model.train()
    t_fwd = t_bwd = 0.0
    for i in range(calib.size(0)):
        ids = calib[i:i+1].to(exec_device)
        t0 = time.time()
        with torch.no_grad():
            embed = model.get_input_embeddings()(ids)
        embed.requires_grad_(True)
        out = model(inputs_embeds=embed, labels=ids)
        logits = out.logits
        t_fwd += time.time() - t0

        t0 = time.time()
        if importance_weighting:
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
        loss.backward()
        t_bwd += time.time() - t0

        if (i + 1) % 4 == 0 or i == 0:
            print(f"[probe] sample {i+1}/{calib.size(0)} "
                  f"loss={float(loss.item()):.3f} "
                  f"fwd_avg={t_fwd/(i+1):.2f}s bwd_avg={t_bwd/(i+1):.2f}s",
                  flush=True)

        del out, loss, ids, embed, logits
        acc._saved_inputs.clear()
        if exec_device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    acc.finalize(tracker)
    acc.remove_hooks()
    if tracker is not None:
        tracker.remove_hooks()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "stats": acc.stats,
            "router_counts": dict(tracker.counts) if tracker else {},
            "router_totals": dict(tracker.total_tokens) if tracker else {},
            "expert_info": expert_info,
            "meta": {
                "model": model_name,
                "dataset": dataset_name,
                "nsamples": calib.size(0),
                "seqlen": seqlen,
                "dtype": dtype_name,
                "device_map": str(load_device_map),
                "execution_device": str(exec_device),
                "top_k": top_k,
                "importance_weighting": importance_weighting,
                "activation_cache_dir": str(cache_dir) if cache_dir else None,
                "linear_include": linear_include,
                "linear_exclude": linear_exclude,
            },
        }, f)
    print(f"[probe] wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Measure per-Linear sensitivity (Fisher trace) with "
                    "route-aware MoE weighting and per-token importance.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ultrachat_200k",
                    help="HF dataset name, or path to .jsonl/.txt")
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None,
                    help="HF from_pretrained device_map. Defaults to --device. "
                         "Use 'auto' to allow CPU/GPU model sharding while still "
                         "running the probe on the embedding device.")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output", required=True,
                    help="Pickle with per-Linear stats")
    ap.add_argument("--activation-cache-dir", default=None,
                    help="Save per-Linear input activation snapshots here "
                         "(for measure_quant_cost.py)")
    ap.add_argument("--linear-include", default=".*")
    ap.add_argument("--linear-exclude",
                    default=r"(?:^lm_head$|\.lm_head$|mlp\.gate$|"
                            r"mlp\..*gate$|\.router(?:$|\.)|"
                            r"block_sparse_moe\.gate$)")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    default=True)
    ap.add_argument("--no-gradient-checkpointing", action="store_false",
                    dest="gradient_checkpointing")
    ap.add_argument("--importance-weighting", action="store_true", default=True)
    ap.add_argument("--no-importance-weighting", action="store_false",
                    dest="importance_weighting")
    ap.add_argument("--unfuse-moe", action="store_true", default=True,
                    help="Unfuse MoE expert tensors into per-expert nn.Linear "
                         "via AutoRound. Needed for any model that uses the "
                         "transformers 5+ fused-experts pattern (Qwen3.x MoE, "
                         "Mixtral 5+, etc.). Requires auto-round installed.")
    ap.add_argument("--no-unfuse-moe", action="store_false", dest="unfuse_moe")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    print(f"[probe] loading {args.model}", flush=True)
    t0 = time.time()
    _, tokenizer, model, exec_device, load_device_map = load_probe_model_and_tokenizer(
        args.model,
        requested_device=args.device,
        dtype=dtype,
        device_map=args.device_map,
        unfuse_moe=args.unfuse_moe,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    print(f"[probe] loaded in {time.time()-t0:.1f}s", flush=True)

    calib = load_calibration(tokenizer, args.dataset, args.nsamples, args.seqlen)
    run_probe_pass(
        model=model,
        tokenizer=tokenizer,
        calib=calib,
        model_name=args.model,
        dataset_name=args.dataset,
        seqlen=args.seqlen,
        dtype_name=args.dtype,
        requested_device=args.device,
        load_device_map=load_device_map,
        exec_device=exec_device,
        linear_include=args.linear_include,
        linear_exclude=args.linear_exclude,
        importance_weighting=args.importance_weighting,
        activation_cache_dir=args.activation_cache_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
