#!/usr/bin/env python3
"""
DPQ with AutoRound first — "escalate only if the best AutoRound at each tier isn't enough".

The principle: DPQ's escalation decisions should compare the *best* version
of each candidate format, not a suboptimal baseline. Using RTN-FP8 as the
"upgrade from FP4" option would unfairly penalize FP8 (making DPQ escalate
all the way to BF16 when AutoRound-FP8 would have sufficed). So we run
AutoRound at each candidate tier first, then let DPQ choose among the
best-at-every-tier versions.

Pipeline:

  Stage 1: Cache BF16 reference logits from the original model.
  Stage 2: Save original BF16 Linear weights to CPU.
  Stage 3a: Run AutoRound at NVFP4 on the whole model → save fp4_weights dict.
            Restore original BF16 weights.
  Stage 3b: Run AutoRound at FP8 on the whole model → save fp8_weights dict.
            Restore original BF16 weights.
  Stage 4: Install EscalationLinear wrappers. Each wrapper holds three
           weights on GPU (or sourced from CPU dicts):
             - AutoRound-FP4 (cheapest — cost 1)
             - AutoRound-FP8 (medium — cost 2)
             - original BF16 (most expensive — cost 4)
           Per-Linear learnable selector picks among the three.
  Stage 5: Gradient descent with efficiency-threshold Lagrangian picks
           format per Linear. DPQ upgrades a Linear only when the marginal
           quality gain justifies the cost, knowing that each candidate is
           at its quality ceiling.
  Stage 6: Hard commit, unwrap, save as BF16 checkpoint (simulated).

Memory footprint (current): 3× model size in CPU (original + fp4 + fp8 dicts).
OK for ≤7B; for larger models we can stream one tier at a time.

Extension path (future): the same pipeline generalizes to FP1..FP15 as
kernel support arrives — just run AutoRound at each target bit-width and
include them in the candidate set. The "use best version of each candidate"
principle becomes more important at finer granularity, not less.

Usage:
    python3 dpq_autoround_first.py \\
        --model <path-to-bf16-model> \\
        --output <output-dir> \\
        --min-efficiency 0.25 \\
        --autoround-iters 100 \\
        --autoround-nsamples 32 \\
        --autoround-seqlen 256 \\
        --autoround-dataset mbpp \\
        --dpq-steps 100 \\
        --dpq-calib-samples 8 \\
        --dpq-calib-seqlen 128
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Quantizers
# ---------------------------------------------------------------------------

def _fp8_round(weight: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 round-trip with per-output-channel scale."""
    max_abs = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 448.0
    return ((weight / scale).to(torch.float8_e4m3fn).to(weight.dtype)) * scale


def _nvfp4_round_rtn(weight: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Per-group NVFP4 (E2M1) round-to-nearest. This is the WORST-CASE FP4 — used
    only to compute a fixed normalization baseline so that efficiency thresholds
    mean the same thing across different pipelines."""
    out_f, in_f = weight.shape
    n_groups = (in_f + group_size - 1) // group_size
    pad = n_groups * group_size - in_f
    w = F.pad(weight, (0, pad)) if pad > 0 else weight
    grouped = w.view(out_f, n_groups, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = grouped / scales * 6.0
    abs_n = normalized.abs()
    sign = normalized.sign()
    q = torch.where(abs_n <= 0.25, torch.zeros_like(abs_n),
        torch.where(abs_n <= 0.75, torch.full_like(abs_n, 0.5),
        torch.where(abs_n <= 1.25, torch.full_like(abs_n, 1.0),
        torch.where(abs_n <= 1.75, torch.full_like(abs_n, 1.5),
        torch.where(abs_n <= 2.5,  torch.full_like(abs_n, 2.0),
        torch.where(abs_n <= 3.5,  torch.full_like(abs_n, 3.0),
        torch.where(abs_n <= 5.0,  torch.full_like(abs_n, 4.0),
                                   torch.full_like(abs_n, 6.0))))))))
    deq = sign * q * scales / 6.0
    return deq.view(out_f, -1)[:, :in_f]


# Relative cost vs FP4 (on Blackwell: memory AND compute scale together)
FORMAT_COSTS = {"fp4": 1.0, "fp8": 2.0, "bf16": 4.0}
CANDIDATES: List[str] = ["fp4", "fp8", "bf16"]


# Always-skip patterns (structural)
ALWAYS_SKIP_PATTERNS = [
    "lm_head", "embed_tokens", "mlp.gate$", "shared_expert_gate",
    "norm", "A_log", "dt_bias", "conv1d",
]


def should_always_skip(name: str) -> bool:
    import re
    for pattern in ALWAYS_SKIP_PATTERNS:
        if pattern.endswith("$"):
            if re.search(pattern, name):
                return True
        elif pattern in name:
            return True
    return False


# ---------------------------------------------------------------------------
# Escalation-selector Linear
# ---------------------------------------------------------------------------

class EscalationLinear(nn.Module):
    """
    Per-Linear format selector holding all three candidate weights:
        fp4_weight  — AutoRound-FP4 optimized
        fp8_weight  — AutoRound-FP8 optimized
        bf16_weight — original unquantized

    Forward uses soft mixture of the three format outputs, weighted by the
    softmax of per-tensor logits. Backward flows gradients through the
    logits via the mixture coefficients.

    Memory: 3× per-Linear weight on GPU. For ≤7B models this is fine; for
    larger models we'd stream one weight at a time.
    """

    def __init__(
        self,
        linear: nn.Linear,
        fp4_weight: torch.Tensor,
        fp8_weight: torch.Tensor,
        bf16_weight: torch.Tensor,
    ):
        super().__init__()
        device = linear.weight.device
        dtype = linear.weight.dtype
        self.register_buffer("fp4_weight", fp4_weight.clone().to(device, dtype))
        self.register_buffer("fp8_weight", fp8_weight.clone().to(device, dtype))
        self.register_buffer("bf16_weight", bf16_weight.clone().to(device, dtype))

        if linear.bias is not None:
            self.register_buffer("bias_buffer", linear.bias.data.clone())
        else:
            self.bias_buffer = None
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.logits = nn.Parameter(torch.zeros(len(CANDIDATES)))
        self.tau: float = 1.0
        self._force_format: Optional[str] = None

    def _weight_for(self, fmt: str) -> torch.Tensor:
        if fmt == "fp4":
            return self.fp4_weight
        if fmt == "fp8":
            return self.fp8_weight
        if fmt == "bf16":
            return self.bf16_weight
        raise ValueError(fmt)

    def soft_probs(self) -> torch.Tensor:
        return F.softmax(self.logits / self.tau, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._force_format is not None:
            qw = self._weight_for(self._force_format)
            return F.linear(x, qw, self.bias_buffer)

        probs = self.soft_probs()

        # Sequential soft mixture
        out = None
        for k, fmt in enumerate(CANDIDATES):
            qw = self._weight_for(fmt)
            partial = F.linear(x, qw, None)
            term = probs[k] * partial
            out = term if out is None else out + term
            # qw is a reference to a buffer or a freshly-computed fp8 tensor.
            # We don't force-delete; Python's ref counting will collect the
            # fp8 temporary after this iteration.

        if self.bias_buffer is not None:
            out = out + self.bias_buffer
        return out

    def expected_cost(self) -> torch.Tensor:
        probs = self.soft_probs()
        costs = torch.tensor(
            [FORMAT_COSTS[c] for c in CANDIDATES],
            device=probs.device, dtype=torch.float32,
        )
        return (probs * costs).sum()

    def hard_choice(self) -> str:
        return CANDIDATES[int(self.logits.argmax().item())]

    def force(self, fmt: Optional[str]) -> None:
        self._force_format = fmt

    def commit(self) -> str:
        """Replace the fp4_weight buffer with the hard-chosen format's weight.
        After commit, the Linear's effective weight is the chosen format's
        materialization, stored as bf16."""
        chosen = self.hard_choice()
        with torch.no_grad():
            self.fp4_weight.copy_(self._weight_for(chosen))
        return chosen


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------

def replace_with_escalation(
    model: nn.Module,
    fp4_weights: Dict[str, torch.Tensor],
    fp8_weights: Dict[str, torch.Tensor],
    bf16_weights: Dict[str, torch.Tensor],
    min_numel: int = 1000,
) -> Dict[str, EscalationLinear]:
    """Replace each sufficiently-large nn.Linear with an EscalationLinear
    that holds all three candidate weights (AutoRound-FP4, AutoRound-FP8,
    original BF16)."""
    wrappers: Dict[str, EscalationLinear] = {}

    def _replace(parent: nn.Module, prefix: str = ""):
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and child.weight.numel() >= min_numel:
                if should_always_skip(full):
                    continue
                if full not in fp4_weights or full not in fp8_weights or full not in bf16_weights:
                    continue
                wrapper = EscalationLinear(
                    child,
                    fp4_weight=fp4_weights[full],
                    fp8_weight=fp8_weights[full],
                    bf16_weight=bf16_weights[full],
                )
                wrapper = wrapper.to(child.weight.device, dtype=child.weight.dtype)
                wrapper.logits.data = wrapper.logits.data.float()
                setattr(parent, name, wrapper)
                wrappers[full] = wrapper
            else:
                _replace(child, full)

    _replace(model)
    return wrappers


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_wikitext_calibration(
    tokenizer, n_samples: int, seq_len: int, seed: int = 42,
) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    enc = tokenizer(full_text, return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    max_start = enc.size(0) - seq_len
    starts = torch.randint(0, max_start, (n_samples,), generator=g)
    return torch.stack([enc[s:s + seq_len] for s in starts])


def kl_divergence(student_logits: torch.Tensor, teacher_log_probs: torch.Tensor) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return kl.mean()


@torch.no_grad()
def cache_reference_logits(
    model: nn.Module, calib_ids: torch.Tensor, device: torch.device,
) -> List[torch.Tensor]:
    refs = []
    for i in range(calib_ids.size(0)):
        batch = calib_ids[i:i+1].to(device)
        logits = model(batch).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        refs.append(log_probs)
    return refs


@torch.no_grad()
def measure_kl(
    model: nn.Module,
    calib_ids: torch.Tensor,
    ref_log_probs: List[torch.Tensor],
    device: torch.device,
) -> float:
    kls = []
    for i in range(calib_ids.size(0)):
        batch = calib_ids[i:i+1].to(device)
        logits = model(batch).logits
        kls.append(kl_divergence(logits, ref_log_probs[i]).item())
    return sum(kls) / len(kls)


# ---------------------------------------------------------------------------
# AutoRound output cache
# ---------------------------------------------------------------------------

def _cache_paths(cache_dir: str) -> Dict[str, Path]:
    p = Path(cache_dir)
    return {
        "root": p,
        "fp4": p / "fp4_weights.safetensors",
        "fp8": p / "fp8_weights.safetensors",
        "meta": p / "cache_manifest.json",
    }


def _cache_is_valid(cache_dir: str) -> bool:
    """Returns True if the cache has BOTH fp4 and fp8 weights + manifest."""
    if not cache_dir:
        return False
    paths = _cache_paths(cache_dir)
    return paths["fp4"].exists() and paths["fp8"].exists() and paths["meta"].exists()


def _cache_has_fp4(cache_dir: str) -> bool:
    if not cache_dir:
        return False
    paths = _cache_paths(cache_dir)
    return paths["fp4"].exists() and paths["meta"].exists()


def _save_weights_to_cache(cache_dir: str, key: str,
                           weights: Dict[str, torch.Tensor]):
    """Save one tier's weights to the cache without touching the others."""
    from safetensors.torch import save_file
    paths = _cache_paths(cache_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in weights.items()}, str(paths[key]))


def _update_cache_meta(cache_dir: str, updates: dict):
    """Merge updates into the cache manifest."""
    paths = _cache_paths(cache_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    meta = {}
    if paths["meta"].exists():
        with open(paths["meta"]) as f:
            meta = json.load(f)
    meta.update(updates)
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)


def _load_autoround_cache(cache_dir: str):
    from safetensors.torch import load_file
    paths = _cache_paths(cache_dir)
    fp4_weights = load_file(str(paths["fp4"]))
    fp8_weights = load_file(str(paths["fp8"]))
    with open(paths["meta"]) as f:
        metadata = json.load(f)
    return fp4_weights, fp8_weights, metadata


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    model_path: str,
    output_dir: str,
    *,
    min_efficiency: float = 0.25,
    cache_dir: Optional[str] = None,
    autoround_iters: int = 100,
    autoround_nsamples: int = 32,
    autoround_seqlen: int = 256,
    autoround_batch_size: int = 4,
    autoround_dataset: str = "mbpp",
    dpq_steps: int = 100,
    dpq_lr: float = 0.05,
    dpq_tau_start: float = 1.5,
    dpq_tau_end: float = 0.1,
    dpq_calib_samples: int = 8,
    dpq_calib_seqlen: int = 128,
    hadamard: bool = True,
    cache_only: bool = False,
    stages: str = "all",  # "all", "fp4", "fp8"
    verbose: bool = True,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def log(msg):
        if verbose:
            print(msg, flush=True)

    t_start = time.time()

    # Vision/multimodal workaround: stage a writable copy of the model dir
    # via symlinks, with a stripped+flattened config.json. This lets us load
    # a multimodal Qwen3.5 checkpoint via AutoModelForCausalLM without
    # touching the source directory (important when source is in /models
    # which is read-only).
    import tempfile, shutil
    src_config_path = Path(model_path) / "config.json"
    staged_model_path = model_path
    staged_dir = None
    if src_config_path.exists():
        with open(src_config_path) as f:
            config_data = json.load(f)
        is_multimodal = "vision_config" in config_data or "text_config" in config_data
        if is_multimodal:
            # Strip vision bits
            for key in ["vision_config", "image_token_id", "video_token_id",
                        "vision_start_token_id", "vision_end_token_id"]:
                config_data.pop(key, None)
            # Promote text_config to top level (transformers ForCausalLM
            # expects flat config)
            if "text_config" in config_data:
                text_cfg = config_data.pop("text_config")
                for k, v in text_cfg.items():
                    if k not in config_data:
                        config_data[k] = v
                # Ensure model_type matches the text model
                if "model_type" in text_cfg:
                    config_data["model_type"] = text_cfg["model_type"]
            # Force architectures to the ForCausalLM variant if present
            archs = config_data.get("architectures", [])
            if archs:
                config_data["architectures"] = [
                    a.replace("ForConditionalGeneration", "ForCausalLM") for a in archs
                ]
            staged_dir = tempfile.mkdtemp(prefix="dpq_af_staged_")
            for p in Path(model_path).iterdir():
                if p.name == "config.json":
                    continue
                (Path(staged_dir) / p.name).symlink_to(p.resolve())
            with open(Path(staged_dir) / "config.json", "w") as f:
                json.dump(config_data, f, indent=2)
            staged_model_path = staged_dir

    try:
        # ==============================================================
        # Stage 1: Load source model and cache BF16 reference logits
        # ==============================================================
        log(f"[pipe] loading source model from {model_path}"
            f"{' (staged at '+staged_model_path+')' if staged_dir else ''}")
        model = AutoModelForCausalLM.from_pretrained(
            staged_model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(staged_model_path, trust_remote_code=True)
        device = next(model.parameters()).device
        log(f"[pipe] model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

        log(f"[pipe] loading wikitext calibration ({dpq_calib_samples} × {dpq_calib_seqlen})")
        calib_ids = load_wikitext_calibration(tokenizer, dpq_calib_samples, dpq_calib_seqlen)

        log(f"[pipe] Stage 1: caching BF16 reference logits")
        ref_log_probs = cache_reference_logits(model, calib_ids, device)

        # ==============================================================
        # Stage 2: Save original BF16 weights for later DPQ trials
        # In cache_only mode, skip this: we reload the model from disk
        # between stages instead, to save ~1× model-size of CPU memory.
        # (Important for 27B+ where holding a second copy OOMs.)
        # ==============================================================
        original_weights: Dict[str, torch.Tensor] = {}
        if not cache_only:
            log(f"[pipe] Stage 2: saving original BF16 weights to CPU")
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Linear) and mod.weight.numel() >= 1000:
                    if should_always_skip(name):
                        continue
                    original_weights[name] = mod.weight.data.cpu().clone()
            log(f"[pipe]   saved {len(original_weights)} Linear weights")
        else:
            log(f"[pipe] Stage 2: skipped (cache_only mode — will reload from disk between stages)")

        def _reload_model_weights():
            """Reload the model's weights from the staged model dir, in place.
            Used in cache_only mode after stages that mutate the weights, to
            avoid holding a CPU copy of originals.

            Reads safetensors files directly and copies into existing tensors,
            so no second full-model copy is ever resident. Handles the
            multimodal `model.language_model.*` → `model.*` prefix remapping
            that happens when loading via `AutoModelForCausalLM`."""
            from safetensors import safe_open
            log(f"[pipe]   reloading model weights from {staged_model_path}")
            state = model.state_dict()
            n_copied = 0
            n_missing = 0
            staged_p = Path(staged_model_path)
            shard_files = sorted(staged_p.glob("*.safetensors"))
            for shard in shard_files:
                with safe_open(str(shard), framework="pt", device="cuda") as f:
                    for shard_key in f.keys():
                        # Try the raw key first, then strip 'language_model.'
                        # which multimodal Qwen3.5 shards have but ForCausalLM
                        # doesn't.
                        model_key = None
                        if shard_key in state:
                            model_key = shard_key
                        elif shard_key.startswith("model.language_model."):
                            candidate = "model." + shard_key[len("model.language_model."):]
                            if candidate in state:
                                model_key = candidate
                        if model_key is None:
                            n_missing += 1
                            continue
                        with torch.no_grad():
                            state[model_key].copy_(f.get_tensor(shard_key))
                            n_copied += 1
            log(f"[pipe]     reloaded {n_copied} tensors from {len(shard_files)} shards "
                f"({n_missing} keys not matched)")
            gc.collect()
            torch.cuda.empty_cache()

        run_fp4 = stages in ("all", "fp4")
        run_fp8 = stages in ("all", "fp8")
        # Cache check: if valid full cache exists, skip AutoRound stages
        cache_hit = _cache_is_valid(cache_dir) if cache_dir else False
        if cache_hit and stages == "all":
            log(f"[pipe] cache HIT at {cache_dir}: skipping AutoRound stages")
            fp4_weights, fp8_weights, cache_meta = _load_autoround_cache(cache_dir)
            rtn_fp4_baseline_kl = cache_meta["rtn_fp4_baseline_kl"]
            autoround_fp4_kl = cache_meta["autoround_fp4_kl"]
            autoround_fp8_kl = cache_meta["autoround_fp8_kl"]
            log(f"[pipe]   loaded {len(fp4_weights)} FP4 weights, {len(fp8_weights)} FP8 weights")
            log(f"[pipe]   RTN-FP4 baseline KL = {rtn_fp4_baseline_kl:.6f}")
            log(f"[pipe]   AutoRound-FP4 KL    = {autoround_fp4_kl:.6f}")
            log(f"[pipe]   AutoRound-FP8 KL    = {autoround_fp8_kl:.6f}")
        else:
            if cache_dir:
                log(f"[pipe] cache MISS at {cache_dir}: running {stages} stages, will save incrementally")

            # Try to reuse cached rtn_fp4_baseline_kl if present
            cached_meta = {}
            meta_path = _cache_paths(cache_dir)["meta"] if cache_dir else None
            if meta_path and meta_path.exists():
                with open(meta_path) as f:
                    cached_meta = json.load(f)

            # ==============================================================
            # Stage 2b: Measure RTN-FP4 baseline KL (fixed reference for DPQ
            # normalization — worst-case FP4, used so that min_efficiency has
            # the same meaning across pipelines regardless of what upgrades
            # are already applied to the "current" state)
            # ==============================================================
            if "rtn_fp4_baseline_kl" in cached_meta and stages == "fp8":
                rtn_fp4_baseline_kl = cached_meta["rtn_fp4_baseline_kl"]
                log(f"[pipe] Stage 2b: skipped (using cached baseline {rtn_fp4_baseline_kl:.6f})")
            else:
                log(f"[pipe] Stage 2b: measuring RTN-FP4 baseline KL (fixed reference)")
                for name, mod in model.named_modules():
                    if isinstance(mod, nn.Linear) and mod.weight.numel() >= 1000:
                        if should_always_skip(name):
                            continue
                        mod.weight.data = _nvfp4_round_rtn(mod.weight.data)
                rtn_fp4_baseline_kl = measure_kl(model, calib_ids, ref_log_probs, device)
                log(f"[pipe]   RTN-FP4 baseline KL (fixed denominator) = {rtn_fp4_baseline_kl:.6f}")
                # Restore originals before Stage 3
                if cache_only:
                    _reload_model_weights()
                else:
                    for name, mod in model.named_modules():
                        if isinstance(mod, nn.Linear) and name in original_weights:
                            mod.weight.data.copy_(original_weights[name].to(device))
                if cache_dir:
                    _update_cache_meta(cache_dir, {
                        "source_model": model_path,
                        "autoround_iters": autoround_iters,
                        "autoround_nsamples": autoround_nsamples,
                        "autoround_seqlen": autoround_seqlen,
                        "autoround_dataset": autoround_dataset,
                        "hadamard": hadamard,
                        "rtn_fp4_baseline_kl": rtn_fp4_baseline_kl,
                    })

            sys.path.insert(0, "/tmp/auto-round")
            from auto_round.compressors import LLMCompressor

            # Build layer_config that skips structural layers; rest use global config
            def _layer_config_skip_structural() -> Dict[str, dict]:
                lc = {}
                for name, mod in model.named_modules():
                    if not isinstance(mod, nn.Linear):
                        continue
                    if mod.weight.numel() < 1000:
                        continue
                    if should_always_skip(name):
                        lc[name] = {"bits": 16, "group_size": -1, "data_type": "int"}
                return lc

            layer_config = _layer_config_skip_structural()

            # ==============================================================
            # Stage 3a: Run AutoRound at NVFP4 on the whole model
            # ==============================================================
            fp4_weights: Dict[str, torch.Tensor] = {}
            autoround_fp4_kl = cached_meta.get("autoround_fp4_kl", float("nan"))
            if run_fp4:
                log(f"[pipe] Stage 3a: running AutoRound at NVFP4 "
                    f"(iters={autoround_iters}, nsamples={autoround_nsamples}, "
                    f"seqlen={autoround_seqlen}, hadamard={hadamard})")
                ar_fp4_kwargs = dict(
                    model=model,
                    tokenizer=tokenizer,
                    bits=4,
                    group_size=16,
                    sym=True,
                    data_type="nv_fp4",
                    batch_size=autoround_batch_size,
                    seqlen=autoround_seqlen,
                    nsamples=autoround_nsamples,
                    iters=autoround_iters,
                    dataset=autoround_dataset,
                    layer_config=layer_config,
                )
                if hadamard:
                    ar_fp4_kwargs["hadamard_config"] = "random_hadamard"
                autoround_fp4 = LLMCompressor(**ar_fp4_kwargs)
                t0 = time.time()
                autoround_fp4.quantize()
                log(f"[pipe]   AutoRound-FP4 done in {time.time() - t0:.0f}s")
                del autoround_fp4
                gc.collect()
                torch.cuda.empty_cache()

                # AutoRound may have moved weights to CPU during optimization — bring back
                model.to(device)

                # Measure AutoRound-FP4 KL BEFORE extracting weights (model still intact)
                autoround_fp4_kl = measure_kl(model, calib_ids, ref_log_probs, device)
                log(f"[pipe]   AutoRound-FP4 KL vs BF16 = {autoround_fp4_kl:.6f}")

                # Extract + evict: for each Linear, clone its weight, then replace
                # the module's weight with a tiny placeholder to free the original
                # storage. Keeps peak memory flat during extraction (vs doubling).
                log(f"[pipe]   extracting FP4 weights (in-place eviction to avoid OOM)")
                for name, mod in model.named_modules():
                    if isinstance(mod, nn.Linear) and mod.weight.numel() >= 1000:
                        if should_always_skip(name):
                            continue
                        fp4_weights[name] = mod.weight.data.cpu().clone()
                        if cache_only:
                            # Replace with tiny placeholder to free original storage.
                            # Safe because in cache_only mode we reload model before
                            # the next stage anyway.
                            mod.weight.data = torch.zeros(
                                1, dtype=mod.weight.dtype, device=mod.weight.device,
                            )
                log(f"[pipe]   saved {len(fp4_weights)} FP4 weights (model storage evicted)")

                # ** INCREMENTAL SAVE ** — persist fp4 weights NOW so we don't
                # lose work if Stage 3b fails later
                if cache_dir:
                    log(f"[pipe] incremental save: fp4 weights → cache")
                    _save_weights_to_cache(cache_dir, "fp4", fp4_weights)
                    _update_cache_meta(cache_dir, {"autoround_fp4_kl": autoround_fp4_kl})
                    log(f"[pipe]   fp4 save complete")

                # Free the extracted dict — we don't need it in this process again
                if cache_only:
                    fp4_weights.clear()
                    gc.collect()
                    torch.cuda.empty_cache()

                # Restore original BF16 weights for the next AutoRound pass
                log(f"[pipe]   restoring original BF16 weights for next stage")
                if cache_only:
                    _reload_model_weights()
                else:
                    for name, mod in model.named_modules():
                        if isinstance(mod, nn.Linear) and name in original_weights:
                            mod.weight.data.copy_(original_weights[name].to(device))

            # ==============================================================
            # Stage 3b: Run AutoRound at FP8 on the whole model
            # ==============================================================
            fp8_weights: Dict[str, torch.Tensor] = {}
            autoround_fp8_kl = cached_meta.get("autoround_fp8_kl", float("nan"))
            if run_fp8:
                log(f"[pipe] Stage 3b: running AutoRound at FP8")
                autoround_fp8 = LLMCompressor(
                    model=model,
                    tokenizer=tokenizer,
                    bits=8,
                    group_size=-1,           # per-output-channel FP8
                    sym=True,
                    data_type="fp8",
                    batch_size=autoround_batch_size,
                    seqlen=autoround_seqlen,
                    nsamples=autoround_nsamples,
                    iters=autoround_iters,
                    dataset=autoround_dataset,
                    layer_config=layer_config,
                )
                t0 = time.time()
                autoround_fp8.quantize()
                log(f"[pipe]   AutoRound-FP8 done in {time.time() - t0:.0f}s")
                del autoround_fp8
                gc.collect()
                torch.cuda.empty_cache()

                model.to(device)

                # Measure FP8 KL before eviction
                autoround_fp8_kl = measure_kl(model, calib_ids, ref_log_probs, device)
                log(f"[pipe]   AutoRound-FP8 KL vs BF16 = {autoround_fp8_kl:.6f}")

                # Same extract-and-evict pattern as FP4 pass
                log(f"[pipe]   extracting FP8 weights (in-place eviction)")
                for name, mod in model.named_modules():
                    if isinstance(mod, nn.Linear) and mod.weight.numel() >= 1000:
                        if should_always_skip(name):
                            continue
                        fp8_weights[name] = mod.weight.data.cpu().clone()
                        if cache_only:
                            mod.weight.data = torch.zeros(
                                1, dtype=mod.weight.dtype, device=mod.weight.device,
                            )
                log(f"[pipe]   saved {len(fp8_weights)} FP8 weights (model storage evicted)")

                if cache_dir:
                    log(f"[pipe] incremental save: fp8 weights → cache")
                    _save_weights_to_cache(cache_dir, "fp8", fp8_weights)
                    _update_cache_meta(cache_dir, {"autoround_fp8_kl": autoround_fp8_kl})
                    log(f"[pipe]   fp8 save complete")

                if cache_only:
                    fp8_weights.clear()
                    gc.collect()
                    torch.cuda.empty_cache()

        if cache_only:
            log(f"[pipe] cache-only mode: exiting before DPQ")
            log(f"[pipe] DONE in {time.time() - t_start:.0f}s")
            return

        # If we only ran fp4 or fp8 (stages != "all") and we're not cache_only,
        # we can't proceed to DPQ without both tiers. Bail out cleanly.
        if stages != "all":
            log(f"[pipe] stages={stages} selected, exiting before DPQ")
            log(f"[pipe] DONE in {time.time() - t_start:.0f}s")
            return

        # Restore original BF16 for DPQ starting state
        log(f"[pipe]   restoring original BF16 weights for DPQ")
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and name in original_weights:
                mod.weight.data.copy_(original_weights[name].to(device))

        # ==============================================================
        # Stage 5: Install EscalationLinear wrappers
        # ==============================================================
        log(f"[pipe] Stage 5: installing EscalationLinear wrappers (3 weights per Linear)")
        wrappers = replace_with_escalation(
            model,
            fp4_weights=fp4_weights,
            fp8_weights=fp8_weights,
            bf16_weights=original_weights,
        )
        n_wrappers = len(wrappers)
        log(f"[pipe]   {n_wrappers} wrappers installed")

        # Sanity checks: pin wrappers to each format and verify KL matches
        # our earlier measurements
        for w in wrappers.values():
            w.force("fp4")
        wrapper_fp4_kl = measure_kl(model, calib_ids, ref_log_probs, device)
        for w in wrappers.values():
            w.force("fp8")
        wrapper_fp8_kl = measure_kl(model, calib_ids, ref_log_probs, device)
        for w in wrappers.values():
            w.force("bf16")
        wrapper_bf16_kl = measure_kl(model, calib_ids, ref_log_probs, device)
        for w in wrappers.values():
            w.force(None)
        log(f"[pipe]   sanity: fp4={wrapper_fp4_kl:.5f} fp8={wrapper_fp8_kl:.5f} "
            f"bf16={wrapper_bf16_kl:.5f} "
            f"(expect fp4 ≈ {autoround_fp4_kl:.5f}, fp8 ≈ {autoround_fp8_kl:.5f}, bf16 ≈ 0)")

        # Use the FIXED RTN-FP4 baseline as the normalization denominator,
        # NOT the current AutoRound-FP4 KL. This ensures that min_efficiency
        # has the same interpretation across pipelines and baselines: a given
        # numeric threshold corresponds to the same absolute trade-off no
        # matter how much the starting state has already been improved.
        #
        # Using wrapper_fp4_kl here would make the threshold looser as the
        # baseline improves (because the same absolute KL reduction looks
        # proportionally larger against a smaller denominator), causing
        # over-escalation. The fixed RTN reference avoids this.
        baseline_kl = rtn_fp4_baseline_kl

        log(f"[pipe]   running gradient descent (min_efficiency={min_efficiency}, "
            f"steps={dpq_steps})")
        params = [w.logits for w in wrappers.values()]
        optimizer = torch.optim.Adam(params, lr=dpq_lr)

        history = []
        n_batches = calib_ids.size(0)

        for step in range(dpq_steps):
            tau = dpq_tau_start * (dpq_tau_end / dpq_tau_start) ** (step / max(1, dpq_steps - 1))
            for w in wrappers.values():
                w.tau = tau

            batch_idx = step % n_batches
            batch = calib_ids[batch_idx:batch_idx+1].to(device)
            teacher = ref_log_probs[batch_idx]

            optimizer.zero_grad()
            student_logits = model(batch).logits
            kl = kl_divergence(student_logits, teacher)

            # Normalize KL by baseline: 1.0 = AutoRound-FP4, 0.0 = perfect
            normalized_kl = kl / max(baseline_kl, 1e-8)

            # Cost excess above FP4
            total_cost_excess = sum(
                w.expected_cost() - 1.0 for w in wrappers.values()
            )
            avg_cost_excess = total_cost_excess / n_wrappers

            loss = normalized_kl + min_efficiency * avg_cost_excess

            loss.backward()
            optimizer.step()

            if verbose and (step % 10 == 0 or step == dpq_steps - 1):
                with torch.no_grad():
                    counts = {f: 0 for f in CANDIDATES}
                    for w in wrappers.values():
                        counts[w.hard_choice()] += 1
                    hard_cost = sum(
                        FORMAT_COSTS[w.hard_choice()] for w in wrappers.values()
                    ) / n_wrappers
                log(f"  step {step:4d}/{dpq_steps}  τ={tau:.3f}  "
                    f"KL={kl.item():.5f}  norm_KL={normalized_kl.item():.4f}  "
                    f"E[cost-1]={avg_cost_excess.item():.3f}  "
                    f"hard[cost]={hard_cost:.2f}  {counts}")

            history.append({
                "step": step, "tau": tau,
                "kl": kl.item(), "normalized_kl": normalized_kl.item(),
                "avg_cost_excess": avg_cost_excess.item(),
                "loss": loss.item(),
            })

        # ==============================================================
        # Stage 6: Hard commit and measure
        # ==============================================================
        log(f"[pipe] Stage 6: hard committing escalation choices")
        decisions: Dict[str, str] = {}
        counts: Dict[str, int] = {f: 0 for f in CANDIDATES}
        for name, w in wrappers.items():
            chosen = w.commit()
            decisions[name] = chosen
            counts[chosen] += 1

        # After commit, the wrappers' fp4_weight buffer contains the
        # chosen format's materialization. The Linear forward still goes
        # through EscalationLinear though — for clean save we need to
        # unwrap back to nn.Linear.

        # Measure final KL with committed choices
        for w in wrappers.values():
            w.force("fp4")  # fp4 now = the committed weight
        final_kl = measure_kl(model, calib_ids, ref_log_probs, device)

        # Compute two gap closure metrics for diagnostics
        gap_closure_vs_rtn = 1.0 - final_kl / max(rtn_fp4_baseline_kl, 1e-8)
        gap_closure_vs_ar = 1.0 - final_kl / max(autoround_fp4_kl, 1e-8)
        log(f"[pipe]   final counts: {counts}")
        log(f"[pipe]   RTN-FP4 baseline KL (denominator): {rtn_fp4_baseline_kl:.5f}")
        log(f"[pipe]   AutoRound-FP4 KL: {autoround_fp4_kl:.5f}")
        log(f"[pipe]   final KL: {final_kl:.5f}")
        log(f"[pipe]   gap closure vs RTN-FP4:       {gap_closure_vs_rtn * 100:.1f}%")
        log(f"[pipe]   gap closure vs AutoRound-FP4: {gap_closure_vs_ar * 100:.1f}%")
        # Keep the old variable name for the manifest
        baseline_kl = rtn_fp4_baseline_kl
        gap_closure = gap_closure_vs_rtn

        avg_cost = sum(FORMAT_COSTS[c] for c in decisions.values()) / max(1, n_wrappers)
        log(f"[pipe]   avg cost (AutoRound-FP4=1): {avg_cost:.2f}")

        # ==============================================================
        # Stage 7: Unwrap and save as standard BF16 checkpoint
        # ==============================================================
        log(f"[pipe] Stage 7: unwrapping and saving to {output_dir}")

        # Replace each EscalationLinear with a plain nn.Linear holding the
        # committed weight. Walk the parent hierarchy.
        def _unwrap(parent: nn.Module, prefix: str = ""):
            for name, child in list(parent.named_children()):
                full = f"{prefix}.{name}" if prefix else name
                if isinstance(child, EscalationLinear):
                    linear = nn.Linear(
                        child.in_features, child.out_features,
                        bias=child.bias_buffer is not None,
                    )
                    linear = linear.to(child.fp4_weight.device, dtype=child.fp4_weight.dtype)
                    linear.weight.data = child.fp4_weight.clone()
                    if child.bias_buffer is not None:
                        linear.bias.data = child.bias_buffer.clone()
                    setattr(parent, name, linear)
                else:
                    _unwrap(child, full)

        _unwrap(model)

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        # Save manifest
        manifest = {
            "source_model": model_path,
            "pipeline": "autoround-first then dpq-escalate",
            "min_efficiency": min_efficiency,
            "autoround_iters": autoround_iters,
            "autoround_nsamples": autoround_nsamples,
            "autoround_seqlen": autoround_seqlen,
            "autoround_dataset": autoround_dataset,
            "rtn_fp4_baseline_kl": rtn_fp4_baseline_kl,
            "autoround_fp4_kl_vs_bf16": autoround_fp4_kl,
            "autoround_fp8_kl_vs_bf16": autoround_fp8_kl,
            "baseline_kl": baseline_kl,
            "final_kl": final_kl,
            "gap_closure": gap_closure,
            "counts": counts,
            "avg_cost_vs_fp4": avg_cost,
            "decisions": decisions,
            "elapsed_sec": time.time() - t_start,
        }
        with open(Path(output_dir) / "dpq_autoround_first_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log(f"[pipe] manifest saved")
        log(f"[pipe] DONE in {time.time() - t_start:.0f}s")

    finally:
        if staged_dir is not None:
            shutil.rmtree(staged_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None,
                        help="Directory to cache AutoRound outputs. If exists, "
                             "skip AutoRound stages and load from cache. If not, "
                             "run AutoRound and save. Enables dense Pareto sweeps.")
    parser.add_argument("--min-efficiency", type=float, default=0.25)
    parser.add_argument("--autoround-iters", type=int, default=100)
    parser.add_argument("--autoround-nsamples", type=int, default=32)
    parser.add_argument("--autoround-seqlen", type=int, default=256)
    parser.add_argument("--autoround-batch-size", type=int, default=4)
    parser.add_argument("--autoround-dataset", default="mbpp")
    parser.add_argument("--dpq-steps", type=int, default=100)
    parser.add_argument("--dpq-lr", type=float, default=0.05)
    parser.add_argument("--dpq-tau-start", type=float, default=1.5)
    parser.add_argument("--dpq-tau-end", type=float, default=0.1)
    parser.add_argument("--dpq-calib-samples", type=int, default=8)
    parser.add_argument("--dpq-calib-seqlen", type=int, default=128)
    parser.add_argument("--stages", choices=["all", "fp4", "fp8"], default="all",
                        help="Which AutoRound stages to run. 'fp4' runs only Stage 3a "
                             "(NVFP4) and saves its weights to the cache, then exits. "
                             "'fp8' runs only Stage 3b (FP8). Useful for splitting the "
                             "work across fresh Python processes to avoid memory leaks.")
    parser.add_argument("--cache-only", action="store_true",
                        help="Run AutoRound stages only (save to cache-dir), then exit. "
                             "Useful for populating the cache on memory-constrained setups "
                             "where the DPQ wrapper stage won't fit.")
    parser.add_argument("--no-hadamard", action="store_true",
                        help="disable AutoRound's random-Hadamard rotation preprocessing "
                             "(rotation is enabled by default — NVFP4-compatible via our "
                             "patched SUPPORTED_QUANTIZATION_SCHEMES whitelist)")
    args = parser.parse_args()

    run_pipeline(
        model_path=args.model,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        min_efficiency=args.min_efficiency,
        autoround_iters=args.autoround_iters,
        autoround_nsamples=args.autoround_nsamples,
        autoround_seqlen=args.autoround_seqlen,
        autoround_batch_size=args.autoround_batch_size,
        autoround_dataset=args.autoround_dataset,
        dpq_steps=args.dpq_steps,
        dpq_lr=args.dpq_lr,
        dpq_tau_start=args.dpq_tau_start,
        dpq_tau_end=args.dpq_tau_end,
        dpq_calib_samples=args.dpq_calib_samples,
        dpq_calib_seqlen=args.dpq_calib_seqlen,
        hadamard=not args.no_hadamard,
        cache_only=args.cache_only,
        stages=args.stages,
    )
