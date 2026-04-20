"""Streaming / incremental PrismQuant exporter.

Mirrors the `streaming_probe.py` pattern for the export stage: loads
the model skeleton via `init_empty_weights`, keeps head/embed/norm/
lm_head/rotary resident, and streams each decoder layer from disk →
quantize → emit → unload. Targets models that don't fit in RAM for a
conventional `from_pretrained(...)`-based export (e.g. Qwen3.5-122B-A10B
at 244 GB BF16 on a 121 GB Spark).

Output contract matches the non-streaming `export_native_compressed.
materialize_tensors`: returns `(out_tensors: dict[str, torch.Tensor],
hist: dict)` for consumption by `write_sharded_safetensors` and
`write_config_with_quantization`.

Memory ceiling (Qwen3.5-122B at 4.75 bpw):
- Head resident (embed + norm + lm_head + rotary): ~3 GB
- One layer resident on device at a time: ~5 GB
- Transient quant workspace per layer: ~10 GB
- `out` accumulator (final quantized artifact): ~78 GB
- Total peak: ~105 GB out of 121 GB available on Spark.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors import safe_open

from .export_native_compressed import (
    _compute_nvfp4_joint_global,
    _is_packed_experts_module,
    _materialize_mtp_tensors,
    _packed_experts_param_names,
    _quantize_2d,
    _quantize_3d_packed,
    compute_nvfp4_global_real,
    write_config_with_quantization,
    write_sharded_safetensors,
    _copy_tokenizer,
    _load_source_passthrough,
)
from .sensitivity_probe import stage_text_only
from .streaming_probe import (
    _build_install_resolver,
    _build_weight_map,
    _fast_install,
    _get_layer_list,
    _get_rotary,
    _head_prefixes,
    _materialize,
    _read_layer_to_device,
    _resolve_base_prefix,
    _unload,
)


def _init_rotary_inplace(base_model: nn.Module, device: torch.device,
                         dtype: torch.dtype) -> None:
    """After init_empty_weights, rotary modules exist but their
    `inv_freq` buffers are on meta. Re-run the module's own rope init
    (which is deterministic from config) so `inv_freq` lives on the
    exec device with correct values — matching what `from_pretrained`
    would have produced."""
    rotary = _get_rotary(base_model)
    if rotary is None:
        return
    cfg = getattr(rotary, "config", None)
    if cfg is None:
        return
    try:
        rope_init_fn = rotary.compute_default_rope_parameters
    except AttributeError:
        return
    inv_freq, attention_scaling = rope_init_fn(cfg, device)
    rotary.register_buffer("inv_freq", inv_freq.to(dtype=torch.float32,
                                                   device=device),
                           persistent=False)
    if hasattr(rotary, "original_inv_freq"):
        rotary.register_buffer(
            "original_inv_freq",
            inv_freq.to(dtype=torch.float32, device=device).clone(),
            persistent=False)
    rotary.attention_scaling = attention_scaling


# --------------------------------------------------------------------
# Per-layer joint NVFP4 global scale
# --------------------------------------------------------------------

_FUSED_SIBLINGS = {
    "q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv",
    "gate_proj": "gate_up", "up_proj": "gate_up",
    # Qwen3.5/3.6 DeltaNet linear-attention pairs. vLLM fuses
    # `in_proj_qkv + in_proj_z → in_proj_qkvz` and
    # `in_proj_b + in_proj_a → in_proj_ba` at load time; the fused
    # packed Linear needs ONE shared NVFP4 `weight_global_scale`.
    # Omitting these triggers vLLM's
    # `compressed_tensors_w4a4_nvfp4.py:97` warning about reduced
    # accuracy from mismatched parallel-layer scales.
    "in_proj_qkv": "qkvz", "in_proj_z": "qkvz",
    "in_proj_b": "ba", "in_proj_a": "ba",
}


def _compute_layer_joint_nvfp4(layer_mod: nn.Module,
                               layer_qname: str,
                               assignment: dict[str, str],
                               profile,
                               ) -> dict[str, torch.Tensor]:
    """Return {recipe_key -> joint global scale} for NVFP4 fused-sibling
    groups inside this decoder layer. Only keys assigned NVFP4 get an
    override entry; the rest compute per-Linear scales at quantize time.

    Semantically equivalent to a scoped `_compute_nvfp4_joint_global`
    across just this layer's modules."""
    # First collect fused sibling sets by (parent_qname, family).
    groups: dict[tuple[str, str], list[tuple[str, nn.Linear]]] = defaultdict(list)
    for sub_name, mod in layer_mod.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        last = sub_name.rsplit(".", 1)[-1]
        fam = _FUSED_SIBLINGS.get(last)
        if fam is None:
            continue
        parent = sub_name.rsplit(".", 1)[0] if "." in sub_name else ""
        groups[(parent, fam)].append((sub_name, mod))

    out: dict[str, torch.Tensor] = {}
    for (_, _), members in groups.items():
        # All members of a fused group must share a format. Check that
        # they all map to NVFP4 in `assignment`; otherwise skip (joint
        # override only applies to NVFP4).
        fqn_fmt = []
        for sub_name, mod in members:
            full = f"{layer_qname}.{sub_name}" if sub_name else layer_qname
            recipe_key = profile.live_to_recipe_name(full)
            fmt = assignment.get(recipe_key)
            fqn_fmt.append((full, recipe_key, fmt, mod))
        fmts = {f for _, _, f, _ in fqn_fmt}
        if fmts != {"NVFP4"}:
            continue
        # Joint global scale = max of per-member compute_nvfp4_global_real.
        candidates = []
        for _, _, _, mod in fqn_fmt:
            w = mod.weight.detach().float()
            candidates.append(compute_nvfp4_global_real(w, group_size=16))
        joint = torch.stack(candidates).max()
        for full, recipe_key, _, _ in fqn_fmt:
            out[recipe_key] = joint
    return out


# --------------------------------------------------------------------
# Core streaming materialization
# --------------------------------------------------------------------

def materialize_tensors_streaming(
    model_path: str,
    assignment: dict[str, str],
    *,
    profile,
    bf16_passthrough: set[str],
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    offload_folder: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Streaming counterpart to `materialize_tensors`. Never holds the
    full model in memory; processes one decoder layer at a time."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate.hooks import remove_hook_from_module

    # ----- 1. Meta skeleton + manual head materialization -----
    # Pure `init_empty_weights` path — avoids accelerate's
    # `from_pretrained` which would write ~244 GB of offload files to
    # disk on Qwen3.5-122B before we ever read them. Instead we:
    #   (a) build the full skeleton on meta (0 bytes),
    #   (b) read head/embed/norm/lm_head tensors directly from the
    #       source safetensors and install on the exec device,
    #   (c) re-run rotary's init_fn to populate `inv_freq` (not in
    #       state_dict — computed from config),
    #   (d) leave decoder layers on meta until the per-layer loop
    #       streams them in.
    staged = stage_text_only(model_path)
    config = AutoConfig.from_pretrained(staged, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_model, layers = _get_layer_list(model)
    base_prefix = _resolve_base_prefix(model, base_model)
    num_layers = len(layers)
    layers_prefix = f"{base_prefix}.layers." if base_prefix else "layers."

    weight_shard, weight_ckpt = _build_weight_map(model_path)

    # Materialize head (embed + norm + lm_head). These are in the
    # safetensors and get populated via `set_module_tensor_to_device`.
    print(f"[export-stream] base_prefix={base_prefix!r}  layers={num_layers}",
          flush=True)
    t0 = time.time()
    head_pfxs = _head_prefixes(None, base_prefix)
    # `_materialize` needs (model, prefixes, shard_map, ckpt_map, device, dtype)
    loaded_n = _materialize(model, head_pfxs, weight_shard, weight_ckpt,
                            device, dtype)

    # Rotary's `inv_freq` isn't in the state_dict — compute from config.
    _init_rotary_inplace(base_model, device, dtype)
    print(f"[export-stream] head materialized ({loaded_n} tensors, rotary "
          f"re-init) in {time.time()-t0:.1f}s", flush=True)

    out: dict[str, torch.Tensor] = {}
    hist: Counter = Counter()
    unmapped_keys: list[str] = []

    # ----- 2. Head / embed / norm / lm_head / rotary passthrough -----
    # These are resident on `device` already. Emit as BF16 passthrough
    # UNLESS `lm_head` (or similar) is explicitly in the assignment.
    t_head = time.time()
    # Top-level params (lm_head) and base-model params (embed_tokens, norm).
    def _emit_head_param(full_qname: str, param: nn.Parameter):
        # Check if the assignment covers this (rare — usually lm_head is
        # left as passthrough).
        recipe_key = profile.live_to_recipe_name(full_qname)
        fmt = assignment.get(recipe_key)
        if fmt is not None and fmt != "BF16":
            joint = None
            compressed = _quantize_2d(
                param.detach().float(), fmt,
                nvfp4_global_real_override=joint,
            )
            for suffix, t in compressed.items():
                key = f"{full_qname}.{suffix}" if suffix != "weight" else f"{full_qname}.weight"
                # Strip the .weight. prefix if baked into qname? No — our
                # inputs are param qnames (end in `.weight` already via the
                # named_parameters() walk). Make keys just `{qname}` for
                # suffix "weight" and `{base}.{suffix}` otherwise.
                base_name = full_qname[:-len(".weight")] if full_qname.endswith(".weight") else full_qname
                out_key = base_name if suffix == "weight" else f"{base_name}.{suffix}"
                out[out_key] = t.cpu()
            hist[("head", fmt)] += 1
        else:
            out[full_qname] = param.detach().to(torch.bfloat16).cpu()
            hist[("head_passthrough", "BF16")] += 1

    for name, p in model.named_parameters():
        if p.is_meta:
            continue  # only head/embed/norm/lm_head resident here
        _emit_head_param(name, p)

    for mod_name, mod in model.named_modules():
        non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
        for buf_name, buf in mod.named_buffers(recurse=False):
            if buf_name in non_persistent:
                continue
            if buf.is_meta:
                continue
            full = f"{mod_name}.{buf_name}" if mod_name else buf_name
            if full in out:
                continue
            out[full] = buf.detach().to(torch.bfloat16).cpu()
            hist[("head_buffer", "BF16")] += 1
    print(f"[export-stream] head+embed+norm+lm_head passthrough: "
          f"{time.time()-t_head:.1f}s  keys={len(out)}", flush=True)

    # ----- 3. Per-layer streaming quantize loop -----
    t_layers = time.time()
    for L in range(num_layers):
        layer_t0 = time.time()
        layer_qname = f"{layers_prefix}{L}".rstrip(".")
        if layer_qname.endswith("."):
            layer_qname = layer_qname[:-1]
        # `layers_prefix` ends in "." so `f"{layers_prefix}{L}"` is e.g.
        # "model.layers.3".

        # 3a. Load layer from safetensors (direct to device).
        load_t0 = time.time()
        tensors = _read_layer_to_device(
            f"{layers_prefix}{L}.", weight_shard, weight_ckpt, dtype, device)
        resolver = _build_install_resolver(model, layer_qname)
        _fast_install(resolver, tensors, device, model=model)
        load_s = time.time() - load_t0

        layer_mod = model.get_submodule(layer_qname)

        # 3b. Joint NVFP4 scales across fused siblings in this layer.
        joint_globals = _compute_layer_joint_nvfp4(
            layer_mod, layer_qname, assignment, profile)

        # 3c. Emit Linears.
        covered: set[str] = set()
        linear_count = 0
        for sub_name, mod in layer_mod.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            linear_count += 1
            full = f"{layer_qname}.{sub_name}"
            recipe_key = profile.live_to_recipe_name(full)
            fmt = assignment.get(recipe_key)
            if fmt is None:
                # No assignment → BF16 passthrough (matches non-streaming
                # materialize_tensors step 1).
                if not mod.weight.is_meta:
                    out[f"{full}.weight"] = mod.weight.detach().to(torch.bfloat16).cpu()
                    if mod.bias is not None and not mod.bias.is_meta:
                        out[f"{full}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
                    hist[("linear", "BF16")] += 1
                    covered.add(full)
                continue

            if fmt == "BF16" or recipe_key in bf16_passthrough:
                out[f"{full}.weight"] = mod.weight.detach().to(torch.bfloat16).cpu()
                if mod.bias is not None:
                    out[f"{full}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
                hist[("linear", "BF16")] += 1
                covered.add(full)
                continue

            override = joint_globals.get(recipe_key) if fmt == "NVFP4" else None
            compressed = _quantize_2d(
                mod.weight.detach().float(), fmt,
                nvfp4_global_real_override=override,
            )
            for suffix, t in compressed.items():
                out[f"{full}.{suffix}"] = t.cpu()
            if mod.bias is not None:
                out[f"{full}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
            hist[("linear", fmt)] += 1
            covered.add(full)

        # 3d. Emit packed MoE experts. Re-implement the same logic as
        # materialize_tensors step 2 but scoped to this layer.
        packed_count = 0
        for sub_name, mod in layer_mod.named_modules():
            if not _is_packed_experts_module(mod):
                continue
            packed_count += 1
            for pn in _packed_experts_param_names(mod):
                experts_qname = f"{layer_qname}.{sub_name}" if sub_name else layer_qname
                full = f"{experts_qname}.{pn}"
                recipe_key = profile.live_to_recipe_name(full)
                fmt = assignment.get(recipe_key)
                if fmt is None:
                    unmapped_keys.append(full)
                    continue
                packed_param = getattr(mod, pn).detach().float()
                E, M, N = packed_param.shape
                if pn == "gate_up_proj":
                    half = M // 2
                    proj_split = [
                        ("gate_proj", packed_param[:, :half, :]),
                        ("up_proj",   packed_param[:, half:, :]),
                    ]
                else:
                    proj_split = [(pn, packed_param)]

                is_bf16 = fmt == "BF16" or full in bf16_passthrough
                disk_qname = profile.on_disk_expert_qname(experts_qname)
                should_split = profile.split_packed_experts_for_format(fmt)

                if not should_split:
                    out[f"{disk_qname}.{pn}"] = packed_param.to(torch.bfloat16).cpu()
                    covered.add(full)
                    hist[("packed_moe", "BF16" if is_bf16 else fmt)] += 1
                    del packed_param
                    continue

                # Per-expert joint global scale when NVFP4 splits gate+up.
                per_expert_joint: list[torch.Tensor | None] = [None] * E
                if fmt == "NVFP4" and len(proj_split) > 1:
                    for e in range(E):
                        cands = [
                            compute_nvfp4_global_real(sp[e].float(),
                                                       group_size=16)
                            for _, sp in proj_split
                        ]
                        per_expert_joint[e] = torch.stack(cands).max()

                for proj_name, sub_packed in proj_split:
                    E_p, Mp, Np = sub_packed.shape
                    for e in range(E_p):
                        expert_2d = sub_packed[e]
                        base = f"{disk_qname}.{e}.{proj_name}"
                        if is_bf16:
                            out[f"{base}.weight"] = expert_2d.to(torch.bfloat16).cpu()
                        else:
                            compressed = _quantize_2d(
                                expert_2d, fmt,
                                nvfp4_global_real_override=per_expert_joint[e],
                            )
                            for suffix, t in compressed.items():
                                key = base if suffix == "weight" else f"{base}.{suffix}"
                                out[key] = t.cpu()
                covered.add(full)
                hist[("packed_moe_per_expert", "BF16" if is_bf16 else fmt)] += 1
                del packed_param, proj_split

        # 3e. Remaining layer-scoped params (norms, conv1d, biases on
        # passthrough-only modules). Also persistent buffers (e.g. Gemma 4
        # layer_scalar).
        for sub_name, param in layer_mod.named_parameters():
            full = f"{layer_qname}.{sub_name}"
            if full in out:
                continue
            if any(full.startswith(c + ".") or full == c for c in covered):
                continue
            if param.is_meta:
                continue
            out[full] = param.detach().to(torch.bfloat16).cpu()
            hist[("layer_passthrough", "BF16")] += 1
        for mod_name, mod in layer_mod.named_modules():
            non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
            for buf_name, buf in mod.named_buffers(recurse=False):
                if buf_name in non_persistent:
                    continue
                full_modpath = f"{layer_qname}.{mod_name}" if mod_name else layer_qname
                full = f"{full_modpath}.{buf_name}"
                if full in out or buf.is_meta:
                    continue
                out[full] = buf.detach().to(torch.bfloat16).cpu()
                hist[("layer_buffer", "BF16")] += 1

        # 3f. Unload.
        _unload(model, [f"{layers_prefix}{L}."])
        del tensors, resolver, joint_globals
        # Aggressive GPU cleanup — we've already `.cpu()`'d every
        # quantized output into `out`, so the per-layer GPU working
        # set (fp32 weight copies, grouped/packed intermediates) can
        # be released immediately. Keeps per-layer peak bounded.
        if device.type == "cuda":
            torch.cuda.synchronize()  # ensure outputs are CPU-resident
            torch.cuda.empty_cache()
        if L % 4 == 0:
            gc.collect()
        if L % 4 == 0 or L == num_layers - 1:
            elapsed = time.time() - layer_t0
            print(f"[export-stream] layer {L:02d}  linears={linear_count} "
                  f"packed={packed_count}  load={load_s:.2f}s  "
                  f"total={elapsed:.2f}s  out_keys={len(out)}", flush=True)

    print(f"[export-stream] layer sweep: {time.time()-t_layers:.1f}s", flush=True)

    if unmapped_keys:
        print(f"[export-stream] WARN {len(unmapped_keys)} unmapped assignment "
              f"keys — first 5: {unmapped_keys[:5]}", flush=True)

    return out, dict(hist)


# --------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------

def _canonicalize_assignment(raw: dict) -> dict[str, str]:
    """Accept either AutoRound-style dicts (`{key: {bits: 4, data_type: nv_fp,
    ...}}`) or shorthand (`{key: "NVFP4"}`). Return `{key: fmt_str}` with
    fmt in {"NVFP4", "MXFP8", "BF16"}."""
    from .export_native_compressed import canonicalize_format
    out: dict[str, str] = {}
    for k, v in raw.items():
        out[k] = canonicalize_format(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="HF model dir (source safetensors + config.json)")
    ap.add_argument("--layer-config", required=True,
                    help="layer_config.json from allocator.py")
    ap.add_argument("--output", required=True,
                    help="Output directory for the compressed checkpoint")
    ap.add_argument("--shard-bytes", type=int, default=5 * 1024**3,
                    help="Approx per-shard size in bytes (default 5 GiB)")
    ap.add_argument("--device", default="cuda",
                    help="Device for quantization arithmetic. Layer "
                         "weights are read into this device; "
                         "_quantize_2d / _quantize_3d_packed run here; "
                         "outputs are moved to CPU before storage.")
    ap.add_argument("--offload-folder", default=None,
                    help="Accelerate disk-offload folder (defaults to "
                         "sibling of output).")
    ap.add_argument("--ignore", nargs="*", default=["lm_head"],
                    help="Module qnames to keep at bf16 even if the "
                         "allocator assigned another format.")
    args = ap.parse_args()

    from .model_profiles import detect_profile
    profile = detect_profile(args.model)
    print(f"[export-stream] model profile: {profile.name}", flush=True)

    with open(args.layer_config) as f:
        raw_recipe = json.load(f)
    assignment = _canonicalize_assignment(raw_recipe)
    fmts = Counter(assignment.values())
    print(f"[export-stream] recipe: {len(assignment)} entries  mix={dict(fmts)}",
          flush=True)

    dtype = torch.bfloat16
    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_passthrough = set(args.ignore)
    if args.offload_folder is None:
        args.offload_folder = str(out_dir / "_streaming_offload")

    tensors, hist = materialize_tensors_streaming(
        args.model, assignment,
        profile=profile, bf16_passthrough=bf16_passthrough,
        dtype=dtype, device=device,
        offload_folder=args.offload_folder,
    )
    print(f"[export-stream] materialized {len(tensors)} tensors  hist={hist}",
          flush=True)

    # Mirror the non-streaming `main()` post-materialize rename:
    # multimodal-umbrella archs (Qwen3.5/3.6 ConditionalGeneration,
    # Gemma 4 ConditionalGeneration) expect body keys under
    # `model.language_model.` on disk, even though our streaming loop
    # produces the text-only `model.layers.X.*` form.
    body_infix = getattr(profile, "body_ondisk_infix", None)
    if callable(body_infix):
        infix = body_infix()
    else:
        # Default: Qwen3.5/3.6 pattern. Profiles for non-multimodal
        # archs can return "" and we'll skip the rename.
        infix = "language_model." if profile.name.startswith("qwen3_5") else ""
    if infix:
        renamed: dict[str, torch.Tensor] = {}
        for k, v in tensors.items():
            if (k.startswith("model.layers.")
                    or k.startswith("model.embed_tokens")
                    or k.startswith("model.norm")):
                renamed[f"model.{infix}{k[len('model.'):]}"] = v
            else:
                renamed[k] = v
        tensors = renamed
        print(f"[export-stream] renamed body → model.{infix}...",
              flush=True)

    # MTP materialization if the profile has heads. Uses the existing
    # non-streaming helper — MTP heads are small enough that full-model
    # residency isn't a concern.
    mtp_tensors: dict[str, torch.Tensor] = {}
    if profile.has_mtp():
        print("[export-stream] materializing MTP tensors ...", flush=True)
        mtp_tensors = _materialize_mtp_tensors(
            args.model, assignment,
            bf16_passthrough=bf16_passthrough, hist=hist)
        print(f"[export-stream] MTP: {len(mtp_tensors)} tensors", flush=True)
    else:
        print(f"[export-stream] profile '{profile.name}' has no MTP — "
              "skipping", flush=True)

    # Merge source passthrough (visual/audio towers etc.) that aren't
    # part of our streaming pass. Drop entries that our MTP materialize
    # already covered — same logic the non-streaming export uses.
    passthrough_prefixes = tuple(profile.source_passthrough_prefixes())
    if passthrough_prefixes:
        src_extra = _load_source_passthrough(
            args.model, prefix_filters=passthrough_prefixes)
        # Collect base-names already materialized in mtp_tensors so we
        # don't double-write them as passthrough.
        materialized_bases: set[str] = set()
        import re as _re
        for k in mtp_tensors:
            base = k
            for suf in (".weight_packed", ".weight_scale",
                        ".weight_global_scale", ".input_global_scale",
                        ".weight"):
                if k.endswith(suf):
                    base = k[:-len(suf)] + ".weight"
                    break
            materialized_bases.add(base)
            m = _re.match(r"^(mtp\.layers\.\d+\.mlp\.experts)\.\d+\.(gate|up|down)_proj\.", k)
            if m:
                if m.group(2) in ("gate", "up"):
                    materialized_bases.add(f"{m.group(1)}.gate_up_proj")
                else:
                    materialized_bases.add(f"{m.group(1)}.down_proj")
        src_extra = {k: v for k, v in src_extra.items()
                     if k not in materialized_bases}
        for k in list(src_extra.keys()):
            if k in tensors or k in mtp_tensors:
                del src_extra[k]
        tensors.update(mtp_tensors)
        tensors.update(src_extra)
        print(f"[export-stream] merged {len(src_extra)} source-passthrough + "
              f"{len(mtp_tensors)} MTP tensors", flush=True)
    else:
        tensors.update(mtp_tensors)

    print("[export-stream] writing safetensors shards ...", flush=True)
    t_write = time.time()
    write_sharded_safetensors(tensors, out_dir, args.shard_bytes)
    print(f"[export-stream] sharded write: {time.time()-t_write:.1f}s",
          flush=True)

    # Scan source safetensors for 2D `.weight` keys not covered by the
    # recipe — these are visual encoder / unmapped Linears that vLLM
    # instantiates during model-construction time. Without an explicit
    # ignore entry, compressed-tensors' `find_matched_target` raises
    # `ValueError: Unable to find matching target for visual.merger.*`.
    # Mirrors the non-streaming export's logic at export_native_compressed.
    # py:1173.
    extra_ignore: list[str] = []
    seen_recipe = {n for n in assignment}
    src_dir = Path(args.model)
    if src_dir.exists():
        from safetensors import safe_open as _safe_open
        import os as _os
        for f in sorted(_os.listdir(src_dir)):
            if not f.endswith(".safetensors"):
                continue
            with _safe_open(str(src_dir / f), framework="pt") as sf:
                for k in sf.keys():
                    if not k.endswith(".weight"):
                        continue
                    base = k[:-7]
                    recipe_name = ("model." + base[len("model.language_model."):]
                                   if base.startswith("model.language_model.")
                                   else base)
                    if recipe_name in seen_recipe:
                        continue
                    try:
                        shape = list(sf.get_slice(k).get_shape())
                    except Exception:
                        shape = []
                    if len(shape) != 2:
                        continue
                    extra_ignore.append(base)
    print(f"[export-stream] extra ignore (unmapped Linears): "
          f"{len(extra_ignore)}", flush=True)

    write_config_with_quantization(
        args.model, out_dir, assignment, bf16_passthrough,
        extra_ignore=extra_ignore)
    _copy_tokenizer(args.model, out_dir)
    print(f"[export-stream] done. Serve with:\n"
          f"  vllm serve {out_dir} --quantization compressed-tensors",
          flush=True)


if __name__ == "__main__":
    main()
