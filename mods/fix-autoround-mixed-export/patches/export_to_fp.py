# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Union

import threadpoolctl as tctl
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from auto_round.compressors.utils import is_mx_fp, is_nv_fp
from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear
from auto_round.export.export_to_llmcompressor.utils import generate_ignore_regex_list
from auto_round.export.utils import filter_quantization_config, release_layer_safely, save_model
from auto_round.logger import logger
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_start_with_block_name,
    check_to_quantized,
    copy_python_files_from_model_cache,
    get_block_names,
    get_module,
    set_amax_for_all_moe_layers,
    set_module,
    unsupported_meta_device,
)
from auto_round.wrapper import WrapperWALayer

from .config import check_compressed_tensors_supported

__all__ = [
    "pack_layer",
    "save_quantized_as_fp",
]


def pack_layer(name, model, device=None):
    layer = get_module(model, name)
    if type(layer) not in SUPPORTED_LAYER_TYPES and not isinstance(layer, WrapperWALayer):  ##already packed
        return

    if isinstance(layer, WrapperWALayer):  # revert WrapperWALayer for offline usage
        wp_layer = layer
        layer = wp_layer.orig_layer
        set_module(model, name, layer)

    orig_device = layer.weight.device
    data_type = layer.data_type
    act_bits = layer.act_bits
    act_data_type = layer.act_data_type
    bits = layer.bits
    if bits > 8:
        return
    group_size = layer.group_size
    sym = layer.sym

    if is_nv_fp(act_data_type) and act_bits <= 8:
        input_global_scale = getattr(layer, "input_global_scale", None)
        if input_global_scale is None:
            assert hasattr(layer, "act_max")
            from auto_round.data_type.nvfp import calculate_gparam

            input_global_scale = calculate_gparam(layer.act_max, layer.group_size)  # , model.device
            setattr(layer, "input_global_scale", input_global_scale)
            delattr(layer, "act_max")

    # QuantLinear = get_fp_qlinear(backend, bits, group_size, sym)

    if type(layer) == nn.Linear:
        in_features = layer.in_features
        out_features = layer.out_features
    elif type(layer) == nn.Conv2d:
        in_features = layer.in_channels
        out_features = layer.out_channels
    elif type(layer) == transformers.pytorch_utils.Conv1D:
        in_features = layer.weight.shape[0]
        out_features = layer.weight.shape[1]

    bias = layer.bias is not None
    ##bias = True  ## if using the above, llama3 lambada RTN will be NAN , TODO why?
    qlayer = QuantLinear(  ##pylint: disable=E1123
        bits,
        group_size,
        in_features,
        out_features,
        bias,
        weight_dtype=layer.weight.dtype,
        sym=sym,
        data_type=data_type,
        act_bits=act_bits,
        act_data_type=act_data_type,
    )

    qlayer.device = orig_device
    scale = layer.scale
    global_scale = getattr(layer, "weight_global_scale", None)
    input_global_scale = getattr(layer, "input_global_scale", None)
    # zero = layer.zp # no zeros to handle, as mxfp/nvfp do not support asym quantization
    qlayer.pack(layer, scale, global_scale=global_scale, input_global_scale=input_global_scale, device=device)
    qlayer.to(orig_device)
    set_module(model, name, qlayer)
    # Note: release weight and bias explicitly, in case they are referenced elsewhere
    release_layer_safely(layer)



# AUTO_ROUND_MIXED_EXPORT_PATCH_V1 -----------------------------------------

_FUSED_GROUPS = {
    "self_attn": {
        frozenset({"q_proj", "k_proj", "v_proj"}): ("q_proj", "k_proj", "v_proj"),
    },
    "mlp": {
        frozenset({"gate_proj", "up_proj"}): ("gate_proj", "up_proj"),
    },
}

_FORMAT_PRIORITY = {"mxfp8-quantized": 2, "nvfp4-pack-quantized": 1, "float-quantized": 0}


def _per_layer_format(cfg: dict) -> str | None:
    """Map a layer_config entry to its target compressed-tensors format."""
    dt = cfg.get("data_type")
    bits = cfg.get("bits")
    act_bits = cfg.get("act_bits")
    if bits is None or bits >= 16:
        return None
    if is_mx_fp(dt) and bits == 8:
        return "mxfp8-quantized"
    if is_nv_fp(dt) and bits == 4:
        return "nvfp4-pack-quantized"
    if is_mx_fp(dt) and bits == 4:
        return "mxfp4-pack-quantized"
    if bits == 8 and act_bits is not None and act_bits <= 8:
        return "float-quantized"
    return None


def _scheme_dict_for(cfg: dict, format_name: str) -> dict:
    """Build a compressed-tensors scheme dict from a layer_config entry."""
    bits = cfg["bits"]
    group_size = cfg.get("group_size", 16)
    act_bits = cfg.get("act_bits", bits)
    act_group_size = cfg.get("act_group_size", group_size)
    dt = cfg.get("data_type", "")

    # Determine scale dtype / strategy per format family
    if format_name == "nvfp4-pack-quantized":
        strategy = "tensor_group"
        scale_dtype = "torch.float8_e4m3fn"
    elif format_name in ("mxfp8-quantized", "mxfp4-pack-quantized"):
        strategy = "group"
        scale_dtype = "torch.uint8"
        group_size = 32
        act_group_size = 32
    else:
        strategy = "group"
        scale_dtype = "torch.float16"

    weights = {
        "num_bits": bits,
        "type": "float",
        "symmetric": True,
        "group_size": group_size,
        "strategy": strategy,
        "block_structure": None,
        "dynamic": False,
        "actorder": None,
        "scale_dtype": scale_dtype,
        "zp_dtype": None,
        "observer": "memoryless_minmax",
        "observer_kwargs": {},
    }

    if act_bits is not None and act_bits <= 8:
        act_dynamic = "local" if format_name == "nvfp4-pack-quantized" else True
        input_activations = {
            "num_bits": act_bits,
            "type": "float",
            "symmetric": True,
            "group_size": act_group_size,
            "strategy": strategy,
            "block_structure": None,
            "dynamic": act_dynamic,
            "actorder": None,
            "scale_dtype": scale_dtype,
            "zp_dtype": None,
            "observer": "static_minmax" if format_name == "nvfp4-pack-quantized" else None,
            "observer_kwargs": {},
        }
    else:
        input_activations = None

    return {
        "targets": [],
        "weights": weights,
        "input_activations": input_activations,
        "output_activations": None,
        "format": format_name,
    }


def _promote_fused_groups(layer_to_fmt: dict[str, str]) -> dict[str, str]:
    """Ensure fused-projection siblings share the same format.

    vLLM fuses q_proj/k_proj/v_proj into qkv_proj and gate_proj/up_proj into
    gate_up_proj during weight loading. If their formats disagree the loader
    sees a parameter-layout mismatch inside a single fused parameter. We
    promote all siblings to the highest-precision format picked for any of
    them (never downgrade).
    """
    promoted = dict(layer_to_fmt)
    # Group by (parent_prefix, container)
    by_parent: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for name in promoted:
        for container, groups in _FUSED_GROUPS.items():
            token = f".{container}."
            if token in name:
                prefix, suffix = name.rsplit(token, 1)
                for members in groups.values():
                    if suffix in members:
                        by_parent.setdefault((prefix, container, members), []).append(
                            (name, suffix)
                        )
                        break
    for (prefix, container, members), entries in by_parent.items():
        # Find best (highest-priority) format assigned among known siblings
        best_fmt = max(
            (promoted[n] for n, _ in entries),
            key=lambda f: _FORMAT_PRIORITY.get(f, -1),
        )
        # Any siblings missing from layer_to_fmt will not be forced; we only
        # promote siblings that are already in the pool (the unlisted ones
        # default to the top-level NVFP4 pick, which matches the "lowest" tier).
        for n, _ in entries:
            promoted[n] = best_fmt
    return promoted


def _build_mixed_config_groups(
    layer_config: dict,
    default_format: str,
    ignore: list[str],
) -> tuple[dict, dict, str]:
    """Return (config_groups, extra_config, top_level_format).

    Groups layers by scheme signature and emits one config_group per unique
    scheme with `targets` containing exact-match regexes. A catch-all
    group_default with targets=['Linear'] captures any remaining layers in
    the default format.
    """
    # 1. Compute per-layer format
    layer_to_fmt: dict[str, str] = {}
    layer_to_cfg: dict[str, dict] = {}
    ignored = set(ignore or [])
    for name, cfg in layer_config.items():
        if name in ignored:
            continue
        if cfg.get("bits", 16) >= 16:
            continue
        fmt = _per_layer_format(cfg)
        if fmt is None:
            continue
        layer_to_fmt[name] = fmt
        layer_to_cfg[name] = cfg

    # 2. Promote fused-group siblings so qkv_proj / gate_up_proj are homogeneous
    layer_to_fmt = _promote_fused_groups(layer_to_fmt)

    # 3. Bucket by format. Within a format family (nvfp4-pack-quantized,
    # mxfp8-quantized, etc.) the scheme parameters are canonical, so all
    # layers sharing a format share a config_group. _scheme_dict_for already
    # enforces the canonical params.
    buckets: dict[tuple, list[str]] = {}
    for name, fmt in layer_to_fmt.items():
        buckets.setdefault((fmt,), []).append(name)

    # V2_EXPLICIT_TARGETS
    # 4. Emit config_groups with EXPLICIT per-layer regex targets everywhere.
    # Using `targets=["Linear"]` as a catch-all for the default group fails
    # when the loader tries to resolve fused projections (qkv_proj,
    # gate_up_proj) -- vLLM's weight loader maps separate safetensors keys
    # through stacked_params_mapping, which requires each original
    # projection name to match an explicit target. Enumerating every
    # non-promoted quantized layer mirrors what llm_compressor's
    # save_pretrained emits for mixed-precision checkpoints, which is the
    # layout vLLM actually supports.
    config_groups: dict[str, dict] = {}
    used_formats: set[str] = set()
    idx = 0
    # Collect all quantized layer names to emit as explicit regex targets
    all_quantized = sorted(layer_to_fmt.keys())
    default_sig = None
    for sig in buckets:
        if sig[0] == default_format:
            default_sig = sig
            break
    # If no bucket matches the default format (unusual), no default group
    # is emitted -- every layer is already covered by an explicit group.

    # Emit non-default groups first so their explicit targets take priority.
    for sig, names in buckets.items():
        if sig == default_sig:
            continue
        fmt = sig[0]
        scheme = _scheme_dict_for(layer_to_cfg[names[0]], fmt)
        scheme["targets"] = sorted(
            f"re:^{n.replace('.', '[.]')}$" for n in names
        )
        config_groups[f"group_{idx}"] = scheme
        used_formats.add(fmt)
        idx += 1
    # The default group gets explicit regex targets for every layer assigned
    # to it -- NO class-name catch-all. This matches llm_compressor's output.
    default_names = [n for n in all_quantized if layer_to_fmt[n] == default_format]
    if default_sig is not None and default_names:
        default_scheme = _scheme_dict_for(
            layer_to_cfg[buckets[default_sig][0]], default_format
        )
        default_scheme["targets"] = sorted(
            f"re:^{n.replace('.', '[.]')}$" for n in default_names
        )
        config_groups[f"group_{idx}"] = default_scheme
        used_formats.add(default_format)

    top_level_format = (
        "mixed-precision" if len(used_formats) > 1 else next(iter(used_formats), default_format)
    )
    extra_config = {
        name: layer_to_cfg[name]
        for name in layer_to_fmt
        if layer_to_fmt[name] != default_format
    }
    return config_groups, extra_config, top_level_format


# ---------------------------------------------------------------------------

def save_quantized_as_fp(
    output_dir: str,
    model: torch.nn.Module = None,
    tokenizer: Callable = None,
    layer_config: dict = None,
    inplace: bool = True,
    device: Union[str, torch.device] = "cpu",
    backend: str = None,
    serialization_dict: dict = None,
    **kwargs,
) -> torch.nn.Module:
    """
    Saves a quantized model of mxfp/nvfp data_type in the llm-compressor format.

    Args:
        output_dir (str): The directory where the quantized model will be saved.
        inplace (bool, optional): If True, modifies the model in place. Otherwise, creates a deepcopy of the model.
                                Default is True.
        backend (str, optional): The backend to be used for quantization.
                                  Default is "autoround:exllamav2".
        **kwargs: Additional keyword arguments including:
            - model (nn.Module): The model to be quantized.
            - layer_config (dict): The layer configuration for each layer.
            - serialization_dict (dict): The serialization configuration.
            - tokenizer (Tokenizer, optional): The tokenizer to be saved.

    Returns:
        None

    Raises:
        ValueError: If the backend is not supported.
    """
    bits = serialization_dict.get("bits", None)
    data_type = serialization_dict.get("data_type", None)
    act_bits = serialization_dict.get("act_bits", None)
    act_data_type = serialization_dict.get("act_data_type", None)
    safe_serialization = True if "safe_serialization" not in kwargs.keys() else kwargs["safe_serialization"]
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))
    processor = kwargs.get("processor", None)
    regex_config = serialization_dict.pop("regex_config")
    extra_config = {}

    if act_bits <= 8:
        # revert WrapperWALayer for offline usage
        for n, m in model.named_modules():
            if isinstance(m, WrapperWALayer):
                orig_layer = m.orig_layer
                set_module(model, n, orig_layer)

    if is_nv_fp(act_data_type) and "static_gs" in str(act_data_type).lower():
        # generate static input_global_scale
        for n, m in model.named_modules():
            if type(m) in SUPPORTED_LAYER_TYPES:
                layer = m
                if hasattr(layer, "act_bits") and layer.act_bits < 8 and not getattr(layer, "input_global_scale", None):
                    assert hasattr(layer, "act_max")
                    from auto_round.data_type.nvfp import calculate_gparam

                    input_global_scale = calculate_gparam(layer.act_max, layer.group_size, model.device)
                    setattr(layer, "input_global_scale", input_global_scale)
                    delattr(layer, "act_max")
        # update fused input_global_scale
        from auto_round.data_type.utils import update_fused_layer_global_scales

        modules = list(model.modules())
        for module in tqdm(modules, desc="Update input global scale for fuse modules"):
            update_fused_layer_global_scales(module, base_name="input")

    names = list(layer_config.keys())
    max_workers = 1
    if not torch.cuda.is_available() or not torch.xpu.is_available():
        max_workers = 2  ## 2 with cuda packing will cause hang occasionally
    if not unsupported_meta_device(model):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(names), leave=True) as pbar:

                def wrapper(name):
                    pbar.set_description(f"packing {name}")
                    with tctl.threadpool_limits(limits=1):
                        pack_layer(name, model, device)
                    pbar.update(1)

                for _ in executor.map(wrapper, names):
                    pass

    ignore = generate_ignore_regex_list(regex_config=regex_config, layer_config=layer_config)
    for _n, _c in layer_config.items():
        if isinstance(_c, dict):
            _types[(_c.get("data_type"), _c.get("bits"))] += 1

    # AUTO_ROUND_MIXED_EXPORT_PATCH_V1: emit real mixed-precision groups
    check_compressed_tensors_supported()

    # Determine the "default" (catch-all) format from the global
    # serialization_dict. layer_config entries whose data_type matches
    # stay in the catch-all group; entries that diverge get their own
    # config_group. This lets AutoScheme picks propagate into the
    # compressed-tensors config without a separate re-quantization pass.
    if is_mx_fp(data_type) and bits == 8:
        default_fmt = "mxfp8-quantized"
    elif is_mx_fp(data_type) and bits == 4:
        default_fmt = "mxfp4-pack-quantized"
    elif is_nv_fp(data_type):
        default_fmt = "nvfp4-pack-quantized"
    else:
        default_fmt = "float-quantized"

    config_groups, extra_cfg, top_fmt = _build_mixed_config_groups(
        layer_config, default_fmt, ignore
    )

    quantization_config = {
        "config_groups": config_groups,
        "format": top_fmt,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "ignore": ignore,
        "kv_cache_scheme": None,
        "global_compression_ratio": None,
        "provider": "auto-round",
    }
    # Preserve AutoScheme's per-layer picks for reference (not used by
    # vLLM but helpful for debugging and re-derivation).
    if extra_cfg:
        quantization_config["extra_config"] = extra_cfg
    def _deep_safe_for_json(obj):
        import torch as _t
        if isinstance(obj, dict):
            return {k: _deep_safe_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_deep_safe_for_json(x) for x in obj)
        if isinstance(obj, _t.dtype):
            return str(obj)
        return obj

    quantization_config = _deep_safe_for_json(quantization_config)

    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config
    if output_dir is None:
        return model

    if output_dir is None:
        model.tokenizer = tokenizer
        return model
    if os.path.exists(output_dir):
        logger.warning(f"{output_dir} already exists, this may cause model conflict")
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    if processor is not None:
        processor.save_pretrained(output_dir)

    dtype = None
    save_model(model, output_dir, safe_serialization=safe_serialization, dtype=dtype)

    return model
