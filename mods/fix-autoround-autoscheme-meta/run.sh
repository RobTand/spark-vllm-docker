#!/bin/bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path


def patch_delta_loss() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/auto_round/auto_scheme/delta_loss.py")
    text = path.read_text()

    old = """    else:\n        model = model.to(\"cpu\")  # TODO this requires large ram\n        if hasattr(model, \"hf_device_map\") and len(model.hf_device_map) > 1:\n            import accelerate\n\n            accelerate.hooks.remove_hook_from_submodules(model)\n            delattr(model, \"hf_device_map\")\n        for n, m in model.named_modules():\n            if hasattr(m, \"scale_dtype\"):  # TODO refine code\n                delattr(m, \"scale_dtype\")\n            if hasattr(m, \"imatrix\"):\n                delattr(m, \"imatrix\")\n            if hasattr(m, \"tuning_device\"):\n                delattr(m, \"tuning_device\")\n        for n, m in model.named_parameters():\n            if hasattr(m, \"grad\"):\n                m.grad = None\n"""
    new = """    else:\n        try:\n            model = model.to(\"cpu\")  # TODO this requires large ram\n        except NotImplementedError:\n            # Some large-model load paths leave meta-backed tensors in the live\n            # module tree. Autoscheme is finished here, so skipping the final\n            # CPU materialization is safe and avoids crashing during cleanup.\n            model = None\n        if model is not None:\n            if hasattr(model, \"hf_device_map\") and len(model.hf_device_map) > 1:\n                import accelerate\n\n                accelerate.hooks.remove_hook_from_submodules(model)\n                delattr(model, \"hf_device_map\")\n            for n, m in model.named_modules():\n                if hasattr(m, \"scale_dtype\"):  # TODO refine code\n                    delattr(m, \"scale_dtype\")\n                if hasattr(m, \"imatrix\"):\n                    delattr(m, \"imatrix\")\n                if hasattr(m, \"tuning_device\"):\n                    delattr(m, \"tuning_device\")\n            for n, m in model.named_parameters():\n                if hasattr(m, \"grad\"):\n                    m.grad = None\n"""
    if old not in text:
        raise SystemExit("expected autoscheme cleanup block not found")
    text = text.replace(old, new, 1)

    old_oom = """    except torch.OutOfMemoryError:\n        logger.warning(\n            \"Fallback to CPU for automatic scheme generation.\"\n            \" Using multiple devices is strongly recommended (e.g., --device_map 0,1,2,3).\"\n        )\n        model.to(\"cpu\")\n        for n, m in model.named_modules():\n            if hasattr(m, \"orig_layer\"):\n                set_module(model, n, m.orig_layer)\n        clear_memory(device_list=device_list)\n        if hasattr(model, \"hf_device_map\") and len(model.hf_device_map) > 1:\n            import accelerate\n\n            accelerate.hooks.remove_hook_from_submodules(model)\n            delattr(model, \"hf_device_map\")\n        res = _gen_layer_config(\n"""
    new_oom = """    except torch.OutOfMemoryError as exc:\n        logger.warning(\n            \"Fallback to CPU for automatic scheme generation.\"\n            \" Using multiple devices is strongly recommended (e.g., --device_map 0,1,2,3).\"\n        )\n        logger.warning(f\"Autoscheme GPU pass raised OutOfMemoryError: {exc!r}\")\n        try:\n            model.to(\"cpu\")\n        except NotImplementedError:\n            model_path = getattr(model, \"name_or_path\", None)\n            if not model_path:\n                raise\n            model, tokenizer, _ = llm_load_model(model_path, device_map=\"cpu\")\n        for n, m in model.named_modules():\n            if hasattr(m, \"orig_layer\"):\n                set_module(model, n, m.orig_layer)\n        clear_memory(device_list=device_list)\n        if hasattr(model, \"hf_device_map\") and len(model.hf_device_map) > 1:\n            import accelerate\n\n            accelerate.hooks.remove_hook_from_submodules(model)\n            delattr(model, \"hf_device_map\")\n        res = _gen_layer_config(\n"""
    if old_oom not in text:
        raise SystemExit("expected autoscheme OOM fallback block not found")
    text = text.replace(old_oom, new_oom, 1)
    path.write_text(text)
    print(f"Patched {path}")


def patch_main() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/auto_round/__main__.py")
    text = path.read_text()
    old = """        scheme = AutoScheme(\n            options=args.options,\n            avg_bits=args.avg_bits,\n            shared_layers=args.shared_layers,\n            ignore_scale_zp_bits=args.ignore_scale_zp_bits,\n            low_gpu_mem_usage=True,  # force it to be True as it uses much smaller vram but similar time cost\n            low_cpu_mem_usage=low_cpu_mem_usage,\n        )\n"""
    new = """        scheme = AutoScheme(\n            options=args.options,\n            avg_bits=args.avg_bits,\n            shared_layers=args.shared_layers,\n            ignore_scale_zp_bits=args.ignore_scale_zp_bits,\n            low_gpu_mem_usage=args.low_gpu_mem_usage,\n            low_cpu_mem_usage=low_cpu_mem_usage,\n        )\n"""
    if old not in text:
        raise SystemExit("autoscheme constructor block not found")
    path.write_text(text.replace(old, new, 1))
    print(f"Patched {path}")


def patch_model_load() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/auto_round/utils/model.py")
    text = path.read_text()
    old = """    load_kwargs = {\n        \"torch_dtype\": torch_dtype,\n        \"trust_remote_code\": trust_remote_code,\n        \"device_map\": \"auto\" if use_auto_mapping else None,\n    }\n"""
    new = """    load_kwargs = {\n        \"torch_dtype\": torch_dtype,\n        \"trust_remote_code\": trust_remote_code,\n        \"device_map\": \"auto\" if use_auto_mapping else None,\n        \"low_cpu_mem_usage\": False,\n    }\n"""
    if old not in text:
        raise SystemExit("load_kwargs block not found")
    path.write_text(text.replace(old, new, 1))
    print(f"Patched {path}")


def patch_dispatch_utils() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/auto_round/auto_scheme/utils.py")
    text = path.read_text()
    old = """    devices = parse_available_devices(device_map)\n\n    if len(devices) == 1:\n        model.to(devices[0])\n        return model\n"""
    new = """    devices = parse_available_devices(device_map)\n\n    has_meta = any(getattr(p, \"is_meta\", False) for p in model.parameters())\n    if has_meta:\n        from auto_round.modeling.fused_moe.replace_modules import materialize_model_\n\n        materialize_model_(model)\n\n    if len(devices) == 1:\n        model.to(devices[0])\n        return model\n"""
    if old not in text:
        raise SystemExit("dispatch block not found")
    path.write_text(text.replace(old, new, 1))
    print(f"Patched {path}")


patch_delta_loss()
patch_main()
patch_model_load()
patch_dispatch_utils()
PY

echo "Applied AutoRound autoscheme GPU/meta fixes"
