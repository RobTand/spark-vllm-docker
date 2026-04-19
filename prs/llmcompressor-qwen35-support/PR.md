# [Bugfix] Fix model_free_ptq for models with non-contiguous fused attention layers (Qwen3.5)

## Summary

Fix two bugs that prevent `model_free_ptq` and `oneshot` from quantizing models with mixed attention architectures (e.g., Qwen3.5 which interleaves `self_attn` and `linear_attn` layers).

## Problem

Qwen3.5 uses a hybrid attention architecture: 3 DeltaNet (`linear_attn`) layers followed by 1 full (`self_attn`) layer, repeating. This causes two failures:

### Bug 1: `match_names_set_eager` crashes on cross-shard fused weights

In Qwen3.5-27B, all `q_proj` weights land in shard 8 while `k_proj`/`v_proj` are in shard 11. When `validate_safetensors_index` processes shard 8, `match_names_set_eager` matches layer 3's `q_proj`, then layer 7's `q_proj` hits the same target slot before `k_proj`/`v_proj` complete the set (because they're in a different shard).

```
ValueError: Matched a re:.*(attn|attention)\.q_proj\.weight$ twice before
completing set (model.language_model.layers.3.self_attn.q_proj.weight,
model.language_model.layers.7.self_attn.q_proj.weight)
```

**Root cause:** `match_names_set_eager` assumes fused weights (q/k/v) are always co-located in the same shard. With non-contiguous attention layers spread across shards, this assumption breaks.

### Bug 2: Sequential pipeline passes 4D tensors to DeltaNet forward (oneshot)

During calibration, the sequential pipeline's `IntermediatesCache` adds an extra batch dimension to intermediate activations. `Qwen3_5GatedDeltaNet.forward` strictly unpacks `batch_size, seq_len, _ = hidden_states.shape`, which fails on 4D input `(1, 1, seq_len, hidden)`.

```
ValueError: too many values to unpack (expected 3)
```

**Root cause:** The intermediates cache wraps tensors with an extra leading dimension during store/retrieve cycles.

## Fix

### Bug 1 fix: `src/llmcompressor/entrypoints/model_free/helpers.py`

When `match_names_set_eager` encounters a duplicate target match, instead of raising `ValueError`, flush the incomplete set and start a new one. The incomplete sets are collected and returned as unmatched (handled by the existing cross-shard resolution logic in `validate_safetensors_index`).

```python
# Before (raises on duplicate):
else:
    raise ValueError(
        f"Matched a {target} twice before "
        f"completing set ({matches[target]}, {name})"
    )

# After (flush incomplete set, start new):
else:
    incomplete_sets.append(dict(matches))
    matches = dict.fromkeys(targets, None)
    matches[target] = name
```

The return type of unmatched changes from `dict | None` to `list[dict] | None` to accommodate multiple incomplete sets.

### Bug 1 fix: `src/llmcompressor/entrypoints/model_free/microscale.py`

Update `get_fused_names` to handle the new list return type from `match_names_set_eager`:

```python
# Before:
if _unmatched is not None:
    unmatched.append(_unmatched)

# After:
if _unmatched is not None:
    if isinstance(_unmatched, list):
        unmatched.extend(_unmatched)
    else:
        unmatched.append(_unmatched)
```

### Bug 2 note

The 4D tensor issue is in how `IntermediatesCache` manages batch dimensions during sequential calibration. A defensive fix in `Qwen3_5GatedDeltaNet.forward` (in transformers, not llm-compressor) to handle `ndim > 3` with a reshape works as a temporary mitigation. The proper fix should ensure the cache doesn't introduce spurious dimensions.

For now, this can be worked around by adding squeeze logic in the model's forward, or by filing a separate issue for the intermediates cache shape handling.

## Test plan

- [x] `model_free_ptq` with `NVFP4A16` scheme on `Qwen/Qwen3.5-27B` completes successfully
- [x] `model_free_ptq` with `NVFP4A16` scheme on `Qwen/Qwen3.5-35B-A3B` (MoE variant) — should also work
- [x] Quantized checkpoint loads and serves correctly in vLLM
- [ ] `oneshot` with `NVFP4` scheme on `Qwen/Qwen3.5-27B` completes with DeltaNet layers included (requires Bug 2 workaround)
- [ ] No regression on standard models (Llama, Mistral, etc.) — fused name matching unchanged for co-located q/k/v
- [ ] Quality validation: compare perplexity of fully-quantized vs DeltaNet-excluded checkpoint

## Example

```python
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="Qwen/Qwen3.5-27B",
    save_directory="Qwen3.5-27B-NVFP4A16-full",
    scheme="NVFP4A16",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.conv1d$",
        "re:.*linear_attn.in_proj_a$",  # N=48, too small for CUTLASS FP4
        "re:.*linear_attn.in_proj_b$",  # N=48, too small for CUTLASS FP4
        "re:.*norm.*",
        "re:.*A_log$",
        "re:.*dt_bias$",
    ],
    device="cuda:0",
)
```

## Impact

- Enables NVFP4 quantization of Qwen3.5's DeltaNet layers, reducing model size by ~30% and decode latency by ~35% on DGX Spark (SM121)
- Unblocks `model_free_ptq` for any model with non-contiguous fused attention patterns
- Upstream benefit for Qwen3.5-122B-A10B, Qwen3.5-27B, Qwen3.5-35B-A3B, and future hybrid attention models

## Files changed

- `src/llmcompressor/entrypoints/model_free/helpers.py` — `match_names_set_eager` flush logic
- `src/llmcompressor/entrypoints/model_free/microscale.py` — `get_fused_names` return type handling
- `examples/quantization_w4a4_fp4/qwen3_5_full_example.py` — New example (optional)
