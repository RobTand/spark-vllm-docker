# Transformers PR: Handle extra batch dimensions in Qwen3_5GatedDeltaNet.forward

## Problem

When Qwen3.5 is traced by llm-compressor's sequential calibration pipeline
(torch.fx), intermediate activations can acquire an extra leading dimension:
`(1, 1, seq_len, hidden)` instead of `(batch, seq_len, hidden)`. This causes
`Qwen3_5GatedDeltaNet.forward` to crash at:

```python
batch_size, seq_len, _ = hidden_states.shape
# ValueError: too many values to unpack (expected 3)
```

## Fix

Squeeze leading singleton dimensions before unpacking, and unsqueeze on output
to preserve the shape contract with the calling code:

```python
# Handle extra dimensions from calibration pipelines (e.g., llm-compressor)
orig_shape = hidden_states.shape
while hidden_states.ndim > 3:
    hidden_states = hidden_states.squeeze(0)
batch_size, seq_len, _ = hidden_states.shape
```

And at the end of forward, before return:
```python
# Restore original leading dimensions if they were squeezed
output = output.view(*orig_shape[:-1], output.shape[-1])
```

This is defensive coding — the squeeze/unsqueeze is a no-op when the shape
is already 3D (normal inference path).
