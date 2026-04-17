# DynaQuant — mixed-precision quantization allocator for LLMs

Model-agnostic, format-extensible, closed-loop-measured. Pairs naturally
with [AutoRound](https://github.com/intel/auto-round) for the final
calibrated quantization pass.

## Why?

Mixed-precision LLM quantization (e.g. 4.75-bit avg with NVFP4 baseline
and MXFP8 for sensitive layers) is a bit-budget allocation problem:
given N layers and K candidate formats, pick a format per layer that
minimizes quality loss subject to a total-bit constraint.

Most existing tools either (a) pick a single format for every layer, or
(b) use a fixed-pattern heuristic (e.g. "down_proj always higher
precision"). DynaQuant measures the actual per-layer curvature of the
loss and the actual per-(layer, format) quantization error, then runs a
proper multi-choice knapsack over them.

## Pipeline

    1.  sensitivity_probe.py       per-Linear Fisher trace (route-aware)
    2.  measure_quant_cost.py      per-(Linear, format) measured RTN error
    3.  allocator.py               multi-choice knapsack → AutoRound config
    4.  (external) auto-round --layer_config …   calibrated tuning

### 1. Sensitivity probe

Streaming backward with hooks so no full gradient tensor ever materializes.
Fisher trace (g²) per Linear. Route-aware weighting for MoE experts (divide
by observed routing probability so sparse experts' Fisher is comparable to
dense layers'). Per-token importance weighting (hard tokens count more).

```
python -m dynaquant.sensitivity_probe \
  --model $MODEL_PATH --dataset ultrachat_200k \
  --nsamples 32 --seqlen 1024 \
  --device cuda --dtype bf16 \
  --output probe.pkl --activation-cache-dir ./act_cache
```

Memory peak on 35B: ~90 GB (fits 128 GB unified). Runtime: ~5 min GPU,
~4 hr CPU.

### 2. Measure quantization cost

For each tracked Linear and each registered format, apply RTN, measure
`‖W·x - Ŵ·x‖²` on saved activations. Replaces analytical constants with
measured quantities.

```
python -m dynaquant.measure_quant_cost \
  --model $MODEL_PATH --probe probe.pkl \
  --activation-cache-dir ./act_cache \
  --formats NVFP4,MXFP4,MXFP6_E3M2,MXFP8 \
  --output costs.pkl
```

Memory: streams weights one at a time, < 5 GB overhead. Runtime: ~3 min
GPU, ~15 min CPU.

### 3. Allocate

Multi-choice knapsack DP. Per-Linear `Δloss ≈ 0.5 · H_trace ·
output_mse · out_features`. Minimize total Δloss subject to
`Σ bits ≤ target`. Fused-projection siblings (q/k/v, gate/up) promoted
to the highest-precision sibling.

```
python -m dynaquant.allocator \
  --probe probe.pkl --costs costs.pkl \
  --target-bits 4.75 \
  --formats NVFP4,MXFP4,MXFP6_E3M2,MXFP8 \
  --layer-config layer_config.json \
  --pareto-csv pareto.csv
```

Outputs:
- `layer_config.json` — drop-in for AutoRound's `--layer_config`
- `pareto.csv` — Δloss vs bits across budget sweep
- Printed Kneedle knee suggestion

### 4. AutoRound tuning

AutoRound respects our per-layer assignment and runs its calibrated
signed-SGD tuning within each layer's scheme:

```
python -m auto_round \
  --model_name $MODEL_PATH \
  --layer_config layer_config.json \
  --iters 200 --nsamples 128 --seqlen 2048 \
  --dataset ultrachat_200k \
  --format llm_compressor --output_dir ./out \
  --device_map cuda:0
```

## Extending formats

Any new microscaling or uniform-int format can be added by registering a
`FormatSpec` in `format_registry.py`:

```python
register_format(FormatSpec(
    name="MXFP4_E3M0",             # hypothetical variant
    weight_bits=4, group_size=32, scale_bits=8,
    scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp4_e3m0",
    family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(4, 32, 4, "fp4_e3m0"),
    quantize_dequantize=_make_rtn("fp4_e3m0", 32),  # add codebook
))
```

Built-in formats:
- **NVFP4**, **NVFP4A16** — NVIDIA NVFP4 (group 16, FP8 scales)
- **MXFP4**, **MXFP6_E3M2**, **MXFP6_E2M3**, **MXFP8**, **MXFP8A16** —
  OCP MX formats (group 32, E8M0 uint8 scales)
- **INT8_W8A16**, **INT4_W4A16_g128** — classic uniform-integer
- **BF16** — passthrough for must-preserve layers

### Recommended format bundles

**NVFP4 and MXFP4 are alternatives for the same 4-bit tier, not separate
precision levels.** Include at most one format per bit tier — otherwise
the allocator picks between them based on per-layer RTN measurement
noise and you end up with a serving mess (two kernel paths for 4-bit
quant). The allocator warns on this by default and errors with
`--enforce-family-coherence`.

#### Hardware + serving-stack support

Everything in this section assumes you're serving with vLLM. The
microscaling formats (NVFP4, MX*) all require NVIDIA Blackwell-era
hardware (SM100+) for native kernel support; on older Ampere/Ada you
get Marlin emulation, which works but at a significant speed penalty.

|                 | Hardware (Blackwell ISA) | vLLM serving kernels today |
|-----------------|-------------------------:|---------------------------:|
| NVFP4           | ✓                        | ✓ (FlashInfer CUTLASS)     |
| MXFP4           | ✓                        | ✓ (FlashInfer CUTLASS)     |
| MXFP6_E3M2      | ✓                        | ✗ (hardware supports it,   |
|                 |                          |    vLLM kernel not yet     |
|                 |                          |    integrated)             |
| MXFP6_E2M3      | ✓                        | ✗ (same)                   |
| MXFP8           | ✓                        | ✓ (FlashInfer CUTLASS)     |
| INT4 / INT8     | (all NV HW)              | ✓ (Marlin)                 |

Until vLLM picks up MXFP6 serving kernels, including `MXFP6_*` in a
bundle means the allocator can pick it, the checkpoint will contain
it, but vLLM won't know how to load it at serve time. Safe to
experiment with on the quantization side; do not ship until the
kernels land.

| Use case                          | Recommended `--formats`           |
|-----------------------------------|-----------------------------------|
| Today, ship on Blackwell via vLLM | `NVFP4,MXFP8` (validated path)    |
| Today, MX-pure on Blackwell       | `MXFP4,MXFP8`                     |
| Experimental, Blackwell w/ MXFP6  | `NVFP4,MXFP6_E3M2,MXFP8`          |
| Legacy INT pipelines              | `INT4_W4A16_g128,INT8_W8A16`      |

## Method notes

### Is this gradient descent?

No. `requires_grad_(False)` on all parameters. Backward runs only to push
gradient signal through autograd so hooks can read it; nothing is
written back. It's a sensitivity measurement, not an optimizer.

### Why Fisher and not Hutchinson?

Hutchinson on a Linear's weights via vHv probes requires a different
hook architecture than we use (hooks see activation gradients, not
parameter gradients). Fisher (g²) is the natural fit for hooks and
gives a first-order proxy for curvature that correlates well with
quantization sensitivity when combined with measured RTN error
(which removes the need for Fisher to predict anything — it only needs
to rank layers).

### Why measured RTN error over analytical formulas?

The uniform-quantization MSE formula overweights max-magnitude outliers
and doesn't model non-uniform FP codebooks. Running RTN once and
measuring `‖W·x - Ŵ·x‖²` captures the actual distribution of the
weight tensor and the actual functional perturbation at the layer
output — no tuning constants, no assumption about weight distributions.

### What about inter-layer interactions?

The multi-choice knapsack assumes per-layer costs are independent. This
is the standard approximation for mixed-precision quantization and is
well-supported empirically at typical bit budgets (4–6 bits avg).
Cross-layer correlation terms would require pair-wise Hessian
measurements, which is prohibitively expensive. AutoRound's block-wise
calibration implicitly captures short-range interactions within each
block (typically a transformer layer), so the combined pipeline handles
this.

## Memory budget

| Stage             | Peak RAM    | Peak VRAM (GB10) |
|-------------------|-------------|------------------|
| sensitivity_probe | 90 GB (35B) | (unified) 90 GB  |
| measure_quant_cost| 75 GB       | 75 GB            |
| allocator         | < 1 GB      | n/a              |

Fits 128 GB unified systems for models up to ~48 B parameters.
