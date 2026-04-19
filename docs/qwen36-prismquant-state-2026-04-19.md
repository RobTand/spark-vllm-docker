# Qwen3.6 PrismQuant State - 2026-04-19

## Summary

The repo now has a working end-to-end path from PrismQuant analysis to a locally served
`Qwen3.6-35B-A3B` artifact on a single GB10/Spark-class machine.

The current served artifact is:

- `/home/rob/spark-vllm-docker/Qwen3.6-35B-A3B-DQ-SERVE`

This is a real exported checkpoint, not just a recipe stub. It is loadable by the
repo-patched vLLM path and has been verified to serve completions successfully.

## What Works

### 1. PrismQuant analysis on Qwen3.6 MoE

The Qwen3.6 MoE path was fixed so that PrismQuant correctly sees:

- packed experts
- routers
- per-layer MoE structure

Probe/cost now run incrementally and persistently:

- probe no longer needs monolithic all-layer hooks
- quant-cost no longer reloads the full model per shard

Important artifacts:

- probe: `dq-runs/qwen36-opt-l2/artifacts/probe.pkl`
- cost: `dq-runs/qwen36-cost-persistent/artifacts/cost.pkl`

### 2. Serving-aware allocator path

The allocator now supports a packed-Qwen MoE serving profile so it can emit recipes
that are legal for the existing runtime constraints instead of mathematically nice but
undeployable ones.

Key recent changes:

- packed Qwen MoE aggregation in allocator
- canonicalization of Qwen expert tensor names in cost measurement
- packed-MoE serving profile for legal format choices

### 3. Qwen3.6 materialization

The direct `llmcompressor` mixed-native path was not sufficient for packed Qwen3.6 MoE.
The working route instead uses the repo’s PrismQuant export/runtime path:

- `quantization/export_prismquant_ct.py`
- `quantization/prismquant_pkg/patch_vllm.py`
- `quantization/prismquant_pkg/prismquant_moe.py`

This produced:

- `/home/rob/spark-vllm-docker/Qwen3.6-35B-A3B-DQ-SERVE`

Export accounting reported:

- packed: `17.71 GB`
- passthrough: `5.47 GB`
- total logical size: `23.18 GB`

Disk footprint is larger because of safetensor shard layout and metadata.

### 4. Local serving

The artifact has been served successfully in vLLM on this machine.

Confirmed:

- server startup succeeded
- `/v1/models` responded
- `/v1/completions` responded
- prompt `The capital of France is` returned a valid completion beginning with `Paris`

## Runtime Reality

### Current stable launch posture

The stable configuration is the conservative one:

- `--enforce-eager`
- `--host 0.0.0.0`
- `--port 8000`
- `--max-model-len 32768`
- `--max-num-batched-tokens 8192`
- `--enable-prefix-caching`
- `--kv-cache-dtype fp8`
- `--attention-backend flashinfer`
- `--gpu-memory-utilization 0.50`

This is slower than an ideal compiled/cudagraph path, but it is stable.

### Why the compiled path blew up

The failure was not just generic “CUDA graphs are expensive”.

The failing compile-enabled vLLM path attempted:

- `FULL_AND_PIECEWISE` graph capture
- `51` piecewise graphs
- `35` full decode graphs

The log estimated:

- `91.26 GiB` CUDA graph memory

That is the main reason the compiled path was not viable with the default settings.

The lower-risk compile experiment using:

- `-cc.cudagraph_mode=piecewise`

was clearly healthier than the default full-plus-piecewise path, but the current
recommended production posture remains eager mode until the compile path is tuned
properly for this model.

### Host RAM spike explanation

The large host-RAM jump during startup was mostly page cache from reading the large
safetensor shards, not a second giant anonymous Python copy of the model.

Observed in the safe/eager debug run:

- process RSS stayed relatively small
- `buff/cache` climbed sharply

The dangerous startup behavior happened when that page-cache pressure was combined
with compile/warmup/cudagraph overhead.

## MTP Status

MTP is preserved in the artifact, but mostly as passthrough.

Verified:

- MTP tensors are still present in `model.safetensors.index.json`
- MTP-related modules are explicitly listed in the quantization `ignore` set

So the current export preserves the MTP heads/modules, but does not aggressively
quantize them.

## Important Constraints

### Upstreamability

The current PrismQuant MoE runtime path is a local proof path, not an upstream-friendly
vLLM architecture.

It relies on:

- repo-specific packed format assumptions
- repo-specific vLLM patching
- repo-specific MoE dispatch logic

This is good enough for local deployment and research iteration, but should be treated
as a bridge, not the final upstream design.

### Standard native path

The standard native mixed-format route is still preferable long-term where possible:

- NVFP4
- FP8
- MXFP8

But packed Qwen3.6 MoE currently required the repo’s PrismQuant-specific runtime bridge
to get a real served model.

## Current Recommended Launch

The intended safe command shape is:

```bash
vllm serve /workspace/Qwen3.6-35B-A3B-DQ-SERVE \
  --quantization compressed-tensors \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --enforce-eager
```

## Near-Term Next Steps

1. Keep only one live server process at a time.
   Repeated retries created duplicate vLLM trees during debugging.

2. Benchmark the stable eager server.
   Measure tokens/sec and memory under real prompt loads before further optimization.

3. Revisit compile mode with `piecewise` only.
   If performance work resumes, the correct starting point is:
   - compile enabled
   - `piecewise` graphs only
   - no return to `FULL_AND_PIECEWISE`

4. Decide whether to spend effort on:
   - making the PrismQuant custom MoE path faster locally
   - or translating the winning PrismQuant policy into a more upstream-compatible format/runtime path

## Files To Know

- `quantization/prismquant/allocator.py`
- `quantization/prismquant/measure_quant_cost.py`
- `quantization/export_prismquant_ct.py`
- `quantization/prismquant_pkg/patch_vllm.py`
- `quantization/prismquant_pkg/prismquant_moe.py`
- `recipes/qwen3.5-35b-a3b-prismquant.yaml`

## Bottom Line

The project crossed the important threshold:

- PrismQuant analysis works on Qwen3.6 MoE
- a local exported artifact exists
- that artifact serves in vLLM on this hardware

The remaining work is mostly about runtime polish and performance, not rescue.
