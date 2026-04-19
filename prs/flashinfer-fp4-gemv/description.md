# FlashInfer PR: Triton FP4 GEMV for M=1 decode on bandwidth-limited devices

## Summary

Add a Triton-based FP4 GEMV (M=1) fast path to `mm_fp4` for small-M inference on
memory-bandwidth-limited GPUs like DGX Spark (SM121/GB10, 273 GB/s LPDDR5x).

## Problem

The CUTLASS FP4 GEMM kernel is optimized for the compute-bound regime (large M).
At M=1 (single-token decode), the kernel's fixed overhead from TMA descriptor setup,
warp specialization initialization, and multi-stage pipeline bookkeeping dominates
execution time for small weight matrices.

Measured on SM121 with Qwen3.5-122B NVFP4 shared expert projections:

| Shape (M×K @ K×N) | CUTLASS FP4 | Effective BW | torch.mm BF16 | Effective BW |
|---|---|---|---|---|
| 1×3072 @ 3072×2048 | 56 µs | 63 GB/s | 16 µs | 807 GB/s |
| 1×1024 @ 1024×3072 | 48 µs | 37 GB/s | 12 µs | 523 GB/s |
| 1×3072 @ 3072×9216 | 68 µs | 236 GB/s | 223 µs | 254 GB/s |

Small shapes achieve only 14-23% of peak bandwidth via CUTLASS, while large shapes
achieve 86%. The crossover where FP4 CUTLASS beats BF16 cuBLAS is around N×K ≈ 10M elements.

On Qwen3.5-122B, the shared expert fires 96 times per token. At 50µs avg overhead per
call, that's 4.8ms/token wasted — 11% of the total 45ms token time.

## Solution

Add a lightweight Triton GEMV kernel specialized for M=1:
- No TMA descriptors (uses simple vectorized global loads)
- No warp specialization (all threads contribute to dot products)
- No multi-stage pipeline (single pass through K dimension)
- Accepts BF16 input directly (W4A16 mode) — skips activation quantization entirely
  since the quant kernel overhead exceeds compute savings at M=1

### Integration

New runner `TritonFp4GemvRunner` in `mm_fp4` dispatch:
- Selected when M <= 4 (configurable threshold)
- Falls through to CUTLASS for larger M
- Added to the `backend_to_runner_factory` in `mm_fp4()`
- Included in auto-selection for SM120/SM121 devices

### Kernel design

```
Grid: (ceil(N / BLOCK_N),)
Each block: BLOCK_N output columns, iterates over K in chunks of BLOCK_K

Per block:
  1. Load BLOCK_K elements of input vector x (BF16 → float32)
  2. Load BLOCK_N × BLOCK_K packed FP4 weight bytes
  3. Unpack nibbles, dequant E2M1 → float32 using bit manipulation
  4. Load BLOCK_N × (BLOCK_K / 16) weight block scales
  5. Multiply and accumulate: acc[col] += sum_k(x[k] * dequant(w[col, k]) * scale)
  6. After K loop: apply global alpha, store BF16 output
```

The E2M1 dequantization uses register-level bit manipulation:
- `exp = (nibble >> 1) & 0x3, man = nibble & 0x1, sign = nibble >> 3`
- Normal (exp > 0): `value = 2^(exp-1) * (1 + 0.5 * man)`
- Subnormal (exp == 0): `value = 0.5 * man`

### Note on W4A4 vs W4A16

The GEMV kernel accepts BF16 input (W4A16 mode) rather than pre-quantized FP4
(W4A4 mode). This is intentional:

1. At M=1, the activation quantization (`scaled_fp4_quant`) costs ~18µs per call
2. The FP4×FP4 MMA compute advantage is irrelevant at M=1 (zero compute, pure bandwidth)
3. BF16 input × FP4 weight dequant achieves the same effective bandwidth
4. Skipping activation quantization saves the `scaled_fp4_quant` kernel launch entirely

For M > 4, the CUTLASS W4A4 path remains correct because the compute:bandwidth ratio
shifts and native FP4 MMA throughput matters.

## Expected performance

| Shape | CUTLASS FP4 | Triton GEMV (target) | Improvement |
|---|---|---|---|
| 1×3072 @ 3072×2048 | 56 µs | ~15 µs | 3.7× |
| 1×1024 @ 1024×3072 | 48 µs | ~12 µs | 4× |

Per-token savings on Qwen3.5-122B: ~3.4ms (96 shared expert calls × 35µs savings)
Combined with skipping `scaled_fp4_quant`: additional ~1.7ms savings

## Files changed

- `flashinfer/gemm/fp4_gemv_triton.py` — New Triton GEMV kernel
- `flashinfer/gemm/gemm_base.py` — Add `TritonFp4GemvRunner`, wire into `mm_fp4` dispatch
- `tests/gemm/test_fp4_gemv.py` — Correctness tests against CUTLASS reference

## Test plan

- [ ] Numerical correctness: GEMV output matches CUTLASS mm_fp4 within FP4 tolerance
- [ ] Performance: benchmark on SM121 shows >2× speedup for N×K < 10M
- [ ] No regression: large shapes still route to CUTLASS
- [ ] Works on SM100/SM103/SM110/SM120/SM121
