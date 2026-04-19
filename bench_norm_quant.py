"""
Benchmark: RMSNorm + FP4 Quant fusion opportunities on SM121.

Measures three paths:
  A) Current vLLM path: torch.compile'd native GemmaRMSNorm → scaled_fp4_quant
  B) FlashInfer two-step: gemma_rmsnorm → nvfp4_quantize (2 kernels, but optimized)
  C) Simulated fused: gemma_rmsnorm_fp4quant (doesn't exist yet, simulated with B)

For Qwen3.5-122B: hidden_size=3072, 48 layers, 2 norms per layer = 96 calls/token.
"""

import torch
import time
import sys


def bench_current_path(hidden_size, eps, num_iters, warmup):
    """Path A: torch native GemmaRMSNorm + vLLM's scaled_fp4_quant."""
    from vllm._custom_ops import scaled_fp4_quant

    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
    global_scale_inv = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    x = torch.randn(1, hidden_size, dtype=torch.bfloat16, device="cuda")

    # Simulate GemmaRMSNorm (native)
    def gemma_rmsnorm_native(x, w, eps):
        orig_dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(var + eps)
        xf = xf * (1.0 + w.float())
        return xf.to(orig_dtype)

    # Compile the norm
    gemma_rmsnorm_compiled = torch.compile(gemma_rmsnorm_native)

    # Warmup
    for _ in range(warmup):
        normed = gemma_rmsnorm_compiled(x, weight, eps)
        fp4, scales = scaled_fp4_quant(normed, global_scale_inv, is_sf_swizzled_layout=True)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        normed = gemma_rmsnorm_compiled(x, weight, eps)
        fp4, scales = scaled_fp4_quant(normed, global_scale_inv, is_sf_swizzled_layout=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_iters * 1e6  # microseconds


def bench_flashinfer_two_step(hidden_size, eps, num_iters, warmup):
    """Path B: FlashInfer gemma_rmsnorm + nvfp4_quantize (2 kernels)."""
    import flashinfer

    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
    global_scale_inv = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    x = torch.randn(1, hidden_size, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(warmup):
        normed = flashinfer.gemma_rmsnorm(x, weight, eps=eps)
        fp4, scales = flashinfer.nvfp4_quantize(normed, global_scale_inv,
                                                  is_sf_swizzled_layout=True)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        normed = flashinfer.gemma_rmsnorm(x, weight, eps=eps)
        fp4, scales = flashinfer.nvfp4_quantize(normed, global_scale_inv,
                                                  is_sf_swizzled_layout=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_iters * 1e6


def bench_flashinfer_rmsnorm_fp4quant(hidden_size, eps, num_iters, warmup):
    """Path C: FlashInfer fused rmsnorm_fp4quant (standard, not Gemma-style).

    This is the target kernel. Since gemma_style doesn't exist yet, we benchmark
    the standard variant to measure the fusion benefit (the Gemma +1.0 adds
    negligible overhead — one PTX add.bf16x2 per 16 elements).
    """
    import flashinfer

    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
    global_scale_inv = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    x = torch.randn(1, hidden_size, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(warmup):
        fp4, scales = flashinfer.rmsnorm_fp4quant(x, weight,
                                                    global_scale=global_scale_inv,
                                                    eps=eps,
                                                    is_sf_swizzled_layout=True)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        fp4, scales = flashinfer.rmsnorm_fp4quant(x, weight,
                                                    global_scale=global_scale_inv,
                                                    eps=eps,
                                                    is_sf_swizzled_layout=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_iters * 1e6


def bench_just_quant(hidden_size, eps, num_iters, warmup):
    """Measure just the scaled_fp4_quant overhead alone."""
    from vllm._custom_ops import scaled_fp4_quant

    global_scale_inv = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    x = torch.randn(1, hidden_size, dtype=torch.bfloat16, device="cuda")

    for _ in range(warmup):
        fp4, scales = scaled_fp4_quant(x, global_scale_inv, is_sf_swizzled_layout=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        fp4, scales = scaled_fp4_quant(x, global_scale_inv, is_sf_swizzled_layout=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_iters * 1e6


def main():
    hidden_size = 3072  # Qwen3.5
    eps = 1e-6
    num_iters = 1000
    warmup = 100
    calls_per_token = 96  # 48 layers × 2 norms

    print(f"Hidden size: {hidden_size}")
    print(f"Iterations: {num_iters}")
    print(f"Norms per token: {calls_per_token}")
    print()

    # Path A: Current vLLM path
    try:
        us_a = bench_current_path(hidden_size, eps, num_iters, warmup)
        print(f"A) Current (compiled native norm + scaled_fp4_quant): {us_a:.1f} µs/call")
        print(f"   Per token ({calls_per_token} calls): {us_a * calls_per_token / 1000:.2f} ms")
    except Exception as e:
        print(f"A) Current path failed: {e}")
        us_a = None

    # Just quant
    try:
        us_q = bench_just_quant(hidden_size, eps, num_iters, warmup)
        print(f"   (scaled_fp4_quant alone: {us_q:.1f} µs/call)")
    except Exception as e:
        print(f"   (quant failed: {e})")

    # Path B: FlashInfer two-step
    try:
        us_b = bench_flashinfer_two_step(hidden_size, eps, num_iters, warmup)
        print(f"B) FlashInfer two-step (gemma_rmsnorm + nvfp4_quantize): {us_b:.1f} µs/call")
        print(f"   Per token ({calls_per_token} calls): {us_b * calls_per_token / 1000:.2f} ms")
    except Exception as e:
        print(f"B) FlashInfer two-step failed: {e}")
        us_b = None

    # Path C: FlashInfer fused (standard rmsnorm, simulating gemma)
    try:
        us_c = bench_flashinfer_rmsnorm_fp4quant(hidden_size, eps, num_iters, warmup)
        print(f"C) FlashInfer fused rmsnorm_fp4quant (1 kernel): {us_c:.1f} µs/call")
        print(f"   Per token ({calls_per_token} calls): {us_c * calls_per_token / 1000:.2f} ms")
    except Exception as e:
        print(f"C) FlashInfer fused failed: {e}")
        us_c = None

    print()
    if us_a and us_c:
        savings_per_call = us_a - us_c
        savings_per_token = savings_per_call * calls_per_token / 1000
        base_token_time = 1000 / 22  # ~45ms at 22 tok/s base
        speedup_pct = savings_per_token / base_token_time * 100
        new_tok_s = 1000 / (base_token_time - savings_per_token)
        print(f"Savings (A→C): {savings_per_call:.1f} µs/call, {savings_per_token:.2f} ms/token")
        print(f"At 22 tok/s base ({base_token_time:.1f} ms/token):")
        print(f"  Speedup: {speedup_pct:.1f}%")
        print(f"  New base: {new_tok_s:.1f} tok/s")


if __name__ == "__main__":
    main()
