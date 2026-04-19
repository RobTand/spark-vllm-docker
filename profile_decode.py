"""
Profile a single decode step of Qwen3.5-122B NVFP4 to get kernel-level timing.

Sends a request to the running vLLM server, waits for it to start generating,
then profiles subsequent decode steps using CUDA events.

Usage: python3 profile_decode.py [--nsys]
  --nsys: Run under nsys (call this script via nsys profile python3 profile_decode.py)
"""

import torch
import time
import sys
import os


def profile_with_cuda_events():
    """Profile by sending requests and examining vLLM internals."""
    # We can't directly instrument the vLLM server process.
    # Instead, let's profile the individual kernel calls that make up a decode step.

    import flashinfer
    from vllm._custom_ops import scaled_fp4_quant

    torch.cuda.synchronize()

    # Qwen3.5-122B dimensions
    hidden_size = 3072
    intermediate_size = 1024  # shared expert intermediate
    num_experts = 256
    top_k = 8
    expert_intermediate = 1024
    num_layers = 48
    num_linear_attn = 36
    num_full_attn = 12

    device = "cuda"
    dtype = torch.bfloat16

    # Simulate one token decode through the model
    # Measure each component separately

    results = {}

    # 1. RMSNorm (Gemma-style, compiled native)
    weight_norm = torch.randn(hidden_size, dtype=dtype, device=device)
    x = torch.randn(1, hidden_size, dtype=dtype, device=device)
    eps = 1e-6

    def gemma_rmsnorm(x, w, eps):
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(var + eps)
        xf = xf * (1.0 + w.float())
        return xf.to(x.dtype)

    gemma_rmsnorm_c = torch.compile(gemma_rmsnorm)
    # Warmup compile
    for _ in range(10):
        _ = gemma_rmsnorm_c(x, weight_norm, eps)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]

    for i in range(100):
        start_events[i].record()
        _ = gemma_rmsnorm_c(x, weight_norm, eps)
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results["gemma_rmsnorm"] = sum(times[10:]) / len(times[10:])  # skip first 10

    # 2. scaled_fp4_quant
    global_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    normed = torch.randn(1, hidden_size, dtype=dtype, device=device)

    for _ in range(10):
        scaled_fp4_quant(normed, global_scale, is_sf_swizzled_layout=True)
    torch.cuda.synchronize()

    for i in range(100):
        start_events[i].record()
        scaled_fp4_quant(normed, global_scale, is_sf_swizzled_layout=True)
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results["scaled_fp4_quant"] = sum(times[10:]) / len(times[10:])

    # 3. FlashInfer fused rmsnorm_fp4quant
    for _ in range(10):
        flashinfer.rmsnorm_fp4quant(x, weight_norm, global_scale=global_scale,
                                     eps=eps, is_sf_swizzled_layout=True)
    torch.cuda.synchronize()

    for i in range(100):
        start_events[i].record()
        flashinfer.rmsnorm_fp4quant(x, weight_norm, global_scale=global_scale,
                                     eps=eps, is_sf_swizzled_layout=True)
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results["fused_rmsnorm_fp4quant"] = sum(times[10:]) / len(times[10:])

    # 4. CUTLASS FP4 dense GEMM (simulating qkv_proj)
    # M=1, N=3*32*256=24576 (merged QKV), K=3072
    M, N, K = 1, 24576, hidden_size
    a_fp4 = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device=device)
    b_fp4 = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device=device)
    a_sf = torch.ones(1, K // 16, dtype=torch.float8_e4m3fn, device=device)
    b_sf = torch.ones(N, K // 16, dtype=torch.float8_e4m3fn, device=device)
    alpha = torch.tensor([1.0], dtype=torch.float32, device=device)

    try:
        for _ in range(10):
            flashinfer.mm_fp4(a_fp4, b_fp4.t(), a_sf, b_sf.t(), alpha,
                              out_dtype=dtype, backend="cutlass")
        torch.cuda.synchronize()

        for i in range(100):
            start_events[i].record()
            flashinfer.mm_fp4(a_fp4, b_fp4.t(), a_sf, b_sf.t(), alpha,
                              out_dtype=dtype, backend="cutlass")
            end_events[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        results["cutlass_fp4_gemm_qkv"] = sum(times[10:]) / len(times[10:])
    except Exception as e:
        results["cutlass_fp4_gemm_qkv"] = f"FAILED: {e}"

    # 5. CUTLASS FP4 dense GEMM (simulating gate_up_proj: M=1, N=2048, K=3072)
    M, N, K = 1, 2048, hidden_size
    a_fp4 = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device=device)
    b_fp4 = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device=device)
    a_sf = torch.ones(1, K // 16, dtype=torch.float8_e4m3fn, device=device)
    b_sf = torch.ones(N, K // 16, dtype=torch.float8_e4m3fn, device=device)

    try:
        for _ in range(10):
            flashinfer.mm_fp4(a_fp4, b_fp4.t(), a_sf, b_sf.t(), alpha,
                              out_dtype=dtype, backend="cutlass")
        torch.cuda.synchronize()

        for i in range(100):
            start_events[i].record()
            flashinfer.mm_fp4(a_fp4, b_fp4.t(), a_sf, b_sf.t(), alpha,
                              out_dtype=dtype, backend="cutlass")
            end_events[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        results["cutlass_fp4_gemm_gate_up"] = sum(times[10:]) / len(times[10:])
    except Exception as e:
        results["cutlass_fp4_gemm_gate_up"] = f"FAILED: {e}"

    # 6. FlashInfer attention (simulating FlashInfer decode for full_attn layers)
    # Skip - too complex to set up standalone

    # Print results
    print("=" * 70)
    print("KERNEL TIMING BREAKDOWN (per call, batch=1, hidden=3072)")
    print("=" * 70)

    for name, t in results.items():
        if isinstance(t, float):
            print(f"  {name:40s}: {t*1000:8.1f} µs")
        else:
            print(f"  {name:40s}: {t}")

    print()
    print("=" * 70)
    print("ESTIMATED PER-TOKEN COST (48 layers)")
    print("=" * 70)

    # Per layer costs
    norm_cost = results.get("gemma_rmsnorm", 0)
    quant_cost = results.get("scaled_fp4_quant", 0)
    fused_cost = results.get("fused_rmsnorm_fp4quant", 0)
    qkv_gemm = results.get("cutlass_fp4_gemm_qkv", 0)
    gate_up_gemm = results.get("cutlass_fp4_gemm_gate_up", 0)

    if isinstance(qkv_gemm, str) or isinstance(gate_up_gemm, str):
        print("  GEMM benchmarks failed - skipping per-token estimate")
        return

    # Current path per layer:
    # 2× (norm + quant) + QKV GEMM + o_proj GEMM + gate_up GEMM + down_proj GEMM
    # For full_attn layers (12): 2×norm + 2×quant + 4×GEMM
    # For linear_attn layers (36): 2×norm + 2×quant + 2×GEMM (gate_up + down only, no attn proj quant)

    current_full_attn = 2 * (norm_cost + quant_cost) + 2 * qkv_gemm + 2 * gate_up_gemm
    current_linear_attn = 2 * (norm_cost + quant_cost) + 2 * gate_up_gemm
    current_total = 12 * current_full_attn + 36 * current_linear_attn

    fused_full_attn = 2 * fused_cost + 2 * qkv_gemm + 2 * gate_up_gemm
    fused_linear_attn = 2 * fused_cost + 2 * gate_up_gemm
    fused_total = 12 * fused_full_attn + 36 * fused_linear_attn

    print(f"  Current norm+quant per layer:  {(norm_cost + quant_cost)*1000*2:.1f} µs")
    print(f"  Fused norm+quant per layer:    {fused_cost*1000*2:.1f} µs")
    print(f"  GEMM (QKV) per call:           {qkv_gemm*1000:.1f} µs")
    print(f"  GEMM (gate_up) per call:       {gate_up_gemm*1000:.1f} µs")
    print()
    print(f"  Current total (norm+quant+GEMM only): {current_total:.2f} ms")
    print(f"  Fused total:                          {fused_total:.2f} ms")
    print(f"  Savings:                              {current_total - fused_total:.2f} ms")
    print()
    print(f"  Note: This excludes MoE expert GEMMs, attention, DeltaNet,")
    print(f"  routing, norms between layers, MTP overhead, etc.")
    print(f"  These are just the dense projection components.")


if __name__ == "__main__":
    profile_with_cuda_events()
