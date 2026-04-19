"""
Targeted GEMM benchmarks for Qwen3.5-122B NVFP4 decode on SM121.
Measures each GEMM size that appears in the model forward pass.
"""

import torch
import time

def bench_cuda_event(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return sum(times[10:]) / len(times[10:])


def main():
    import flashinfer
    from vllm._custom_ops import scaled_fp4_quant

    device = "cuda"
    H = 3072  # Qwen3.5 hidden
    E_INT = 1024  # expert intermediate
    S_INT = 2048  # shared expert intermediate (gate+up merged = 2*1024)

    print("=" * 70)
    print("GEMM BENCHMARKS — Qwen3.5-122B decode (M=1)")
    print("=" * 70)
    print()

    # 1. Dense FP4 GEMMs via flashinfer.mm_fp4
    gemm_configs = [
        ("shared_expert gate_up", 1, S_INT, H),    # (1, 3072) → (1, 2048)
        ("shared_expert down",    1, H, E_INT),     # (1, 1024) → (1, 3072)
        ("qkv_proj (full_attn)",  1, 8192+512+512, H),  # merged QKV for 32 heads
        ("o_proj (full_attn)",    1, H, 8192),      # output proj
    ]

    for name, M, N, K in gemm_configs:
        a = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device=device)
        b = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device=device)
        a_sf = torch.ones(M, K // 16, dtype=torch.float8_e4m3fn, device=device)
        b_sf = torch.ones(N, K // 16, dtype=torch.float8_e4m3fn, device=device)
        alpha = torch.tensor([1.0], dtype=torch.float32, device=device)

        try:
            ms = bench_cuda_event(lambda: flashinfer.mm_fp4(
                a, b.t(), a_sf, b_sf.t(), alpha,
                out_dtype=torch.bfloat16, backend="cutlass"))
            bytes_loaded = (K // 2) * N + (K // 16) * N  # weight + scale bytes
            bw_gbps = bytes_loaded / (ms / 1000) / 1e9
            print(f"  {name:30s} ({M}×{K} @ {K}×{N}): {ms*1000:7.1f} µs  [{bw_gbps:.1f} GB/s eff]")
        except Exception as e:
            print(f"  {name:30s}: FAILED ({e})")

    print()

    # 2. BF16 matmul for comparison (this is what batch=1 decode could achieve
    # if weights were in BF16 — the memory bandwidth ceiling)
    print("--- BF16 torch.mm comparison ---")
    for name, M, N, K in gemm_configs:
        w = torch.randn(K, N, dtype=torch.bfloat16, device=device)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        ms = bench_cuda_event(lambda: torch.mm(x, w))
        bytes_loaded = K * N * 2  # BF16 = 2 bytes
        bw_gbps = bytes_loaded / (ms / 1000) / 1e9
        print(f"  {name:30s} ({M}×{K} @ {K}×{N}): {ms*1000:7.1f} µs  [{bw_gbps:.1f} GB/s eff]")

    print()

    # 3. scaled_fp4_quant overhead
    print("--- Activation quantization ---")
    x_bf16 = torch.randn(1, H, dtype=torch.bfloat16, device=device)
    gs = torch.tensor([0.5], dtype=torch.float32, device=device)
    ms_q = bench_cuda_event(lambda: scaled_fp4_quant(x_bf16, gs, is_sf_swizzled_layout=True))
    print(f"  scaled_fp4_quant (1×{H}):     {ms_q*1000:7.1f} µs")

    ms_fi = bench_cuda_event(lambda: flashinfer.rmsnorm_fp4quant(
        x_bf16, torch.randn(H, dtype=torch.bfloat16, device=device),
        global_scale=gs, eps=1e-6, is_sf_swizzled_layout=True))
    print(f"  fused rmsnorm_fp4quant (1×{H}): {ms_fi*1000:7.1f} µs")

    print()

    # 4. Per-token cost estimate
    print("=" * 70)
    print("PER-TOKEN COST ESTIMATE (48 layers, M=1)")
    print("=" * 70)

    # At 22 tok/s base, each token = ~45ms
    # MoE is handled by fused kernel (not benchmarked here due to API complexity)
    # Dense components we measured:
    # Per full_attn layer (12): 2×norm_quant + qkv_gemm + o_proj_gemm + gate_up_gemm + down_gemm
    # Per linear_attn layer (36): 2×norm_quant + gate_up_gemm + down_gemm (DeltaNet attn is BF16)
    print()
    print(f"  Total token time at 22 tok/s: ~45 ms")
    print(f"  What we measured above covers only dense projections.")
    print(f"  The MoE expert GEMMs (fused kernel) are the dominant cost.")
    print(f"  Need to benchmark the fused MoE kernel separately.")
    print(f"  Bandwidth ceiling: 273 GB/s")


if __name__ == "__main__":
    main()
