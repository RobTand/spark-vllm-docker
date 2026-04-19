"""
Inline CUDA profiling via the vLLM server's /v1/completions endpoint.

Instead of profiling externally, we instrument the client-side timing
and use CUDA_LAUNCH_BLOCKING to serialize kernel execution, giving us
per-kernel wallclock time from torch profiler.

This script runs INSIDE the vLLM container alongside the server.
It imports vLLM internals directly to profile a single forward pass.
"""

import torch
import json
import time
import sys
import os

# We need to profile the actual model forward pass.
# The cleanest way is to use torch.profiler on a standalone forward pass
# using the same model weights that vLLM loaded.

# Alternative: use CUDA_LAUNCH_BLOCKING + client timing
# Let's go with torch.profiler tracing all CUDA kernels.

def profile_via_torch_profiler():
    """Use torch.profiler to capture all CUDA activity during inference."""

    print("Setting up profiler...")

    # Send a request while profiling CUDA from this process
    # Since we're in a different process, we need to trace via the OS

    # Actually, the simplest approach: use torch.cuda.Event timing
    # on a synthetic forward pass that loads the same model

    # Even simpler: use the CUDA profiling API via cupti
    # Set CUDA_INJECTION64_PATH or use nsys start/stop API

    # Let's just parse what we already know and do targeted microbenchmarks
    # of the MoE GEMM path, which is the dominant cost.

    import flashinfer

    device = "cuda"
    dtype = torch.bfloat16

    # Qwen3.5 MoE dimensions:
    # 256 experts, top-8, expert hidden=1024, model hidden=3072
    # MoE GEMM1: (8, 3072) × (3072, 2048) per expert (gate_up merged)
    # MoE GEMM2: (8, 1024) × (1024, 3072) per expert (down_proj)
    # But at batch=1 with top-8, it's 8 experts each processing 1 token

    # The fused MoE kernel handles routing + GEMM1 + activation + GEMM2 in one call.
    # Let's measure it directly using FlashInfer's API.

    num_experts = 256
    top_k = 8
    hidden = 3072
    intermediate = 1024  # per expert

    # Simulate MoE weights (FP4 packed)
    # w1: gate_up = (num_experts, 2*intermediate, hidden) in FP4 = (256, 2048, 3072/2) uint8
    # w2: down = (num_experts, hidden, intermediate) in FP4 = (256, 3072, 1024/2) uint8
    w1 = torch.randint(0, 255, (num_experts, 2 * intermediate, hidden // 2),
                        dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 255, (num_experts, hidden, intermediate // 2),
                        dtype=torch.uint8, device=device)

    # Scales
    w1_scale = torch.ones(num_experts, 2 * intermediate, hidden // 16,
                           dtype=torch.float8_e4m3fn, device=device)
    w2_scale = torch.ones(num_experts, hidden, intermediate // 16,
                           dtype=torch.float8_e4m3fn, device=device)

    # Input (1 token)
    x = torch.randn(1, hidden, dtype=dtype, device=device)

    # Routing (top-8 of 256)
    topk_ids = torch.randint(0, num_experts, (1, top_k), dtype=torch.int32, device=device)
    topk_weights = torch.ones(1, top_k, dtype=torch.float32, device=device) / top_k

    # Global alpha scales
    g1_alphas = torch.ones(num_experts, dtype=torch.float32, device=device)
    g2_alphas = torch.ones(num_experts, dtype=torch.float32, device=device)
    a1_gscale = torch.tensor([1.0], dtype=torch.float32, device=device)
    a2_gscale = torch.tensor([1.0], dtype=torch.float32, device=device)

    print(f"MoE config: {num_experts} experts, top-{top_k}, hidden={hidden}, intermediate={intermediate}")
    print(f"w1 shape: {w1.shape}, w2 shape: {w2.shape}")
    print()

    # Try the FlashInfer fused MoE directly
    try:
        output = torch.zeros(1, hidden, dtype=dtype, device=device)

        # Warmup
        for _ in range(5):
            flashinfer.cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_ids,
                token_final_scales=topk_weights,
                fc1_expert_weights=w1.view(torch.long),
                fc2_expert_weights=w2.view(torch.long),
                fc1_expert_biases=None,
                fc2_expert_biases=None,
                output=output,
                output_dtype=dtype,
                quant_scales=[a1_gscale, w1_scale.view(torch.int32), g1_alphas,
                              a2_gscale, w2_scale.view(torch.int32), g2_alphas],
                activation_type=1,  # SwiGLU
                tune_max_num_tokens=1,
            )
        torch.cuda.synchronize()

        # Benchmark
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(50)]

        for i in range(50):
            start_events[i].record()
            flashinfer.cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_ids,
                token_final_scales=topk_weights,
                fc1_expert_weights=w1.view(torch.long),
                fc2_expert_weights=w2.view(torch.long),
                fc1_expert_biases=None,
                fc2_expert_biases=None,
                output=output,
                output_dtype=dtype,
                quant_scales=[a1_gscale, w1_scale.view(torch.int32), g1_alphas,
                              a2_gscale, w2_scale.view(torch.int32), g2_alphas],
                activation_type=1,
                tune_max_num_tokens=1,
            )
            end_events[i].record()
        torch.cuda.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        avg_ms = sum(times[5:]) / len(times[5:])  # skip warmup
        print(f"FlashInfer fused MoE (CUTLASS FP4):  {avg_ms:.3f} ms/call")
        print(f"  Per token (48 layers):             {avg_ms * 48:.1f} ms")

    except Exception as e:
        print(f"FlashInfer fused MoE failed: {e}")
        import traceback
        traceback.print_exc()

    # Also benchmark a simple BF16 matmul for comparison (what Marlin INT4 effectively does)
    print()
    w_bf16 = torch.randn(hidden, 2 * intermediate, dtype=dtype, device=device)
    for _ in range(10):
        _ = torch.mm(x, w_bf16)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
    for i in range(50):
        start_events[i].record()
        _ = torch.mm(x, w_bf16)
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_mm = sum(times[5:]) / len(times[5:])
    print(f"torch.mm BF16 (1x3072 @ 3072x2048): {avg_mm:.3f} ms/call")
    print(f"  (This is the compute floor for a single expert-sized GEMM)")


if __name__ == "__main__":
    profile_via_torch_profiler()
