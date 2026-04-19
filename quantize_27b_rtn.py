"""
Quantize Qwen3.5-27B to NVFP4 using model_free_ptq (calibration-free RTN).

This avoids the calibration forward pass that crashes on DeltaNet layers.
RTN (round-to-nearest) quantization computes scales from weight statistics
alone — no calibration data needed.

The trade-off: slightly less optimal scales than calibrated quantization,
but the quality difference is typically small for weight-only quantization.
"""

from llmcompressor import model_free_ptq

MODEL_ID = "Qwen/Qwen3.5-27B"
SAVE_DIR = "/models/Qwen3.5-27B-NVFP4-full-rtn"

print(f"Running model_free_ptq on {MODEL_ID}...")
print(f"Output: {SAVE_DIR}")

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="NVFP4A16",  # Weight-only NVFP4 (no activation quantization metadata)
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.conv1d$",  # conv1d is 3D
        "re:.*linear_attn.in_proj_a$",  # N=48, too small for CUTLASS FP4 (needs N%64==0)
        "re:.*linear_attn.in_proj_b$",  # N=48, same issue
        "re:.*norm.*",                # norm weights are 1D
        "re:.*A_log$",               # DeltaNet state params
        "re:.*dt_bias$",             # DeltaNet state params
    ],
    device="cuda:0",
)

print("Done!")
