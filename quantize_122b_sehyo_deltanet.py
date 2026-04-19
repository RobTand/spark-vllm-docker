"""
Quantize DeltaNet BF16 layers in Sehyo's 122B checkpoint to NVFP4A16.
Leaves Sehyo's already-quantized MoE experts, shared experts, and self_attn intact.
model_free_ptq only touches tensors named '.weight' (BF16), not '.weight_packed' (already FP4).
"""

from llmcompressor import model_free_ptq

MODEL_ID = "Sehyo/Qwen3.5-122B-A10B-NVFP4"
SAVE_DIR = "/root/.cache/huggingface/hub/models--local--Qwen3.5-122B-NVFP4-DeltaNet-Included"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="NVFP4A16",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.conv1d$",
        "re:.*norm.*",
        "re:.*A_log$",
        "re:.*dt_bias$",
    ],
    device="cuda:0",
    max_workers=1,
)

print("Done!")
