"""
Quantize Qwen3.5-122B-A10B to NVFP4A16 using model_free_ptq.
No forward passes — processes safetensors shard-by-shard.

We'll compute input_global_scale on the Spark afterward
to upgrade to full NVFP4 W4A4.
"""

from llmcompressor import model_free_ptq

MODEL_ID = "Qwen/Qwen3.5-122B-A10B"
SAVE_DIR = "/storage_pool/genomics/quantize_122b/output"

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
