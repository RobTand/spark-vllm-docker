"""
Quantize Qwen3.5-122B-A10B to full NVFP4 (W4A4) with calibration.
Run on RunPod H200 with 188GB RAM + 141GB VRAM.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3.5-122B-A10B"

print(f"Loading model {MODEL_ID}...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
    offload_folder="/tmp/offload",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
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
)

print("Loading calibration data...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:256]")

def preprocess(example):
    return {"text": processor.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    result = processor(text=sample["text"], padding=False, max_length=4096, truncation=True)
    return {k: v[0] if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list)
            else v for k, v in result.items()}
ds = ds.map(tokenize, remove_columns=ds.column_names)

print("Running calibrated quantization...")
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=4096,
    num_calibration_samples=256,
)

SAVE_DIR = "/workspace/Qwen3.5-122B-A10B-NVFP4-DeltaNet-Included"
print(f"Saving to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR, safe_serialization=True)
processor.save_pretrained(SAVE_DIR)

try:
    from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
    save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
    print("MTP tensors saved.")
except Exception as e:
    print(f"Warning: Could not save MTP tensors: {e}")

print("Done!")
