"""
Quantize Qwen3.5-27B to NVFP4 with ALL linear layers including DeltaNet.

Based on Sehyo's recipe but removes the linear_attn exclusion.
Uses llm-compressor's oneshot with calibration data.
"""

import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3.5-27B"

print(f"Loading model {MODEL_ID}...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# NVFP4 recipe — same as Sehyo's but without linear_attn exclusion
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
        # NOTE: linear_attn is NOT excluded — this is the key change
        "re:.*linear_attn.conv1d$",  # conv1d is 3D, can't be NVFP4
    ],
)

# Calibration data — same as Sehyo's recipe
print("Loading calibration data...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:256]")

def preprocess(example):
    return {
        "text": processor.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

def tokenize(sample):
    return processor(
        text=sample["text"],
        padding=False,
        max_length=4096,
        truncation=True,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

print("Running quantization...")
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=4096,
    num_calibration_samples=256,
)

SAVE_DIR = "/models/Qwen3.5-27B-NVFP4-full"
print(f"Saving to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR, safe_serialization=True)
processor.save_pretrained(SAVE_DIR)

# Copy MTP weights if they exist in the base model
try:
    from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
    save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
    print("MTP tensors saved.")
except Exception as e:
    print(f"MTP tensor copy skipped: {e}")

print("Done!")
