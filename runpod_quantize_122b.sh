#!/bin/bash
# Run on RunPod 4x H100 box. Quantizes Qwen3.5-122B-A10B to NVFP4 with DeltaNet included.
# Total time: ~45 minutes including download.

set -e

# === CONFIG ===
HF_TOKEN="${HF_TOKEN:-}"  # Set this before running
HF_REPO="rdtand/Qwen3.5-122B-A10B-NVFP4-DeltaNet-Included"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Set HF_TOKEN env var first"
    exit 1
fi

# === INSTALL DEPS ===
echo "Installing deps..."
# Install no-deps packages first to avoid conflicts
pip install --quiet --no-deps \
    'llmcompressor @ git+https://github.com/RobTand/llm-compressor.git@fix/qwen35-mixed-attention-fused-names' \
    'compressed-tensors @ git+https://github.com/vllm-project/compressed-tensors.git' \
    'transformers @ git+https://github.com/huggingface/transformers.git'
# Then the rest
pip install --quiet accelerate datasets huggingface_hub safetensors loguru pydantic

# === QUANTIZE ===
cat > /tmp/quantize.py <<'PYEOF'
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3.5-122B-A10B"
SAVE_DIR = "/workspace/Qwen3.5-122B-A10B-NVFP4-DeltaNet-Included"

print(f"Loading {MODEL_ID} across all GPUs...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",  # Spreads across all 4 GPUs
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
        "re:.*linear_attn.conv1d$",  # 3D conv
        # in_proj_a/b are N=64 on 122B, quantizable
        "re:.*norm.*",
        "re:.*A_log$",
        "re:.*dt_bias$",
    ],
)

print("Loading calibration data...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:256]")

def preprocess(ex):
    return {"text": processor.apply_chat_template(ex["messages"], tokenize=False)}

ds = ds.map(preprocess)

def tokenize(sample):
    result = processor(text=sample["text"], padding=False, max_length=4096, truncation=True)
    # Unwrap processor batch dim — critical fix
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

print(f"Saving to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR, safe_serialization=True)
processor.save_pretrained(SAVE_DIR)

try:
    from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
    save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
    print("MTP tensors saved.")
except Exception as e:
    print(f"Warning: Could not save MTP tensors: {e}")

print("Quantization done!")
PYEOF

HF_TOKEN="$HF_TOKEN" python3 /tmp/quantize.py

# === FIX TENSOR NAMES (triple-nested language_model bug) ===
echo "Fixing tensor names..."
cat > /tmp/fix_names.py <<'PYEOF'
import json, os, glob
from safetensors import safe_open
from safetensors.torch import save_file
import torch

CHECKPOINT = "/workspace/Qwen3.5-122B-A10B-NVFP4-DeltaNet-Included"
BAD = "model.language_model.language_model.language_model."
GOOD = "model.language_model."

for shard_path in sorted(glob.glob(os.path.join(CHECKPOINT, "*.safetensors"))):
    print(f"  {os.path.basename(shard_path)}")
    tensors = {}
    renamed = 0
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k.startswith(BAD):
                tensors[GOOD + k[len(BAD):]] = t
                renamed += 1
            else:
                tensors[k] = t
    if renamed > 0:
        save_file(tensors, shard_path)

# Fix index
index_path = os.path.join(CHECKPOINT, "model.safetensors.index.json")
if os.path.exists(index_path):
    with open(index_path) as f:
        index = json.load(f)
    new_map = {(GOOD + k[len(BAD):] if k.startswith(BAD) else k): v
               for k, v in index["weight_map"].items()}
    index["weight_map"] = new_map
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

print("Tensor names fixed.")
PYEOF
python3 /tmp/fix_names.py

# === UPLOAD TO HF ===
echo "Uploading to HF..."
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.create_repo('$HF_REPO', repo_type='model', exist_ok=True)
api.upload_folder(
    folder_path='/workspace/Qwen3.5-122B-A10B-NVFP4-DeltaNet-Included',
    repo_id='$HF_REPO',
    token='$HF_TOKEN',
)
print('Upload complete')
"

echo "DONE! Model at https://huggingface.co/$HF_REPO"
