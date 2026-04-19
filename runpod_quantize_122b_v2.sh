#!/bin/bash
# Self-contained script to quantize Qwen3.5-122B-A10B with DeltaNet on a fresh GPU box.
# Tested on 4x H100 80GB; should work on 2x H200 141GB or similar configs with >=240GB total VRAM.
#
# Usage:
#   HF_TOKEN=hf_... bash runpod_quantize_122b_v2.sh
#
# Total time: ~3-4 hours on 2x H200, ~4-5 hours on 4x H100.

set -e

# === CONFIG ===
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="rdtand/Qwen3.5-122B-A10B-NVFP4-DeltaNet-Included"
NUM_CALIBRATION_SAMPLES="${NUM_CALIBRATION_SAMPLES:-256}"  # Override to 32 for fast testing
MODEL_ID="Qwen/Qwen3.5-122B-A10B"
SAVE_DIR="/workspace/Qwen3.5-122B-A10B-NVFP4-DeltaNet-Included"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Set HF_TOKEN env var first"
    exit 1
fi

export HF_TOKEN
export HF_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# === INSTALL TORCH (cu128 for driver compatibility) ===
echo "=== Installing torch+cu128 ==="
pip uninstall -y --break-system-packages torch torchvision 2>&1 | tail -1 || true
pip install --quiet --break-system-packages torch==2.10.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'torch {torch.__version__}, cuda {torch.version.cuda}, gpus {torch.cuda.device_count()}')"

# === INSTALL DEPS ===
echo "=== Installing llm-compressor + deps ==="
pip install --quiet --break-system-packages --no-deps \
    'llmcompressor @ git+https://github.com/RobTand/llm-compressor.git@fix/qwen35-mixed-attention-fused-names' \
    'compressed-tensors @ git+https://github.com/vllm-project/compressed-tensors.git' \
    'transformers @ git+https://github.com/huggingface/transformers.git' 2>&1 | tail -1

pip install --quiet --break-system-packages \
    accelerate datasets huggingface_hub safetensors loguru pydantic regex \
    tokenizers sentencepiece protobuf einops numpy pillow 2>&1 | tail -1

python -c "from llmcompressor import oneshot; from transformers import AutoModelForImageTextToText; print('deps OK')"

# === PATCH llmcompressor MoE unfuse to avoid GPU OOM ===
echo "=== Patching qwen3_5_moe.py (CPU unfuse) ==="
python << 'PYEOF'
import glob
candidates = glob.glob("/usr/local/lib/python*/dist-packages/llmcompressor/modeling/qwen3_5_moe.py")
if not candidates:
    candidates = glob.glob("/usr/lib/python*/dist-packages/llmcompressor/modeling/qwen3_5_moe.py")
path = candidates[0]
with open(path) as f:
    c = f.read()

orig = """        for i in range(self.num_experts):
            gate_up = gate_up_data[i]  # [2*intermediate, hidden]
            down = down_data[i]  # [hidden, intermediate]

            # gate_up_proj stores [gate; up] stacked along dim 0
            # nn.Linear weight is [out_features, in_features]
            self[i].gate_proj.weight.data = (
                gate_up[:intermediate_size, :].clone().contiguous()
            )
            self[i].up_proj.weight.data = (
                gate_up[intermediate_size:, :].clone().contiguous()
            )
            self[i].down_proj.weight.data = down.clone().contiguous()"""

# Old buggy version (left weights on CPU after unfuse)
buggy = """        # Unfuse on CPU to avoid GPU OOM during clone
        import torch
        gate_up_cpu = gate_up_data.cpu()
        down_cpu = down_data.cpu()
        original.gate_up_proj.data = torch.empty(0, device=gate_up_data.device)
        original.down_proj.data = torch.empty(0, device=down_data.device)
        del gate_up_data, down_data
        torch.cuda.empty_cache()

        for i in range(self.num_experts):
            gate_up = gate_up_cpu[i]
            down = down_cpu[i]
            self[i].gate_proj.weight.data = gate_up[:intermediate_size, :].clone().contiguous()
            self[i].up_proj.weight.data = gate_up[intermediate_size:, :].clone().contiguous()
            self[i].down_proj.weight.data = down.clone().contiguous()"""

# Fixed: clone per-expert directly on GPU. Each clone is small (~11.5MB), peak ~2x MoE size fits.
new = """        # Unfuse on GPU per-expert (slices are small, peak ~2x MoE size fits in free VRAM)
        import torch
        device = gate_up_data.device
        for i in range(self.num_experts):
            gate_up = gate_up_data[i]
            down = down_data[i]
            self[i].gate_proj.weight.data = gate_up[:intermediate_size, :].clone().contiguous()
            self[i].up_proj.weight.data = gate_up[intermediate_size:, :].clone().contiguous()
            self[i].down_proj.weight.data = down.clone().contiguous()
        original.gate_up_proj.data = torch.empty(0, device=device)
        original.down_proj.data = torch.empty(0, device=device)
        del gate_up_data, down_data
        torch.cuda.empty_cache()"""

# Previous attempt: CPU staging with .to(device) — works but slow CPU clones during MoE-replace
prev_cpu_stage = """        # Unfuse via CPU staging to avoid GPU OOM, then move results back to GPU
        import torch
        device = gate_up_data.device
        gate_up_cpu = gate_up_data.cpu()
        down_cpu = down_data.cpu()
        original.gate_up_proj.data = torch.empty(0, device=device)
        original.down_proj.data = torch.empty(0, device=device)
        del gate_up_data, down_data
        torch.cuda.empty_cache()

        for i in range(self.num_experts):
            gate_up = gate_up_cpu[i]
            down = down_cpu[i]
            self[i].gate_proj.weight.data = gate_up[:intermediate_size, :].clone().contiguous().to(device)
            self[i].up_proj.weight.data = gate_up[intermediate_size:, :].clone().contiguous().to(device)
            self[i].down_proj.weight.data = down.clone().contiguous().to(device)"""

if orig in c:
    c = c.replace(orig, new)
    with open(path, "w") as f:
        f.write(c)
    print(f"Patched (from original) {path}")
elif buggy in c:
    c = c.replace(buggy, new)
    with open(path, "w") as f:
        f.write(c)
    print(f"Patched (replaced buggy version) {path}")
elif prev_cpu_stage in c:
    c = c.replace(prev_cpu_stage, new)
    with open(path, "w") as f:
        f.write(c)
    print(f"Patched (replaced CPU-stage version) {path}")
elif new in c:
    print(f"Already fixed {path}")
else:
    print(f"WARNING: pattern not found in {path}")
PYEOF

# === DOWNLOAD MODEL ===
echo "=== Downloading $MODEL_ID ==="
python << 'PYEOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3.5-122B-A10B",
    token=os.environ["HF_TOKEN"],
    max_workers=16,
)
print("Download complete")
PYEOF

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
NUM_SAMPLES = int(os.environ.get("NUM_CALIBRATION_SAMPLES", "256"))

print(f"Loading {MODEL_ID} across all GPUs...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
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
        # in_proj_a/b are N=64 on 122B — quantizable
        "re:.*norm.*",
        "re:.*A_log$",
        "re:.*dt_bias$",
    ],
)

print(f"Loading {NUM_SAMPLES} calibration samples...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_SAMPLES}]")

def preprocess(ex):
    return {"text": processor.apply_chat_template(ex["messages"], tokenize=False)}

ds = ds.map(preprocess)

def tokenize(sample):
    result = processor(text=sample["text"], padding=False, max_length=4096, truncation=True)
    # Critical: unwrap processor's batch dim to avoid 4D tensors during calibration
    return {k: v[0] if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list)
            else v for k, v in result.items()}

ds = ds.map(tokenize, remove_columns=ds.column_names)

print(f"Running calibrated quantization with {NUM_SAMPLES} samples...")
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=4096,
    num_calibration_samples=NUM_SAMPLES,
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

NUM_CALIBRATION_SAMPLES="$NUM_CALIBRATION_SAMPLES" python /tmp/quantize.py

# === FIX TENSOR NAMES (triple-nested language_model bug) ===
echo "=== Fixing tensor names ==="
python << PYEOF
import json, os, glob
from safetensors import safe_open
from safetensors.torch import save_file
import torch

CHECKPOINT = "$SAVE_DIR"
BAD = "model.language_model.language_model.language_model."
GOOD = "model.language_model."

renamed_total = 0
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
        renamed_total += renamed

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

print(f"Renamed {renamed_total} tensors.")
PYEOF

# === UPLOAD TO HF ===
echo "=== Uploading to HF ($HF_REPO) ==="
python << PYEOF
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("$HF_REPO", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="$SAVE_DIR",
    repo_id="$HF_REPO",
    token=os.environ["HF_TOKEN"],
)
print("Upload complete")
PYEOF

echo ""
echo "================================================================"
echo "DONE!"
echo "Model: https://huggingface.co/$HF_REPO"
echo "Local: $SAVE_DIR"
echo "================================================================"
