"""Fix tensor naming in our calibrated checkpoint to match vLLM's expected format.

Our checkpoint: model.language_model.language_model.language_model.layers.X...
Expected:       model.language_model.layers.X...
"""

import json
import os
import sys
from safetensors import safe_open
from safetensors.torch import save_file
import torch

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "/root/.cache/huggingface/hub/models--local--Qwen3.5-27B-NVFP4-full-calibrated-v2"

BAD_PREFIX = "model.language_model.language_model.language_model."
GOOD_PREFIX = "model.language_model."

import glob
shards = sorted(glob.glob(os.path.join(CHECKPOINT, "*.safetensors")))

for shard_path in shards:
    print(f"Processing {os.path.basename(shard_path)}...")
    tensors = {}
    renamed = 0
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k.startswith(BAD_PREFIX):
                new_k = GOOD_PREFIX + k[len(BAD_PREFIX):]
                tensors[new_k] = t
                renamed += 1
            else:
                tensors[k] = t

    if renamed > 0:
        save_file(tensors, shard_path)
        print(f"  Renamed {renamed} tensors")
    else:
        print(f"  No renames needed")

# Fix index
index_path = os.path.join(CHECKPOINT, "model.safetensors.index.json")
if os.path.exists(index_path):
    with open(index_path) as f:
        index = json.load(f)

    new_map = {}
    for k, v in index["weight_map"].items():
        if k.startswith(BAD_PREFIX):
            new_map[GOOD_PREFIX + k[len(BAD_PREFIX):]] = v
        else:
            new_map[k] = v

    index["weight_map"] = new_map
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print("Fixed index.json")

print("Done!")
