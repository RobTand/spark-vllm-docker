"""Debug v2: verify shapes are correct after fixing the tokenize squeeze."""

import torch
import sys

# Minimal test: check what the dataloader produces
from transformers import AutoProcessor
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen3.5-27B"
processor = AutoProcessor.from_pretrained(MODEL_ID)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:2]")

def preprocess(example):
    return {"text": processor.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    result = processor(text=sample["text"], padding=False, max_length=512, truncation=True)
    return {k: v.squeeze(0) if hasattr(v, 'squeeze') else v for k, v in result.items()}

ds = ds.map(tokenize, remove_columns=ds.column_names)

print("Dataset sample shapes:")
sample = ds[0]
for k, v in sample.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: {type(v).__name__} shape={v.shape}")
    elif isinstance(v, list):
        print(f"  {k}: list len={len(v)}")
    else:
        print(f"  {k}: {type(v).__name__}")

# Now test with DataLoader
from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=1)
batch = next(iter(dl))
print("\nDataLoader batch shapes:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")
