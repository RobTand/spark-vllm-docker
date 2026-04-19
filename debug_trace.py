"""Debug: trace Qwen3.5-27B and dump generated subgraph code to find the extra dimension."""

import torch
import sys

# Patch Subgraph.forward to print shapes and generated code
import llmcompressor.pipelines.sequential.helpers as helpers
_orig_forward = helpers.Subgraph.forward

def _debug_forward(self, *args, **kwargs):
    if self._code is None:
        self._code = self.graph.python_code("self")
        exec(self._code.src, self._code.globals)
        # Dump the first 3 subgraphs' generated code
        if not hasattr(helpers, '_dump_count'):
            helpers._dump_count = 0
        if helpers._dump_count < 3:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"SUBGRAPH {helpers._dump_count} GENERATED CODE:", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            # Print with line numbers
            for i, line in enumerate(self._code.src.split('\n'), 1):
                print(f"  {i:3d}  {line}", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
            helpers._dump_count += 1

    # Print input shapes
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f"  INPUT {k}: shape={v.shape}", file=sys.stderr, flush=True)

    result = _orig_forward(self, *args, **kwargs)

    # Print output shapes
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                print(f"  OUTPUT {k}: shape={v.shape}", file=sys.stderr, flush=True)

    return result

helpers.Subgraph.forward = _debug_forward

# Now run oneshot with minimal calibration
from transformers import AutoModelForImageTextToText, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen3.5-27B"

print("Loading model...", file=sys.stderr)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head", "re:visual.*", "re:model.visual.*",
        "re:.*mlp.gate$", "re:.*embed_tokens$", "re:.*shared_expert_gate$",
        "re:.*linear_attn.conv1d$", "re:.*linear_attn.in_proj_a$",
        "re:.*linear_attn.in_proj_b$",
    ],
)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:4]")

def preprocess(example):
    return {"text": processor.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return processor(text=sample["text"], padding=False, max_length=512, truncation=True)
ds = ds.map(tokenize, remove_columns=ds.column_names)

print("Running oneshot (4 samples, short seq)...", file=sys.stderr)
try:
    oneshot(
        model=model,
        recipe=recipe,
        dataset=ds,
        max_seq_length=512,
        num_calibration_samples=4,
    )
except Exception as e:
    print(f"\nFailed with: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)

print("Done.", file=sys.stderr)
