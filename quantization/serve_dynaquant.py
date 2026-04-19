#!/usr/bin/env python3
"""
serve_dynaquant.py — Lightweight OpenAI-compatible server for DynaQuant models.

Loads a DynaQuant full-spectrum model (arbitrary bit widths) and serves it
via an OpenAI-compatible chat completions API. Uses transformers for text
generation with DynaQuantLinear fused Triton kernels.

Usage:
    python3 serve_dynaquant.py \
        --model /models/Qwen3.5-27B-bf16 \
        --dynaquant /tmp/dynaquant-27b-full \
        --host 192.168.1.180 --port 8000
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "kernels"))

app = FastAPI(title="DynaQuant Server")

# Global model state
model = None
tokenizer = None
model_name = "dynaquant"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "dynaquant"
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    stream: bool = False


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "dynaquant",
        }]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    global model, tokenizer

    # Build prompt using chat template
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    t0 = time.time()
    # Ensure proper stop tokens for Qwen3.5
    eos_ids = [tokenizer.eos_token_id]
    # Also stop at </think> boundary if needed
    think_end = tokenizer.encode("</think>", add_special_tokens=False)
    if think_end:
        eos_ids.extend(think_end)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature if request.temperature > 0 else None,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.temperature > 0,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    gen_time = time.time() - t0

    # Parse reasoning vs content (Qwen3 style)
    # The model outputs <think>...</think> then the answer
    reasoning = None
    content = response_text
    if "<think>" in response_text:
        parts = response_text.split("</think>", 1)
        if len(parts) == 2:
            reasoning = parts[0].replace("<think>", "").strip()
            content = parts[1].strip()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "reasoning": reasoning,
            },
            "finish_reason": "stop" if len(new_tokens) < request.max_tokens else "length",
        }],
        "usage": {
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": len(new_tokens),
            "total_tokens": inputs.input_ids.shape[1] + len(new_tokens),
        },
        "timing": {
            "generation_seconds": round(gen_time, 2),
            "tokens_per_second": round(len(new_tokens) / gen_time, 1),
        },
    }


def main():
    global model, tokenizer, model_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base HF model path")
    parser.add_argument("--dynaquant", required=True, help="DynaQuant packed weights dir")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from build_rtn_cache import stage_multimodal
    from dynaquant_vllm import load_dynaquant_weights
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    staged, cleanup = stage_multimodal(args.model)
    print(f"[server] Loading model from {staged}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        staged, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)

    # Inject DynaQuant weights
    print(f"[server] Injecting DynaQuant weights from {args.dynaquant}...", flush=True)
    model = load_dynaquant_weights(model, args.dynaquant, device="cuda")
    model.eval()

    # Read recipe stats
    with open(Path(args.dynaquant) / "dynaquant_config.json") as f:
        dq_config = json.load(f)
    stats = dq_config["stats"]
    model_name = f"dynaquant-{Path(args.model).name}"

    print(f"[server] Model ready: {stats['n_quantized']} quantized layers", flush=True)
    print(f"[server] Bits: {stats['bits_histogram']}", flush=True)
    print(f"[server] Serving on http://{args.host}:{args.port}", flush=True)

    if cleanup:
        import shutil
        shutil.rmtree(cleanup, ignore_errors=True)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
