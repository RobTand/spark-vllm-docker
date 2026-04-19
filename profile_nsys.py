"""
nsys-compatible profiling script.
Sends a single request to the vLLM server and profiles the CUDA activity.

Usage inside container:
  nsys profile --trace cuda,nvtx --output /workspace/profile \
    --force-overwrite true --duration 10 \
    python3 /workspace/profile_nsys.py

Then: nsys stats /workspace/profile.nsys-rep
"""

import requests
import time
import json

API = "http://localhost:8000/v1/chat/completions"
MODEL = "Sehyo/Qwen3.5-122B-A10B-NVFP4"

# Short prompt, force short output for clean decode profiling
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Count from 1 to 50."}],
    "max_tokens": 100,
    "temperature": 0.0,
    "stream": False,
}

print("Sending request...")
r = requests.post(API, json=payload, timeout=60)
print(f"Status: {r.status_code}")
data = r.json()
tokens = data["usage"]["completion_tokens"]
print(f"Generated {tokens} tokens")
