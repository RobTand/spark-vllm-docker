import json, os, shutil
from pathlib import Path

hub = Path("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots")
if not hub.exists():
    # inside container
    hub = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots")
snap = next(hub.iterdir())
print(f"Source snapshot: {snap}")

with open(snap / "config.json") as f:
    cfg = json.load(f)

for k in ["vision_config", "image_token_id", "video_token_id",
          "vision_start_token_id", "vision_end_token_id"]:
    cfg.pop(k, None)
if "text_config" in cfg:
    text_cfg = cfg.pop("text_config")
    for k, v in text_cfg.items():
        if k not in cfg:
            cfg[k] = v
    if "model_type" in text_cfg:
        cfg["model_type"] = text_cfg["model_type"]
archs = cfg.get("architectures", [])
if archs:
    cfg["architectures"] = [
        a.replace("ForConditionalGeneration", "ForCausalLM") for a in archs
    ]

staged = Path("/tmp/qwen3.6_textonly_stage")
if staged.exists():
    shutil.rmtree(staged)
staged.mkdir()

skip = {
    "config.json",
    # Multimodal files trigger is_mllm_model() in AutoRound. We strip them
    # so AutoScheme treats the checkpoint as plain text-only.
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
}
for p in snap.iterdir():
    if p.name in skip:
        continue
    (staged / p.name).symlink_to(p.resolve())

with open(staged / "config.json", "w") as f:
    json.dump(cfg, f, indent=2)

print(f"Staged to: {staged}")
print(f"  model_type: {cfg.get('model_type')}")
print(f"  architectures: {cfg.get('architectures')}")
print(f"  has vision_config: {'vision_config' in cfg}")
print(f"  num_hidden_layers: {cfg.get('num_hidden_layers')}")
print(f"  num_experts: {cfg.get('num_experts') or cfg.get('num_local_experts')}")
