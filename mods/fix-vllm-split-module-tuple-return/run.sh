#!/bin/bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/compilation/backends.py")
text = path.read_text()

old = """    with _use_lazy_graph_module(True):\n        has_tuple_return = is_torch_equal_or_newer(\"2.12.0.dev\")\n        tuple_return_kwarg = {\"tuple_return\": True} if has_tuple_return else {}\n        split_gm = torch.fx.passes.split_module.split_module(\n            graph,\n            None,\n            lambda node: node_to_subgraph_id[node],\n            keep_original_order=True,\n            **tuple_return_kwarg,\n        )\n"""

new = """    with _use_lazy_graph_module(True):\n        import inspect\n\n        split_module_fn = torch.fx.passes.split_module.split_module\n        has_tuple_return = \"tuple_return\" in inspect.signature(split_module_fn).parameters\n        tuple_return_kwarg = {\"tuple_return\": True} if has_tuple_return else {}\n        split_gm = split_module_fn(\n            graph,\n            None,\n            lambda node: node_to_subgraph_id[node],\n            keep_original_order=True,\n            **tuple_return_kwarg,\n        )\n"""

if old not in text:
    raise SystemExit("expected split_graph block not found")

path.write_text(text.replace(old, new, 1))
print(f"Patched {path}")
PY

echo "Applied vLLM split_module tuple_return compatibility fix"
