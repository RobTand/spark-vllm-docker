#!/bin/bash
set -e

# Install DynaQuant kernel files into container's Python path
DEST="/usr/local/lib/python3.12/dist-packages/dynaquant"
mkdir -p "$DEST/kernels"

# Copy from the mod's working directory (set by run-recipe.py)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$SCRIPT_DIR/pack_utils.py" "$DEST/kernels/"
cp "$SCRIPT_DIR/fused_dequant_gemv.py" "$DEST/kernels/"
cp "$SCRIPT_DIR/dynaquant_linear.py" "$DEST/"
cp "$SCRIPT_DIR/dynaquant_inject.py" "$DEST/"
touch "$DEST/__init__.py"
touch "$DEST/kernels/__init__.py"

# Fix imports in dynaquant_linear.py to use the installed package path
sed -i "s|sys.path.insert.*kernels.*||" "$DEST/dynaquant_linear.py"
sed -i "s|from pack_utils|from dynaquant.kernels.pack_utils|" "$DEST/dynaquant_linear.py"
sed -i "s|from fused_dequant_gemv|from dynaquant.kernels.fused_dequant_gemv|" "$DEST/dynaquant_linear.py"

# Fix imports in fused_dequant_gemv.py
sed -i "s|sys.path.insert.*parent.*||" "$DEST/kernels/fused_dequant_gemv.py"
sed -i "s|from pack_utils|from dynaquant.kernels.pack_utils|" "$DEST/kernels/fused_dequant_gemv.py"

# Patch vLLM's gpu_model_runner to call our injection hook after model loading
RUNNER="/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"

if ! grep -q "dynaquant" "$RUNNER" 2>/dev/null; then
    python3 << 'PYEOF'
import sys

runner_path = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"

with open(runner_path) as f:
    content = f.read()

target = "Model loading took"
if target in content:
    idx = content.index(target)
    end_of_line = content.index('\n', idx)

    inject_code = '''
        # DynaQuant: inject packed N-bit weights
        try:
            from dynaquant.dynaquant_inject import maybe_inject_dynaquant
            maybe_inject_dynaquant(self.model)
        except Exception as e:
            import traceback
            logger.warning(f"DynaQuant injection skipped: {e}")
            traceback.print_exc()
'''
    content = content[:end_of_line+1] + inject_code + content[end_of_line+1:]

    with open(runner_path, 'w') as f:
        f.write(content)
    print("Patched gpu_model_runner.py with DynaQuant hook")
else:
    print("WARNING: Could not find model loading log line to patch")
PYEOF
fi

echo "DynaQuant mod installed"
