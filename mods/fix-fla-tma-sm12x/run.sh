#!/bin/bash
# Fix FLA Hopper/TMA misclassification on SM12x desktop Blackwell
# Backport of vLLM PR #37700
set -e

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FLA_OPS="$SITE_PACKAGES/vllm/model_executor/layers/fla/ops"

# Patch chunk_o.py: expand BKV_LIST and NUM_WARPS, remove check_shared_mem/is_nvidia_hopper gating
sed -i 's/^from .utils import FLA_CHUNK_SIZE, check_shared_mem, is_nvidia_hopper$/from .utils import FLA_CHUNK_SIZE/' "$FLA_OPS/chunk_o.py"
sed -i 's/^BKV_LIST = \[64, 128\] if check_shared_mem() else \[32, 64\]$/BKV_LIST = [32, 64, 128]/' "$FLA_OPS/chunk_o.py"
sed -i 's/^NUM_WARPS = \[2, 4\] if is_nvidia_hopper else \[2, 4, 8\]$/NUM_WARPS = [2, 4, 8]/' "$FLA_OPS/chunk_o.py"

# Patch utils.py: move is_tma_supported after get_all_max_shared_mem and add SMEM threshold
FLA_OPS="$FLA_OPS" python3 << 'PYEOF'
import re, os

path = os.environ["FLA_OPS"] + "/utils.py"
with open(path) as f:
    content = f.read()

# Remove the old is_tma_supported block (before get_all_max_shared_mem)
old_tma = re.compile(
    r'^is_tma_supported = \(\s*\n'
    r'    is_nvidia_hopper\s*\n'
    r'    and os\.getenv\("FLA_USE_TMA", "0"\) == "1"\s*\n'
    r'    and \(\s*\n'
    r'        hasattr\(triton\.language, "_experimental_make_tensor_descriptor"\)\s*\n'
    r'        or hasattr\(triton\.language, "make_tensor_descriptor"\)\s*\n'
    r'    \)\s*\n'
    r'\)\s*\n',
    re.MULTILINE
)
content = old_tma.sub('', content)

# Insert the new is_tma_supported after get_all_max_shared_mem function
new_tma = '''

# TMA code paths require significant shared memory for the Triton autotuner
# to compile. SM12x desktop GPUs (RTX 5090/5080, DGX Spark GB10) have ~101KB
# SMEM per SM, which is insufficient and causes OOM in fla/solve_tril. Gate
# TMA on a 128KB SMEM threshold rather than the architecture.
is_tma_supported = (
    is_nvidia_hopper
    and os.getenv("FLA_USE_TMA", "0") == "1"
    and get_all_max_shared_mem()[0] >= 131072
    and (
        hasattr(triton.language, "_experimental_make_tensor_descriptor")
        or hasattr(triton.language, "make_tensor_descriptor")
    )
)

'''

# Find the end of get_all_max_shared_mem function and insert after it
pattern = re.compile(r'(def get_all_max_shared_mem\(\).*?return \[-1\]\s*\n)', re.DOTALL)
match = pattern.search(content)
if match:
    insert_pos = match.end()
    content = content[:insert_pos] + new_tma + content[insert_pos:]
    with open(path, 'w') as f:
        f.write(content)
    print("Patched utils.py successfully")
else:
    print("WARNING: Could not find get_all_max_shared_mem, skipping utils.py patch")
PYEOF

echo "Applied FLA TMA SM12x fix (PR #37700)"
