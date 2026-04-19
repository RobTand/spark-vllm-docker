#!/bin/bash
set -e
# Lower the min_sms threshold for max_autotune_gemm from 68 (RTX 3080 era)
# to 32, allowing GB10 (48 SMs) to use inductor's GEMM autotuning.
UTILS_FILE="/usr/local/lib/python3.12/dist-packages/torch/_inductor/utils.py"
sed -i 's/min_sms = 16 if device.type == "xpu" else 68  # 3080/min_sms = 16 if device.type == "xpu" else 32  # lowered for GB10/' "$UTILS_FILE"
echo "Applied autotune SM threshold fix (68 -> 32)"
