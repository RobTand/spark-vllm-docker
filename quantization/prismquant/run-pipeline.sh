#!/usr/bin/env bash
# run-pipeline.sh — end-to-end PrismQuant pipeline: probe → cost →
# allocator → native compressed-tensors export → vLLM validate.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.6-35B-A3B \
#   WORK_DIR=./dq-runs/qwen36 \
#   FORMATS=NVFP4,MXFP8_E4M3,BF16 \
#   TARGET_BITS=4.75 \
#   ./quantization/prismquant/run-pipeline.sh
#
# Memory note: probe + cost peak around 90 GB on a 35B model under
# BF16 calibration. The watchdog in incremental_measure_quant_cost
# aborts cleanly on swap pressure rather than OOM-killing the host.

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the source HF model directory}"
: "${WORK_DIR:?Set WORK_DIR to a writable directory for artifacts}"
: "${FORMATS:=NVFP4,MXFP8_E4M3,BF16}"
: "${TARGET_BITS:=4.75}"
: "${PARETO_TARGETS:=4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25}"
: "${NSAMPLES:=4}"
: "${SEQLEN:=256}"
: "${LAYERS_PER_SHARD:=2}"
: "${DATASET:=ultrachat_200k}"
: "${DEVICE:=cuda}"
: "${EXPORT_DEVICE:=cpu}"   # cpu is safer for streaming; cuda is faster
: "${TARGET_PROFILE:=vllm_qwen3_5_packed_moe}"
: "${INCLUDE_MTP:=1}"

BODY_PROBE_PATH="${WORK_DIR}/artifacts/probe_body.pkl"
BODY_COST_PATH="${WORK_DIR}/artifacts/cost_body.pkl"
MTP_PROBE_PATH="${WORK_DIR}/artifacts/mtp_probe.pkl"
MTP_COST_PATH="${WORK_DIR}/artifacts/mtp_cost.pkl"
PROBE_PATH="${WORK_DIR}/artifacts/probe.pkl"
COST_PATH="${WORK_DIR}/artifacts/cost.pkl"

mkdir -p "${WORK_DIR}"/{artifacts,act,work,logs,exported}

echo "[pipeline] config:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  WORK_DIR=$WORK_DIR"
echo "  FORMATS=$FORMATS  TARGET_BITS=$TARGET_BITS"
echo "  NSAMPLES=$NSAMPLES SEQLEN=$SEQLEN LAYERS_PER_SHARD=$LAYERS_PER_SHARD"
echo

# -----------------------------------------------------------------------
# 1. Sensitivity probe (per-Linear empirical Fisher diagonal trace)
# -----------------------------------------------------------------------
if [[ ! -f "${BODY_PROBE_PATH}" ]]; then
  echo "[pipeline] [1/4] running sensitivity probe ..."
  python3 -m quantization.prismquant.incremental_probe \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --nsamples "$NSAMPLES" --seqlen "$SEQLEN" \
    --device "$DEVICE" --dtype bf16 \
    --output "${BODY_PROBE_PATH}" \
    --activation-cache-dir "${WORK_DIR}/act" \
    --work-dir "${WORK_DIR}/work" \
    --layers-per-shard "$LAYERS_PER_SHARD" \
    2>&1 | tee "${WORK_DIR}/logs/probe.log"
else
  echo "[pipeline] [1/4] probe_body.pkl exists, skipping"
fi

# -----------------------------------------------------------------------
# 2. Cost measurement (per-(Linear, format) measured RTN error)
# -----------------------------------------------------------------------
if [[ ! -f "${BODY_COST_PATH}" ]]; then
  echo "[pipeline] [2/4] measuring per-(layer, format) cost ..."
  python3 -m quantization.prismquant.incremental_measure_quant_cost \
    --model "$MODEL_PATH" \
    --probe "${BODY_PROBE_PATH}" \
    --activation-cache-dir "${WORK_DIR}/act" \
    --formats "$FORMATS" \
    --output "${BODY_COST_PATH}" \
    --work-dir "${WORK_DIR}/work" \
    --device "$DEVICE" --dtype bf16 \
    --mode batched --chunk-size 256 \
    --layers-per-shard "$LAYERS_PER_SHARD" \
    --skip-missing-activations \
    2>&1 | tee "${WORK_DIR}/logs/cost.log"
else
  echo "[pipeline] [2/4] cost_body.pkl exists, skipping"
fi

if [[ "${INCLUDE_MTP}" == "1" ]]; then
  if [[ ! -f "${MTP_PROBE_PATH}" ]]; then
    echo "[pipeline] [2a/4] running mtp probe ..."
    python3 -m quantization.prismquant.mtp_probe \
      --model "$MODEL_PATH" \
      --dataset "$DATASET" \
      --nsamples "$NSAMPLES" --seqlen "$SEQLEN" \
      --device "$DEVICE" --dtype bf16 \
      --activation-cache-dir "${WORK_DIR}/act_mtp" \
      --output "${MTP_PROBE_PATH}" \
      2>&1 | tee "${WORK_DIR}/logs/mtp_probe.log"
  else
    echo "[pipeline] [2a/4] mtp_probe.pkl exists, skipping"
  fi

  if [[ ! -f "${MTP_COST_PATH}" ]]; then
    echo "[pipeline] [2b/4] running mtp cost ..."
    python3 -m quantization.prismquant.mtp_cost \
      --model "$MODEL_PATH" \
      --formats "$FORMATS" \
      --activation-cache-dir "${WORK_DIR}/act_mtp" \
      --output "${MTP_COST_PATH}" \
      --device "$DEVICE" --dtype bf16 --mode batched --chunk-size 256 \
      2>&1 | tee "${WORK_DIR}/logs/mtp_cost.log"
  else
    echo "[pipeline] [2b/4] mtp_cost.pkl exists, skipping"
  fi

  echo "[pipeline] [2c/4] merging body + mtp probe/cost ..."
  python3 - <<PY
from pathlib import Path
from quantization.prismquant.incremental_probe import merge_probe_pickles
from quantization.prismquant.incremental_measure_quant_cost import merge_cost_pickles
merge_probe_pickles([Path("${BODY_PROBE_PATH}"), Path("${MTP_PROBE_PATH}")], Path("${PROBE_PATH}"))
merge_cost_pickles([Path("${BODY_COST_PATH}"), Path("${MTP_COST_PATH}")], Path("${COST_PATH}"))
PY
else
  cp "${BODY_PROBE_PATH}" "${PROBE_PATH}"
  cp "${BODY_COST_PATH}" "${COST_PATH}"
fi

# -----------------------------------------------------------------------
# 3. Allocator (multi-choice knapsack over per-layer formats)
# -----------------------------------------------------------------------
echo "[pipeline] [3/4] running allocator at target=${TARGET_BITS} bpp ..."
python3 -m quantization.prismquant.allocator \
  --probe "${PROBE_PATH}" \
  --costs "${COST_PATH}" \
  --target-bits "$TARGET_BITS" \
  --formats "$FORMATS" \
  --target-profile "$TARGET_PROFILE" \
  --pareto-targets "$PARETO_TARGETS" \
  --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
  --pareto-csv "${WORK_DIR}/artifacts/pareto.csv" \
  2>&1 | tee "${WORK_DIR}/logs/allocator.log"

# -----------------------------------------------------------------------
# 4. Native compressed-tensors export
# -----------------------------------------------------------------------
echo "[pipeline] [4/4] exporting to compressed-tensors ..."
python3 -m quantization.prismquant.export_native_compressed \
  --model "$MODEL_PATH" \
  --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
  --output "${WORK_DIR}/exported" \
  --device "$EXPORT_DEVICE" \
  2>&1 | tee "${WORK_DIR}/logs/export.log"

echo
echo "[pipeline] done."
echo "  Artifact: ${WORK_DIR}/exported"
echo "  Validate: python3 -m quantization.prismquant.validate_native_export \\"
echo "              --model ${WORK_DIR}/exported"
echo "  Serve:    vllm serve ${WORK_DIR}/exported \\"
echo "              --quantization compressed-tensors --trust-remote-code"
