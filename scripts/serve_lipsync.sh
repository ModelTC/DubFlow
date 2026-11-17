#!/usr/bin/env bash
set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:?Set LIGHTX2V_PATH to the LightX2V checkout}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-5}
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:$PYTHONPATH}"
cd "${lightx2v_path}"
exec "${PYTHON_BIN:-python}" -m lightx2v.server \
  --model_cls tbdub --model-variant student --task v2v \
  --model_path "${MODEL_PATH:?Set MODEL_PATH to the TBDub checkpoint directory}" \
  --config_json "${CONFIG_PATH:?Set CONFIG_PATH to the TBDub service config}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-18441}" \
  --metric_port "${METRIC_PORT:-18442}" --max_queue_size 8 "$@"
