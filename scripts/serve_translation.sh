#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${GPU_ID:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
exec "${VLLM_BIN:-vllm}" serve \
  "${MODEL_PATH:?Set MODEL_PATH to the Qwen3.5-9B directory}" \
  --served-model-name "${MODEL_NAME:-qwen35-dubbing}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-18440}" \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --language-model-only --max-model-len 8192 --max-num-seqs 8 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}" \
  --default-chat-template-kwargs '{"enable_thinking":false}' "$@"
