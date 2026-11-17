#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_JSON=${DUB_CONFIG:-$ROOT/config.local.json}
TRANSLATION_BACKEND=${TRANSLATION_BACKEND:-$(python3 - "$CONFIG_JSON" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text()) if config_path.is_file() else {}
print(config.get("translation_backend", "lightllm"))
PY
)}

export CUDA_VISIBLE_DEVICES=${GPU_ID:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the Qwen3.5-9B directory}
MODEL_NAME=${MODEL_NAME:-qwen35-dubbing}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18440}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}

case "$TRANSLATION_BACKEND" in
  lightllm)
    exec "${LIGHTLLM_PYTHON:-python3}" -m lightllm.server.api_server \
      --model_dir "$MODEL_PATH" --model_name "$MODEL_NAME" \
      --host "$HOST" --port "$PORT" --tp 1 --data_type bfloat16 \
      --disable_vision --disable_audio \
      --max_req_total_len 8192 --running_max_req_size 8 \
      --graph_max_batch_size 8 --graph_split_batch_size 8 \
      --mem_fraction "$GPU_MEMORY_UTILIZATION" "$@"
    ;;
  vllm)
    exec "${VLLM_BIN:-vllm}" serve "$MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --host "$HOST" --port "$PORT" \
      --tensor-parallel-size 1 --dtype bfloat16 \
      --language-model-only --max-model-len 8192 --max-num-seqs 8 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --default-chat-template-kwargs '{"enable_thinking":false}' "$@"
    ;;
  *)
    echo "Unsupported translation backend: $TRANSLATION_BACKEND (choose lightllm or vllm)" >&2
    exit 2
    ;;
esac
