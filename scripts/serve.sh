#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export DUB_CONFIG=${DUB_CONFIG:-$PROJECT_ROOT/config.local.json}
export DUB_DATA_DIR=${DUB_DATA_DIR:-$PROJECT_ROOT/outputs/web}
export DUB_GPUS=${DUB_GPUS:-6,7}

cd -- "$PROJECT_ROOT"
PYTHON_BIN=${PYTHON_BIN:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$DUB_CONFIG")}
exec "$PYTHON_BIN" -m uvicorn web.server:app \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-18320}" \
  --workers 1 --timeout-keep-alive 15 --no-access-log \
  --proxy-headers --forwarded-allow-ips 127.0.0.1
