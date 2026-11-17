#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DUB_ENV=${1:-${DUB_ENV:-$ROOT/.venv}}

if [[ ! -x "$DUB_ENV/bin/python" ]]; then
  uv venv --python 3.11 "$DUB_ENV"
fi
uv pip install --python "$DUB_ENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128 xformers==0.0.32.post2
uv pip install --python "$DUB_ENV/bin/python" -r "$ROOT/requirements-unified.txt"
uv pip install --python "$DUB_ENV/bin/python" --no-deps \
  descript-audiotools==0.7.2 scenedetect==0.7.2 \
  -e "$ROOT/third_party/SAM-Audio" -e "$ROOT/third_party/perception_models" \
  -e "$ROOT/third_party/dacvae"
"$DUB_ENV/bin/python" "$ROOT/scripts/link_ffmpeg.py"
