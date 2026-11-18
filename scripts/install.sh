#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DUB_ENV=${DUB_ENV:-$ROOT/.venv}
TTS_ENV=${TTS_ENV:-$ROOT/.venv-tts}
LIGHTLLM_ENV=${LIGHTLLM_ENV:-$ROOT/.venv-lightllm}
TTS=local
TRANSLATION=external

usage() {
  cat <<'HELP'
Usage: bash scripts/install.sh [--tts local|api] [--translation lightllm]

Install the main pipeline and local Qwen3-TTS (default).
  --tts api               Skip the local TTS environment; use a speech API.
  --translation lightllm  Also install the local LightLLM translation server.

Requires uv. Environments default to .venv, .venv-tts and .venv-lightllm.
This script does not change config.local.json or download model weights.
HELP
}

while (($#)); do
  case "$1" in
    --tts)
      case "${2:-}" in
        local|api) TTS=$2 ;;
        *) echo '--tts requires local or api' >&2; exit 2 ;;
      esac
      shift 2
      ;;
    --translation)
      if [[ "${2:-}" != lightllm ]]; then
        echo '--translation requires lightllm' >&2
        exit 2
      fi
      TRANSLATION=$2
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is required. Install it with: python3 -m pip install uv' >&2
  exit 1
fi

bash "$ROOT/scripts/install_unified.sh" "$DUB_ENV"

if [[ "$TTS" == local ]]; then
  if [[ ! -x "$TTS_ENV/bin/python" ]]; then
    uv venv --python 3.11 "$TTS_ENV"
  fi
  uv pip install --python "$TTS_ENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0+cu128 torchaudio==2.8.0+cu128
  uv pip install --python "$TTS_ENV/bin/python" -r "$ROOT/requirements-tts.txt"
fi

if [[ "$TRANSLATION" == lightllm ]]; then
  LIGHTLLM_REVISION=1b38e8d4e133874f1af8d0533c668d4fe66a306e
  if [[ ! -x "$LIGHTLLM_ENV/bin/python" ]]; then
    uv venv --python 3.10 --seed "$LIGHTLLM_ENV"
  fi
  uv pip install --python "$LIGHTLLM_ENV/bin/python" \
    -r "https://raw.githubusercontent.com/ModelTC/LightLLM/$LIGHTLLM_REVISION/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  uv pip install --python "$LIGHTLLM_ENV/bin/python" --no-deps --no-build-isolation \
    "https://github.com/ModelTC/LightLLM/archive/$LIGHTLLM_REVISION.tar.gz"
fi

printf '\nInstalled. Create your local configuration with scripts/configure.py.\n'
