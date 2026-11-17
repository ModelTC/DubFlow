#!/usr/bin/env bash
# 下载当前管线和两个独立服务的权重。
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$ROOT"
MODEL_DIR=${MODEL_DIR:-$ROOT/models}
CONFIG_JSON=${CONFIG_JSON:-$ROOT/config.local.json}
[[ -f "$CONFIG_JSON" ]] || CONFIG_JSON=$ROOT/config.example.json
PYTHON_BIN=${PYTHON_BIN:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$CONFIG_JSON")}
PYTHON_BIN=$(command -v "$PYTHON_BIN")
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
MODE=${1:-all}

download_file() {
  local url=$1 destination=$2
  if [[ ! -f "$destination" ]]; then
    curl -fL "$url" -o "$destination.part"
    mv "$destination.part" "$destination"
  fi
}

core() {
  mkdir -p "$MODEL_DIR"

  hf download TaoLiveAIGC/TBDub \
    config.json null_prompt_emb.pt tbdub_student.safetensors \
    --revision b1ac43698859d35bca475301834be9d6133472e2 \
    --local-dir "$MODEL_DIR/tbdub"
  hf download KlingTeam/X-Dub Wan2.2_VAE.safetensors \
    --revision c84e9612b7c1160f06ea609882e12efd83cacbbe \
    --local-dir "$MODEL_DIR/tbdub"
  hf download facebook/hubert-large-ll60k \
    config.json preprocessor_config.json pytorch_model.bin \
    --revision ff022d095678a2995f3c49bab18a96a9e553f782 \
    --local-dir "$MODEL_DIR/tbdub/hubert-large-ll60k"
  download_file \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    "$MODEL_DIR/tbdub/face_landmarker.task"

  mkdir -p "$MODEL_DIR/talknet"
  loconet
  if [[ ! -f "$MODEL_DIR/talknet/sfd_face.pth" ]]; then
    gdown 1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt \
      -O "$MODEL_DIR/talknet/sfd_face.pth"
  fi
  mkdir -p "$MODEL_DIR/insightface/buffalo_l"
  download_file https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
    "$MODEL_DIR/insightface/buffalo_l.zip"
  unzip -n -j "$MODEL_DIR/insightface/buffalo_l.zip" det_10g.onnx w600k_r50.onnx \
    -d "$MODEL_DIR/insightface/buffalo_l"

}

speech() {
  mkdir -p "$MODEL_DIR"
  "$PYTHON_BIN" "$ROOT/scripts/download_recognition.py" --model-dir "$MODEL_DIR"
  hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --revision fd4b254389122332181a7c3db7f27e918eec64e3 \
    --local-dir "$MODEL_DIR/Qwen3-TTS-12Hz-1.7B-Base"
}

loconet() {
  mkdir -p "$MODEL_DIR/loconet"
  if [[ ! -f "$MODEL_DIR/loconet/loconet_unitalk.model" ]]; then
    gdown 10HvRqNO34QsJpZgQTMYZdIAW4A2kP5fw -O "$MODEL_DIR/loconet/loconet_unitalk.model.part"
    mv "$MODEL_DIR/loconet/loconet_unitalk.model.part" "$MODEL_DIR/loconet/loconet_unitalk.model"
  fi
}

sam_audio() {
  "$PYTHON_BIN" "$ROOT/scripts/download_sam_audio.py" \
    --model-dir "${SAM_AUDIO_MODEL_DIR:-$MODEL_DIR/SAM-Audio-Large-TV}" \
    --source "${SAM_AUDIO_DOWNLOAD_SOURCE:-modelscope}"
}

qwen35_translation() {
  HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3.5-9B \
    --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --max-workers 4 \
    --local-dir "$MODEL_DIR/Qwen3.5-9B"
}

case "$MODE" in
  core) core ;;
  speech) speech ;;
  loconet) loconet ;;
  sam-audio) sam_audio ;;
  qwen35-translation) qwen35_translation ;;
  all) core; speech; sam_audio; qwen35_translation ;;
  *) echo "用法：bash scripts/download_models.sh [core|speech|loconet|sam-audio|qwen35-translation|all]" >&2; exit 2 ;;
esac
