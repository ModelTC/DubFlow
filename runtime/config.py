"""Configuration for the single supported dubbing route."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


CURRENT_ROUTE = {"asr_backend": "qwen3", "asr_device": "cuda", "alignment": "native",
                 "diarization_backends": ["nemotron3"], "separation_backend": "sam-large-tv",
                 "translation_backend": "vllm", "tts_backend": "qwen3tts",
                 "require_complete_dialogue": True,
                 "fusion_mode": "speaker_visual"}
PYTHON_KEYS = ("video_python", "audio_python", "speech_python", "vision_python", "identity_python",
               "separation_python", "tts_python")


def load_config(path):
    config = json.loads(Path(path).read_text())
    for key in PYTHON_KEYS:
        config.setdefault(key, config["python"])
    return {**config, **CURRENT_ROUTE, "profile": "modern"}
