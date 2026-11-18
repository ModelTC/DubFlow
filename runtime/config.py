"""Backend selection and path normalization at the application boundary."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.example.json"

CURRENT_ROUTE = {"asr_backend": "qwen3", "asr_device": "cuda", "alignment": "native",
                 "diarization_backends": ["nemotron3"], "separation_backend": "sam-large-tv",
                 "require_complete_dialogue": True, "fusion_mode": "speaker_visual"}
PYTHON_KEYS = ("python", "video_python", "audio_python", "speech_python", "vision_python", "identity_python",
               "separation_python", "tts_python")
MODEL_PATH_KEYS = ("models_dir", "s3fd_model", "loconet_model", "grounding_model",
                   "sam_audio_model", "sam_audio_text_encoder", "asr_model", "alignment_model",
                   "diar_model", "identity_model", "landmarker_model")
BACKEND_DEFAULTS = {"translation_backend": "lightllm", "translation_api_key_env": "LIGHTDUBBER_LLM_API_KEY",
                    "tts_backend": "qwen3tts", "tts_url": "", "tts_api_key_env": "LIGHTDUBBER_TTS_API_KEY",
                    "tts_voice": "alloy"}


def normalize_config(config):
    if any(key in config for key in ("translation_api_key", "tts_api_key")):
        raise ValueError("Store API keys in environment variables; configure translation_api_key_env or tts_api_key_env")
    config = {**BACKEND_DEFAULTS, **config, **CURRENT_ROUTE, "profile": "modern"}
    config.pop("s3fd_repo", None)
    for key in PYTHON_KEYS:
        config.setdefault(key, config["python"])
        config[key] = str((ROOT / config[key]).absolute())
    if config["translation_backend"] not in ("lightllm", "vllm", "openai_compatible"):
        raise ValueError(f"Unsupported translation backend: {config['translation_backend']}")
    if config["tts_backend"] not in ("qwen3tts", "openai_compatible"):
        raise ValueError(f"Unsupported TTS backend: {config['tts_backend']}")
    if config["tts_backend"] == "openai_compatible" and not config["tts_url"]:
        raise ValueError("tts_url is required for the OpenAI-compatible TTS backend")
    path_keys = (*MODEL_PATH_KEYS, "tts_model") if config["tts_backend"] == "qwen3tts" else MODEL_PATH_KEYS
    for key in path_keys:
        if config.get(key):
            config[key] = str((ROOT / config[key]).resolve())
    return config


def load_config(path):
    return normalize_config(json.loads(Path(path).read_text()))
