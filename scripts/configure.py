"""Create local paths from the portable config template."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(models, python, *, translation_backend="lightllm", translation_url=None,
              translation_model=None, translation_api_key_env="LIGHTDUBBER_LLM_API_KEY", translation_options=None,
              tts_backend="qwen3tts", tts_python=None, tts_url=None, tts_model=None, tts_voice=None):
    config = json.loads((ROOT / "config.example.json").read_text())
    models = Path(models).resolve()
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("models/"):
            config[key] = str(models / value.removeprefix("models/"))
    config["models_dir"] = str(models)
    config["python"] = str(Path(python).absolute())
    if translation_backend == "openai_compatible" and (not translation_url or not translation_model):
        raise ValueError("API translation requires --translation-url and --translation-model")
    config["translation_backend"] = translation_backend
    config["translation_api_key_env"] = translation_api_key_env
    if translation_options is not None and not isinstance(translation_options, dict):
        raise ValueError("--translation-options must be a JSON object")
    config["translation_options"] = translation_options or {}
    if translation_url is not None:
        config["translation_url"] = translation_url
    if translation_model is not None:
        config["translation_model"] = translation_model
    config["tts_backend"] = tts_backend
    if tts_backend == "openai_compatible":
        if not tts_url or not tts_model:
            raise ValueError("API TTS requires --tts-url and --tts-model")
        config["tts_url"] = tts_url
        config["tts_model"] = tts_model
        tts_python = tts_python or python
    else:
        if tts_model:
            config["tts_model"] = str(Path(tts_model).resolve())
        tts_python = tts_python or ROOT / ".venv-tts/bin/python"
    config["tts_python"] = str(Path(tts_python).absolute())
    if tts_voice is not None:
        config["tts_voice"] = tts_voice
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--translation-backend", choices=("lightllm", "vllm", "openai_compatible"),
                        default="lightllm")
    parser.add_argument("--translation-url", help="OpenAI-compatible translation API base URL")
    parser.add_argument("--translation-model", help="Served translation model identifier")
    parser.add_argument("--translation-api-key-env", default="LIGHTDUBBER_LLM_API_KEY",
                        help="Name of the environment variable containing the translation API key")
    parser.add_argument("--translation-options", type=json.loads,
                        help="JSON object with provider-specific chat completion options")
    parser.add_argument("--tts-backend", choices=("qwen3tts", "openai_compatible"), default="qwen3tts")
    parser.add_argument("--tts-python", help="Override the TTS interpreter selected for the backend")
    parser.add_argument("--tts-url", help="OpenAI-compatible TTS API base URL")
    parser.add_argument("--tts-model", help="Local model directory or API model identifier")
    parser.add_argument("--tts-voice", help="Voice identifier for the TTS API")
    parser.add_argument("--output", type=Path, default=ROOT / "config.local.json")
    parser.add_argument("--force", action="store_true", help="Replace an existing local configuration")
    args = parser.parse_args()
    try:
        config = configure(args.models, args.python,
                           translation_backend=args.translation_backend, translation_url=args.translation_url,
                           translation_model=args.translation_model, translation_api_key_env=args.translation_api_key_env,
                           translation_options=args.translation_options,
                           tts_backend=args.tts_backend, tts_python=args.tts_python,
                           tts_url=args.tts_url, tts_model=args.tts_model, tts_voice=args.tts_voice)
    except ValueError as error:
        parser.error(str(error))
    with args.output.open("w" if args.force else "x") as handle:
        handle.write(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
