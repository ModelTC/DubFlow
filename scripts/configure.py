"""Create local paths from the portable config template."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(models, python):
    config = json.loads((ROOT / "config.example.json").read_text())
    models = Path(models).resolve()
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("models/"):
            config[key] = str(models / value.removeprefix("models/"))
    config["models_dir"] = str(models)
    config["python"] = str(Path(python).absolute())
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tts-python", default=str(ROOT / ".venv-tts/bin/python"))
    parser.add_argument("--output", type=Path, default=ROOT / "config.local.json")
    parser.add_argument("--force", action="store_true", help="Replace an existing local configuration")
    args = parser.parse_args()
    config = configure(args.models, args.python)
    config["tts_python"] = str(Path(args.tts_python).absolute())
    with args.output.open("w" if args.force else "x") as handle:
        handle.write(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
