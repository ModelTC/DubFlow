"""Download pinned Qwen ASR, forced alignment and Nemotron weights."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = [
    ("Qwen/Qwen3-ASR-1.7B-hf", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3", "Qwen3-ASR-1.7B-hf", ["*.json", "*.jinja", "*.safetensors", "README.md"]),
    ("Qwen/Qwen3-ForcedAligner-0.6B-hf", "c07281df297b9905d24a508279258cccf987a064", "Qwen3-ForcedAligner-0.6B-hf", ["*.json", "*.jinja", "*.safetensors", "README.md"]),
    ("nvidia/Nemotron-3-Diarization", "f667ed73aee57d40cc39428eb768b4fd87a0a29e", "Nemotron-3-Diarization", ["config.json", "processor_config.json", "model.safetensors", "README.md"]),
]


def download(root, spec):
    repo, revision, directory, patterns = spec
    print(f"Downloading {repo} -> {root / directory}", flush=True)
    snapshot_download(repo, revision=revision, local_dir=root / directory,
                      allow_patterns=patterns, ignore_patterns=["train/*", "imgs/*", "qwen0.6bemo4-merge/*"],
                      max_workers=2)
    print(f"Ready: {repo} @ {revision}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda spec: download(args.model_dir, spec), MODELS))
