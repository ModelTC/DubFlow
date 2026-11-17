"""Download SAM-Audio Large-TV and its T5 encoder; never fetch optional judges or rankers."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

HF_REPO = "facebook/sam-audio-large-tv"
HF_REVISION = "main"
MODELSCOPE_REVISION = "5777e8e25dd83384b13d62feca0de84fb7b4e9eb"
T5_REVISION = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
SHA256 = {"checkpoint.pt": "90e047269238c498c5abe0da6e6ba40859d152111d4a09f582a209027a33b72f",
          "config.json": "b768c3b04e107a7afb3865d9126c6159d4b604b4d3279f99446f4f50b9f2c26d",
          "LICENSE": "4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be"}
CHECKPOINT_SIZE = 14861356211


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def modelscope_file(name, output, workers=16):
    destination = output / name
    if destination.exists():
        if file_hash(destination) != SHA256[name]:
            raise ValueError(f"Existing SAM file has a different checksum: {destination}")
        return
    url = f"https://modelscope.cn/models/{HF_REPO}/resolve/{MODELSCOPE_REVISION}/{name}"
    print(f"Downloading {url}", flush=True)
    if name == "checkpoint.pt":
        os.environ["MODELSCOPE_DOWNLOAD_PARALLELS"] = str(workers)
        from modelscope.hub.file_download import parallel_download

        # SDK ranges resume separately from any single-stream download.
        temporary = destination.with_suffix(destination.suffix + ".parallel")
        actual = parallel_download(url, local_dir=str(output), file_name=temporary.name,
                                   cookies=None, file_size=CHECKPOINT_SIZE)
    else:
        temporary = destination.with_suffix(destination.suffix + ".download")
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for block in response.iter_content(8 * 1024 * 1024):
                    handle.write(block)
        actual = file_hash(temporary)
    if actual != SHA256[name]:
        raise ValueError(f"SAM download checksum mismatch: {temporary}")
    temporary.replace(destination)


def download(model_dir, source, workers=16):
    model_dir.mkdir(parents=True, exist_ok=True)
    if source == "modelscope":
        for name in SHA256:
            modelscope_file(name, model_dir, workers)
    else:
        snapshot_download(HF_REPO, revision=HF_REVISION, local_dir=model_dir,
                          allow_patterns=[*SHA256, "LICENSE", "README.md"], max_workers=2)
    for name, expected in SHA256.items():
        if file_hash(model_dir / name) != expected:
            raise ValueError(f"SAM checkpoint/config checksum mismatch: {model_dir / name}")
    snapshot_download("google-t5/t5-base", revision=T5_REVISION, local_dir=model_dir / "t5-base",
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "spiece.model"], max_workers=2)
    manifest = {"source": source, "repo": HF_REPO,
                "revision": MODELSCOPE_REVISION if source == "modelscope" else HF_REVISION,
                "sha256": SHA256, "text_encoder": {"repo": "google-t5/t5-base", "revision": T5_REVISION}}
    (model_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"SAM-Audio ready: {model_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models/SAM-Audio-Large-TV")
    parser.add_argument("--source", choices=("modelscope", "huggingface"), default="modelscope",
                        help="Explicit source; authentication/network failures never switch providers")
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=16,
                        help="ModelScope checkpoint download connections (default 16); SDK resumes each range")
    args = parser.parse_args()
    download(args.model_dir, args.source, args.workers)
