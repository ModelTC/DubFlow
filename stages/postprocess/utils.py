"""Audio artifact IO and sample-grid utilities shared by output stages."""

import hashlib
import json
import math
import subprocess
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

SAMPLE_RATE = 48000


def temporary_path(path):
    path = Path(path)
    return path.with_name(f".{path.stem}-{uuid4().hex}.tmp{path.suffix}")


def write_json(path, value):
    temporary = temporary_path(path)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_audio(path, values, subtype):
    temporary = temporary_path(path)
    sf.write(temporary, values, SAMPLE_RATE, subtype=subtype)
    temporary.replace(path)


def extract_audio(video, output, reuse=False):
    output = Path(output)
    if output.exists():
        if reuse:
            return read_stereo(output)
        raise FileExistsError(output)
    temporary = temporary_path(output)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-i", str(video), "-map", "0:a:0", "-vn", "-ar", str(SAMPLE_RATE),
        "-ac", "2", "-c:a", "pcm_f32le", str(temporary),
    ], check=True)
    audio, rate = sf.read(temporary, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or audio.shape[1] != 2 or not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Invalid decoded stereo audio: {output}")
    temporary.replace(output)
    return audio


def fit_tail(audio, samples):
    """只在末尾裁切/补零，不改变时间原点、语速或声道。"""
    output = np.zeros((samples, 2), dtype=np.float32)
    output[:min(samples, len(audio))] = audio[:samples]
    return output, {"input_samples": len(audio), "output_samples": samples,
                    "padded_samples": max(0, samples - len(audio)),
                    "trimmed_samples": max(0, len(audio) - samples)}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_stereo(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or audio.shape[1] != 2 or not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite 48kHz stereo audio: {path}")
    return audio


def new_output(output_dir, name, duration):
    if duration <= 0 or not math.isfinite(duration):
        raise ValueError("Duration must be positive and finite")
    output = Path(output_dir).resolve()
    manifest = output / name
    if manifest.exists():
        raise FileExistsError(f"Use pipeline resume or a new output directory: {manifest}")
    output.mkdir(parents=True, exist_ok=True)
    return output, manifest


def resume_output(output_dir, name, duration, request, generated):
    if duration <= 0 or not math.isfinite(duration):
        raise ValueError("Duration must be positive and finite")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / name
    marker = output / f"{Path(name).stem}.request.json"
    if marker.exists():
        if json.loads(marker.read_text()) != request:
            raise ValueError(f"Output belongs to another audio request: {output}")
    else:
        if manifest.exists() or any((output / item).exists() for item in generated):
            raise FileExistsError(f"Audio output has no matching request marker: {output}")
        write_json(marker, request)
    if manifest.exists():
        return output, manifest, json.loads(manifest.read_text())
    new_output(output, name, duration)
    return output, manifest, None
