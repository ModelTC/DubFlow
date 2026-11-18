"""Separation cache signatures, speech anchors, and window weights."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from stages.postprocess.utils import SAMPLE_RATE


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def analysis_intervals(plan, duration):
    """Use original ASR parts, not a merged turn spanning intervening pauses."""
    fps = plan.get("fps")
    if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or
            not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0):
        raise ValueError("Analysis plan requires finite positive fps and source duration")
    intervals = []
    for utterance in plan["utterances"]:
        text = utterance.get("text")
        if utterance.get("human_speech") is not True or not isinstance(text, str) or not text.strip():
            continue
        for part in utterance.get("source_parts") or [utterance]:
            part_text = part.get("text", text)
            if part.get("human_speech", True) is not True or not isinstance(part_text, str) or not part_text.strip():
                continue
            interval = part.get("source_interval", part)
            start, end = interval.get("start"), interval.get("end")
            if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in (start, end)) or not 0 <= start < end or start >= duration or
                    end > duration + 1 / fps + 1e-9):
                raise ValueError(f"ASR speech interval is outside the source timeline: {interval}")
            intervals.append({"start": float(start), "end": min(float(end), duration)})
    merged = []
    for interval in sorted(intervals, key=lambda row: (row["start"], row["end"])):
        if merged and interval["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], interval["end"])
        else:
            merged.append(interval.copy())
    return merged


def chunk_anchors(intervals, start, end):
    """Clip source-time intervals to a chunk and express seconds from its start."""
    if intervals is None:
        return None  # Standalone separation can use a text prompt without a transcript.
    first, last = start / SAMPLE_RATE, end / SAMPLE_RATE
    return [["+", max(first, row["start"]) - first, min(last, row["end"]) - first]
            for row in intervals if max(first, row["start"]) < min(last, row["end"])]


def file_signature(path):
    path = Path(path).resolve()
    info = path.stat()
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def implementation_signature():
    return [file_signature(path) for path in sorted(Path(__file__).parent.rglob("*.py"))]


def model_signature(model_dir, text_encoder_dir):
    for name in ("config.json", "checkpoint.pt"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    if not (text_encoder_dir / "config.json").is_file():
        raise FileNotFoundError(text_encoder_dir / "config.json")
    files = [model_dir / "config.json", model_dir / "checkpoint.pt"]
    files += [path for path in sorted(text_encoder_dir.rglob("*"))
              if path.is_file() and ".cache" not in path.parts
              and path.suffix in (".json", ".safetensors", ".bin", ".model")]
    if not any(path.suffix in (".safetensors", ".bin") for path in files[2:]):
        raise FileNotFoundError(f"Missing local T5 weights: {text_encoder_dir}")
    if not any((text_encoder_dir / name).is_file() for name in ("tokenizer.json", "spiece.model")):
        raise FileNotFoundError(f"Missing local T5 tokenizer: {text_encoder_dir}")
    return [file_signature(path) for path in files]


def windows(samples, chunk_samples, overlap_samples):
    if not 0 <= overlap_samples < chunk_samples:
        raise ValueError("Require 0 <= overlap < chunk size")
    start = 0
    while start < samples:
        end = min(start + chunk_samples, samples)
        yield start, end
        if end == samples:
            break
        start += chunk_samples - overlap_samples


def window_weights(start, end, samples, overlap):
    weights = np.ones(end - start, dtype=np.float64)
    edge = min(overlap, len(weights))
    if edge:
        ramp = np.sin(np.linspace(0, np.pi / 2, edge + 2)[1:-1]) ** 2
        if start:
            weights[:edge] *= ramp
        if end < samples:
            weights[-edge:] *= ramp[::-1]
    return weights
