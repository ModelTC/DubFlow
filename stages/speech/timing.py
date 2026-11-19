"""Measure speech duration and fit it into pauses on a fixed video timeline."""

import math
import re
from collections import defaultdict
from statistics import median

import numpy as np

MAX_TEMPO = 1.10
TIMING_POLICY = "fixed_timeline_v1"
ATEMPO_GUARD_SECONDS = .04
CJK = ("zh", "ja", "ko")
DEFAULT_RATES = {"zh": 4.7, "ja": 5., "ko": 4.}


def text_units(text, language):
    if language in CJK:
        return sum(character.isalnum() for character in text)
    return len(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text))


def estimated_seconds(text, language):
    return .10 + text_units(text, language) / DEFAULT_RATES.get(language, 2.5)


def unit_budget(seconds, language, rate=None):
    return max(1, math.floor(max(0., seconds - .10) * (rate or DEFAULT_RATES.get(language, 2.5))))


def voice_key(record, references):
    reference = references[record["speaker_id"]]
    if "voice" in reference:
        return f"preset:{reference['voice']}"
    return reference.get("reference_speaker_id") or record["speaker_id"]


def calibrate_voices(records, references, language):
    """Use real prepared TTS takes. Single-word clips are poor rate estimators."""
    by_voice = defaultdict(list)
    for row in records:
        if row["status"] == "preserve_original" or "tts_duration" not in row:
            continue
        units = text_units(row["target_text"], language)
        duration = row["tts_duration"]
        if units >= (4 if language in CJK else 3) and duration >= .55:
            by_voice[voice_key(row, references)].append(units / (duration - .10))
    rates = {key: {"units_per_second": median(values), "samples": len(values)}
             for key, values in by_voice.items()}
    values = [value for group in by_voice.values() for value in group]
    return {"voices": rates, "language_units_per_second": median(values) if values else None,
            "samples": len(values)}


def available_seconds(record):
    return max(0., record["window_end"] - record["window_start"])


def placement(record):
    """Keep the original onset when possible, then use earlier silence."""
    duration = record["tts_duration"]
    capacity = available_seconds(record)
    if duration <= capacity + 1e-9:
        start = max(record["window_start"], min(record["start"], record["window_end"] - duration))
        return start, 1.0
    # Keep a small safety margin for frame rounding inside FFmpeg atempo.
    return record["window_start"], duration / max(.01, capacity - ATEMPO_GUARD_SECONDS)


def shortening_budget(record, language, calibration, references):
    units = text_units(record["target_text"], language)
    capacity = available_seconds(record)
    voice = calibration["voices"].get(voice_key(record, references), {})
    rate = voice.get("units_per_second") or calibration["language_units_per_second"]
    measured_budget = math.floor(units * capacity / record["tts_duration"])
    calibrated_budget = unit_budget(capacity, language, rate) if rate else measured_budget
    return max(1, min(units - 1, calibrated_budget, measured_budget))


class SourcePauses:
    """Constrain borrowed silence using the separated original speech waveform."""

    def __init__(self, waveform, rate):
        self.block_seconds = .02
        block = round(rate * self.block_seconds)
        count = len(waveform) // block
        audio = waveform[:count * block].reshape(count, block)
        rms = np.sqrt(np.mean(audio * audio, axis=1))
        floor = float(np.quantile(rms, .15)) if rms.size else 0.
        self.active = rms > np.clip(4 * floor, .0025, .006)

    def bound(self, boundary, limit):
        block = self.block_seconds
        left = limit < boundary
        if left:
            edge = math.ceil(boundary / block) - 1
            stop = max(0, math.floor(limit / block))
            active = np.flatnonzero(self.active[stop:edge + 1][::-1])
        else:
            edge = math.floor(boundary / block)
            stop = math.ceil(limit / block)
            active = np.flatnonzero(self.active[edge:stop])

        # A boundary can land inside a word. Follow the immediately connected
        # voice tail for up to 180 ms; later activity ends the borrowed pause.
        voice_tail = .08
        for offset in map(int, active):
            distance = offset * block
            if distance <= voice_tail and distance < .18:
                voice_tail = max(voice_tail, distance + .06)
                continue
            cut = (edge - offset + 1 if left else edge + offset) * block
            return min(boundary, max(limit, cut)) if left else max(boundary, min(limit, cut))
        return limit
