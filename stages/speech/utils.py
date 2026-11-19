"""Speech artifact, timing, and waveform utilities."""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from runtime.utils import write_json


def record_timing(output, operation, seconds, utterances):
    """Accumulate caller wall time, including failed attempts and worker I/O."""
    path = output / "timings.json"
    timings = json.loads(path.read_text()) if path.exists() else {}
    total = timings.setdefault(operation, {"seconds": 0., "calls": 0, "utterances": 0})
    total["seconds"] += seconds
    total["calls"] += 1
    total["utterances"] += utterances
    write_json(path, timings)


def summarize_timings(output):
    rounds = []
    directories = [(output, {"round": 0})]
    directories.extend((directory, {"round": int(directory.name.rsplit("_", 1)[1])})
                       for directory in sorted(output.glob("shortening_round_*")))
    directories.append((output / "short_utterance_resample", {"phase": "short_utterance_resample"}))
    for directory, identity in directories:
        path = directory / "timings.json"
        if path.exists():
            rounds.append({**identity, **json.loads(path.read_text())})
    if not rounds:
        return None
    result = {"measurement": "client_wall_time", "rounds": rounds,
              "translation_seconds": 0., "translation_requests": 0,
              "tts_seconds": 0., "tts_batches": 0, "tts_requested_utterances": 0, "tts_cache_hits": 0}
    for row in rounds:
        translation, tts = row.get("translation", {}), row.get("tts", {})
        result["translation_seconds"] += translation.get("seconds", 0.)
        result["translation_requests"] += translation.get("calls", 0)
        result["tts_seconds"] += tts.get("seconds", 0.)
        result["tts_batches"] += tts.get("calls", 0)
        result["tts_requested_utterances"] += tts.get("utterances", 0)
        result["tts_cache_hits"] += row.get("tts_cache", {}).get("utterances", 0)
    return result


def write_voice_reference(path, waveform, rate):
    """Preserve the WAV byte hash when only its container metadata would change."""
    if path.is_file():
        try:
            existing, existing_rate = sf.read(path, dtype="float32")
        except RuntimeError:
            pass  # An incomplete reference from an interrupted run can be rebuilt.
        else:
            if existing_rate == rate and np.array_equal(existing, waveform):
                return
    temporary = path.with_suffix(".tmp.wav")
    sf.write(temporary, waveform, rate, subtype="FLOAT")
    temporary.replace(path)


def audio_signature(path):
    """Reference WAV content stays reusable when resume rewrites identical bytes."""
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def save_record(output, record):
    directory = output / "items"
    directory.mkdir(exist_ok=True)
    write_json(directory / f"{record['id']}.json", record)


def atempo_filter(tempo):
    factors = []
    while tempo < .5:
        factors.append(.5)
        tempo *= 2
    while tempo > 2:
        factors.append(2.)
        tempo /= 2
    factors.append(tempo)
    return ",".join(f"atempo={factor:.10g}" for factor in factors)


def prepare_waveform(audio, rate, reference_rms):
    """Keep short guards around actual speech, then match the reference's RMS."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("invalid_tts_audio")
    block = max(1, round(rate * .02))
    energy = np.array([np.sqrt(np.mean(audio[index:index + block] ** 2))
                       for index in range(0, len(audio), block)])
    active = np.flatnonzero(energy > max(1e-4, float(energy.max()) * .015))
    if not active.size:
        raise ValueError("silent_tts_audio")
    first = max(0, int(active[0]) * block - round(.04 * rate))
    last = min(len(audio), (int(active[-1]) + 1) * block + round(.06 * rate))
    selected = audio[first:last]
    rms = float(np.sqrt(np.mean(selected ** 2)))
    gain = float(np.clip(reference_rms / max(rms, 1e-8), .5, 2.0)) if reference_rms else 1.0
    peak = float(np.max(np.abs(selected)))
    gain = min(gain, .98 / peak) if peak else gain
    selected = selected * gain
    # Some codecs end the generated utterance without a release guard. Touch
    # only the final 8 ms, then add the missing guard rather than trimming speech.
    active_end = min(len(audio), (int(active[-1]) + 1) * block)
    padding = max(0, round(.04 * rate) - (last - active_end))
    fade = min(len(selected), max(1, round(.008 * rate))) if padding else 0
    if fade:
        selected[-fade:] *= (.5 + .5 * np.cos(np.linspace(0, np.pi, fade))).astype(np.float32)
        selected[-1] = 0
        selected = np.pad(selected, (0, padding))
    return selected, {"trimmed_leading_seconds": first / rate,
                             "trimmed_trailing_seconds": (len(audio) - last) / rate,
                             "rms_before_gain": rms, "speech_gain": gain,
                             "duration_before_trim": len(audio) / rate,
                             "tail_fade_seconds": fade / rate, "padded_trailing_seconds": padding / rate}


def loudness_references(records, waveform, rate, voice):
    """Preset voices need only source loudness, including overlapping or short speech."""
    references = {}
    for record in records:
        if record["status"] == "preserve_original":
            continue
        speaker = record["speaker_id"]
        reference = references.setdefault(speaker, {"sum_squares": 0., "samples": 0})
        audio = waveform[round(record["start"] * rate):round(record["end"] * rate)]
        reference["sum_squares"] += float(np.sum(audio.astype(np.float64) ** 2))
        reference["samples"] += len(audio)
    return {speaker: {"rms": float(np.sqrt(values["sum_squares"] / max(1, values["samples"]))),
                      "voice_origin": "preset_voice", "reference_speaker_id": None, "voice": voice}
            for speaker, values in references.items()}
