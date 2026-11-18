"""Mix translated dialogue with the separated background on one sample timeline."""

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from stages.postprocess.utils import (
    SAMPLE_RATE, file_hash, fit_tail, read_stereo,
    resume_output, temporary_path, write_audio, write_json,
)


def mix_components(background, speech, headroom=0.98):
    if background.shape != speech.shape or not np.isfinite(background).all() or not np.isfinite(speech).all():
        raise ValueError("Separated background/speech must share a finite sample grid")
    mixed = background.astype(np.float64) + speech.astype(np.float64)
    peak = float(np.max(np.abs(mixed)))
    gain = min(1.0, headroom / peak) if peak else 1.0
    return (mixed * gain).astype(np.float32), {"global_gain": gain, "global_gain_db": 20 * math.log10(gain),
                                              "peak_before_gain": peak, "peak_after_gain": peak * gain,
                                              "background_gain": 1.0, "speech_gain": 1.0}


def balance_background(background, speech, rate=SAMPLE_RATE, margin_db=10.):
    """Limit loud background under translated dialogue, without boosting residual speech."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    hop = max(1, round(.05 * rate))
    samples = len(speech)
    blocks = math.ceil(samples / hop)

    def levels(audio):
        padded = np.pad(audio, ((0, blocks * hop - samples), (0, 0)))
        return np.sqrt(np.mean(padded.reshape(blocks, hop, 2).astype(np.float64) ** 2, axis=(1, 2)))

    voice, bed = levels(speech), levels(background)
    active = voice > max(1e-5, float(voice.max()) * .12)
    if not active.any():
        return background, {"policy": "dialogue_background_ceiling_v1", "min_gain_db": 0.}
    reference = float(np.median(voice[active]))
    ceiling = reference * 10 ** (-margin_db / 20)
    # Bridge brief pauses; look ahead before smoothing to avoid a swell at word starts.
    support = maximum_filter1d(active.astype(float), size=21, mode="nearest")
    reduction = np.maximum(0., 20 * np.log10(np.maximum(bed, 1e-10) / ceiling)) * support
    reduction = uniform_filter1d(maximum_filter1d(reduction, size=9, mode="nearest"),
                                 size=9, mode="nearest")
    gain = np.interp(np.arange(samples), (np.arange(blocks) + .5) * hop,
                     10 ** (-reduction / 20)).astype(np.float32)
    return background * gain[:, None], {"policy": "dialogue_background_ceiling_v1",
                                        "margin_db": margin_db, "voice_rms": reference,
                                        "min_gain_db": -float(reduction.max())}


def sample_interval(interval, samples, rate):
    first, last = round(interval["start"] * rate), round(interval["end"] * rate)
    if not 0 <= first < last <= samples:
        raise ValueError(f"Dialogue interval is outside the source timeline: {interval}")
    return first, last


def overlaps(left, right):
    return max(left[0], right[0]) < min(left[1], right[1])


def restore_nonlinguistic_audio(background, original, target, utterances, rate=SAMPLE_RATE,
                                fade_seconds=.04):
    """Restore intact non-linguistic events removed with the source speech stem.

    Never reintroduce original audio where translated speech or its source line
    occurs. The small in-interval fade prevents a hard splice at event edges.
    """
    if background.shape != original.shape or background.shape != target.shape:
        raise ValueError("Original, background and target must share the sample grid")
    samples = len(original)
    protected = []
    for row in utterances:
        if row.get("human_speech", True):
            protected.extend((sample_interval(row["source_interval"], samples, rate),
                              sample_interval(row, samples, rate)))
    mask = np.zeros(samples, dtype=np.float32)
    restored, skipped = [], []
    for row in utterances:
        if row.get("human_speech", True):
            continue
        try:
            first, last = sample_interval(row["source_interval"], samples, rate)
        except (KeyError, ValueError):
            skipped.append(row["id"])
            continue
        span = first, last
        if any(overlaps(span, interval) for interval in protected) or np.any(np.abs(target[first:last]) > 1e-4):
            skipped.append(row["id"])
            continue
        edge = min(round(fade_seconds * rate), (last - first) // 2)
        envelope = np.ones(last - first, dtype=np.float32)
        if edge:
            ramp = np.sin(np.linspace(0, np.pi / 2, edge + 1))[:-1].astype(np.float32) ** 2
            envelope[:edge], envelope[-edge:] = ramp, ramp[::-1]
        mask[first:last] = np.maximum(mask[first:last], envelope)
        restored.append(row["id"])
    retained = background + (original - background) * mask[:, None]
    return retained.astype(np.float32), {"restored_original_ids": restored,
                                         "skipped_original_ids": skipped,
                                         "original_restore_fade_seconds": fade_seconds}


def save_mix(source_video, target, background, speech, output, duration, metadata, background_path=None, prepared_mix=None):
    samples = round(duration * SAMPLE_RATE)
    background, source_tail = fit_tail(background, samples)
    speech, target_tail = fit_tail(speech, samples)
    mixed, gain = mix_components(background, speech) if prepared_mix is None else prepared_mix
    drive_path = output / "target_speech_16k.wav"
    if drive_path.exists() and prepared_mix is None:
        raise FileExistsError(drive_path)
    if background_path is None:
        background_path = output / "source_background_48k.wav"
        write_audio(background_path, background, "FLOAT")
    speech_path = output / "target_speech_48k.wav"
    mixed_path = output / "mixed_48k.wav"
    write_audio(speech_path, speech, "FLOAT")
    write_audio(mixed_path, mixed, "PCM_16")
    speech_peak = float(np.max(np.abs(speech)))
    drive_gain = min(1.0, 0.98 / speech_peak) if speech_peak else 1.0
    temporary_drive = temporary_path(drive_path)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(speech_path),
        "-af", f"volume={drive_gain:.12g}", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(temporary_drive),
    ], check=True)
    drive = sf.info(temporary_drive)
    if drive.frames != round(duration * 16000) or drive.samplerate != 16000 or drive.channels != 1:
        raise ValueError("Driving speech no longer matches the video frame grid")
    temporary_drive.replace(drive_path)

    metadata.update(source_background_tail=source_tail, target_speech_tail=target_tail,
                    mix_gain=gain, speech_drive_gain=drive_gain)
    result = {"source_video": str(source_video), "target_video": str(target),
              "duration": duration, "sample_rate": SAMPLE_RATE,
              "mixed_audio_path": str(mixed_path), "speech_drive_path": str(drive_path),
              "target_speech_path": str(speech_path), "source_background_path": str(background_path),
              "metadata": metadata}
    manifest = output / "audio_mix.json"
    write_json(manifest, result)
    print(manifest)
    return result


def mix_dialogue(source_manifest, dialogue_path, output_dir, duration, fade_seconds=.04):
    """Use the SAM residual throughout; never silently restore untranslated dialogue."""
    source = json.loads(Path(source_manifest).read_text())
    dialogue = json.loads(Path(dialogue_path).read_text())
    if dialogue["audio_policy"] != "complete_dialogue":
        raise ValueError("Only complete_dialogue mixing is supported")
    if abs(source["duration"] - duration) > 1 / SAMPLE_RATE or abs(dialogue["duration"] - duration) > 1 / SAMPLE_RATE:
        raise ValueError("Source and dialogue must share the requested video duration")
    request = {"source_manifest": str(Path(source_manifest).resolve()), "source_manifest_sha256": file_hash(source_manifest),
               "dialogue": str(Path(dialogue_path).resolve()), "dialogue_sha256": file_hash(dialogue_path),
               "duration": duration, "fade_seconds": fade_seconds, "audio_policy": "complete_dialogue", "recipe": 5}
    output, _, completed = resume_output(output_dir, "audio_mix.json", duration, request,
                                         ("preserved_original_48k.wav", "target_speech_48k.wav",
                                          "mixed_48k.wav", "target_speech_16k.wav"))
    if completed is not None:
        for key in ("mixed_audio_path", "speech_drive_path", "target_speech_path", "source_background_path"):
            if not Path(completed[key]).is_file():
                raise FileNotFoundError(completed[key])
        print(output / "audio_mix.json")
        return completed
    human = [row for row in dialogue["utterances"] if row.get("human_speech", True)]
    missing = [row["id"] for row in human if row["status"] != "translated" or not row.get("target_text", "").strip()]
    if missing:
        write_json(output / "incomplete.json", {"status": "incomplete", "audio_policy": "complete_dialogue",
                                               "stage": "audio_mix", "missing_dialogue_ids": missing})
        raise RuntimeError(f"Cannot mix incomplete translated dialogue: {', '.join(missing)}")
    original = read_stereo(source["source_mix_path"])
    retained = read_stereo(source["source_background_path"])
    target = Path(dialogue["target_speech_path"]).resolve()
    speech = read_stereo(target)
    if len(original) != round(duration * SAMPLE_RATE) or retained.shape != original.shape or speech.shape != original.shape:
        raise ValueError("Original, background and translated speech must share the video sample grid")
    for row in human:
        sample_interval(row["source_interval"], len(original), SAMPLE_RATE)
        sample_interval(row, len(original), SAMPLE_RATE)
    retained, restoration = restore_nonlinguistic_audio(retained, original, speech,
                                                        dialogue["utterances"], fade_seconds=fade_seconds)
    retained, balance = balance_background(retained, speech)
    mixed, gain = mix_components(retained, speech)
    retained_path = output / "preserved_original_48k.wav"
    write_audio(retained_path, retained, "FLOAT")
    background_method = "sam_predicted_residual_full_timeline"
    if restoration["restored_original_ids"]:
        background_method += "_with_nonlinguistic_restoration"
    metadata = {"source_manifest": str(Path(source_manifest).resolve()), "dialogue": str(Path(dialogue_path).resolve()),
                "source_separation": source["metadata"], "source_mix_method": "decoded_original_mix",
                "source_background_method": background_method, "target_speech_method": "accepted_tts_only_no_target_separation",
                "accepted_replacement_ids": [row["id"] for row in human],
                "skipped_replacements": [{"id": row["id"], "reason": row.get("reason", "non_linguistic_audio")}
                                        for row in dialogue["utterances"] if not row.get("human_speech", True)],
                "audio_policy": "complete_dialogue", "audio_complete": True, "mix_gain": gain,
                "source_mask_fade_seconds": 0., "target_speech_gain": gain["global_gain"],
                "background_balance": balance, **restoration,
                "limitations": [
                    "SAM residual may retain faint original speech or alter other sounds.",
                    "Loud background is smoothly attenuated below translated dialogue; non-linguistic events are restored only outside dialogue.",
                    "Every recognized human line must have translated audio; failed lines never silently restore original dialogue."]}
    return save_mix(source["source_video"], target, retained, speech, output, duration, metadata, retained_path,
                    prepared_mix=(mixed, gain))


def run_stage(*, source_manifest, dialogue, output_dir, duration, speech_fade_seconds=.04):
    return mix_dialogue(source_manifest, dialogue, output_dir, duration, speech_fade_seconds)
