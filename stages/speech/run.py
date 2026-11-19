"""翻译原片对白，按说话人克隆音色，并在同一时间轴安排目标语言语音。"""

import hashlib
from dataclasses import dataclass, field
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from runtime.utils import file_signature as signature, write_json
from runtime.clients import Translator

from stages.speech.model import REVIEW_SYSTEM, create_speech_model, semantic_review_prompt, translation_prompt
from stages.speech.timing import (CJK, MAX_TEMPO, TIMING_POLICY, SourcePauses, available_seconds,
                                  calibrate_voices, estimated_seconds, placement, shortening_budget, text_units)
from stages.speech.utils import (atempo_filter, audio_signature, loudness_references, prepare_waveform, record_timing, save_record,
                                 summarize_timings, write_voice_reference)

SHORTENING_ROUNDS = 3
SHORTENING_TRIGGER_TEMPO = 1.05
SHORT_SAMPLE_RETRIES = 3


@dataclass(frozen=True, kw_only=True)
class SpeechRequest:
    """Shared translation, synthesis, and timing settings for one source plan."""

    plan: str
    speech_source: str
    output_dir: str
    source_language: str
    target_language: str
    translation_url: str
    translation_model: str
    tts_model: str
    device: str = "cuda:0"
    seed: int = 42
    duration_mode: str = "fixed"
    translation_backend: str = "lightllm"
    translation_api_key_env: str = "LIGHTDUBBER_LLM_API_KEY"
    translation_options: dict = field(default_factory=dict)
    tts_backend: str = "qwen3tts"
    tts_url: str = ""
    tts_api_key_env: str = "LIGHTDUBBER_TTS_API_KEY"
    tts_voice: str = "alloy"
    tts_python: str = sys.executable
    qwen_x_vector_only_mode: bool = True


def preserve(record, reason):
    return {**record, "status": "preserve_original", "reason": reason}


def speaking_windows(plan, waveform=None, rate=48000):
    """Allow bounded neighboring pauses; picture cuts are handled by lip planning."""
    pauses = SourcePauses(waveform, rate) if waveform is not None else None
    duration, fps = plan["frames"] / plan["fps"], plan["fps"]
    records = []
    for record in plan["utterances"]:
        if record["end"] > duration:
            if record["end"] > duration + 1 / fps:
                raise ValueError(f"Source utterance exceeds the video by more than one frame: {record['id']}")
            record = {**record, "end": duration,
                      "raw_source_interval": {"start": record["start"], "end": record["end"]},
                      "source_timing_adjustment": {"method": "clamp_end_within_one_frame",
                                                   "seconds": record["end"] - duration,
                                                   "frame_seconds": 1 / fps, "video_duration": duration}}
        if not 0 <= record["start"] < record["end"] <= duration:
            raise ValueError(f"Invalid source utterance on the video timeline: {record['id']}")
        records.append(record)
    records.sort(key=lambda item: (item["start"], item["end"]))
    result = []
    for index, record in enumerate(records):
        reason = None
        if not record.get("human_speech", True):
            reason = "not_confirmed_human_speech"
        elif record["speaker_id"] is None or len(record["speaker_ids"]) != 1:
            reason = "unseparated_speech"
        elif any(other["id"] != record["id"] and
                 min(other["end"], record["end"]) - max(other["start"], record["start"]) > 1 / fps
                 for other in records):
            reason = "overlapping_speech"
        left = max(0., record["start"] - .35)
        right = min(duration, record["end"] + .65)
        if index:
            left = max(left, min(record["start"], (records[index - 1]["end"] + record["start"]) / 2))
        if index + 1 < len(records):
            right = min(right, max(record["end"], (record["end"] + records[index + 1]["start"]) / 2))
        if pauses is not None:
            left = pauses.bound(record["start"], left)
            right = pauses.bound(record["end"], right)
        item = {**record, "window_start": left, "window_end": right, "tts_seed_index": index,
                "human_speech": record.get("human_speech", True), "status": "pending", "reason": None}
        if item["human_speech"]:
            item.update(source_speaker_id=record["speaker_id"], source_speaker_ids=list(record["speaker_ids"]),
                        audio_only=record["person_id"] is None or reason in ("overlapping_speech", "unseparated_speech"))
            if reason in ("overlapping_speech", "unseparated_speech"):
                item.update(window_start=record["start"], window_end=record["end"])
            if item["speaker_id"] is None or len(item["speaker_ids"]) != 1:
                speaker = f"audio_only_{record['id']}"
                item.update(speaker_id=speaker, speaker_ids=[speaker])
            reason = None
        result.append(preserve(item, reason) if reason else item)
    return result


def _voice_reference_aliases(records, speaker_bindings):
    """Share voice references, not face locations, across unambiguous namespaces."""
    if not speaker_bindings:
        return {}
    # Inspect original parts before reference extraction applies the merged
    # record's final visual binding. Otherwise a conflicting part can disappear.
    parts = [{**record, **part, "status": record["status"]}
             for record in records for part in record.get("source_parts", [record])]
    acoustic_ids = []
    for part in parts:
        speakers = part.get("acoustic_speaker_ids")
        if speakers is None:
            speaker = part.get("acoustic_speaker_id")
            speakers = [speaker] if speaker else []
        acoustic_ids.append(set(speakers))
    blocked, people, trusted = set(), {}, {}
    for index, part in enumerate(parts):
        speakers = acoustic_ids[index]
        overlaps = any(index != other_index and
                       min(part["end"], other["end"]) > max(part["start"], other["start"])
                       for other_index, other in enumerate(parts))
        if (len(speakers) != 1 or len(part.get("speaker_ids", [])) != 1 or
                len(part.get("source_speaker_ids", part.get("speaker_ids", []))) != 1 or
                not part.get("human_speech", True) or overlaps):
            blocked.update(speakers)
            continue
        speaker = next(iter(speakers))
        person = part.get("person_id")
        if person is not None:
            people.setdefault(speaker, set()).add(person)
            if (part.get("speaker_identity_source") == "bound_visual_person" and
                    part.get("speaker_id") == f"visual__{person}" and
                    part.get("assignment_source") in ("word_local_visual_evidence", "consistent_word_visual_evidence") and
                    part["status"] != "preserve_original"):
                trusted.setdefault(speaker, set()).add(person)
    aliases = {}
    for speaker in dict.fromkeys(record["speaker_id"] for record in records):
        if speaker in blocked or len(people.get(speaker, ())) != 1:
            continue
        person = next(iter(people[speaker]))
        binding = speaker_bindings.get(speaker, {})
        if (trusted.get(speaker) != {person} or binding.get("status") != "bound" or
                binding.get("person_id") != person):
            continue
        recipients = [record for record in records if record["speaker_id"] == speaker]
        if not recipients or any(
                record.get("person_id") is not None or record["status"] == "preserve_original" or
                not record.get("human_speech", True) or record.get("speaker_ids") != [speaker] or
                record.get("acoustic_speaker_id") != speaker or
                record.get("acoustic_speaker_ids", [speaker]) != [speaker]
                for record in recipients):
            continue
        aliases[speaker] = {"speaker_id": f"visual__{person}", "person_id": person,
                            "acoustic_speaker_id": speaker,
                            "policy": "unique_nonoverlapping_acoustic_visual_voice_v1"}
    return aliases


def voice_references(records, waveform, rate, output, speaker_bindings=None):
    references = {}
    aliases = _voice_reference_aliases(records, speaker_bindings)
    records = [{**record, **part, "speaker_id": record["speaker_id"],
                "speaker_ids": record["speaker_ids"], "person_id": record["person_id"],
                "status": record["status"],
                "source_speaker_ids": record.get("source_speaker_ids", record["speaker_ids"])}
               for record in records for part in record.get("source_parts", [record])]
    eligible = [record for record in records if record["status"] != "preserve_original"]
    for number, speaker in enumerate(dict.fromkeys(record["speaker_id"] for record in eligible)):
        people = {record["person_id"] for record in eligible if record["speaker_id"] == speaker}
        if len(people) > 1:
            continue
        candidates = [record for record in eligible if record["speaker_id"] == speaker
                      and record["end"] > record["start"]
                      and len(record.get("source_speaker_ids", record["speaker_ids"])) <= 1
                      and not any(other["speaker_id"] != speaker and
                                  min(record["end"], other["end"]) > max(record["start"], other["start"])
                                  for other in records)]
        candidates = [{**record, "end": min(record["end"], record["start"] + 10)} for record in candidates]
        selected, seconds = [], 0.0
        for record in sorted(candidates, key=lambda item: item["end"] - item["start"], reverse=True):
            length = record["end"] - record["start"]
            if seconds + length <= 10:
                selected.append(record)
                seconds += length
            if seconds >= 3:
                break
        if not selected:
            continue
        selected.sort(key=lambda item: item["start"])
        audio = np.concatenate([waveform[round(record["start"] * rate):round(record["end"] * rate)]
                                for record in selected])
        if not audio.size or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
            continue
        reference_rms = float(np.sqrt(np.mean(audio ** 2)))
        real_seconds = len(audio) / rate
        if len(audio) < rate:
            audio = np.pad(audio, (0, rate - len(audio)))
        path = output / f"voice_{number:02d}.wav"
        write_voice_reference(path, audio, rate)
        references[speaker] = {"path": str(path), "text": " ".join(record["text"] for record in selected),
                               "duration": len(audio) / rate,
                               "source_intervals": [{"id": record["id"], "start": record["start"], "end": record["end"]}
                                                    for record in selected],
                               "clone_mode": "in_context" if seconds >= 3 else "speaker_embedding_only"}
        references[speaker].update(rms=reference_rms, voice_origin="same_speaker_source_speech",
                                   reference_speaker_id=speaker, real_speech_seconds=real_seconds,
                                   padded_seconds=len(audio) / rate - real_seconds)
    for speaker, identity in aliases.items():
        reference = references.get(identity["speaker_id"])
        if (reference is not None and reference.get("voice_origin") == "same_speaker_source_speech" and
                reference.get("reference_speaker_id") == identity["speaker_id"]):
            # Keep unresolved visual rows untouched, and do not add their audio
            # to the clean visual reference. Reuse only the proven donor stem.
            references[speaker] = {**reference, "reference_identity": identity}
    if references:
        donor = max((key for key in references if "reference_identity" not in references[key]),
                    key=lambda key: references[key]["real_speech_seconds"])
        for speaker in dict.fromkeys(record["speaker_id"] for record in eligible):
            if speaker not in references:
                references[speaker] = {**references[donor], "voice_origin": "same_video_other_speaker_non_cloned",
                                       "reference_speaker_id": donor}
    return references


def require_complete_dialogue(records, output, stage, allow_pending=False):
    missing = [{"id": row["id"], "text": row["text"], "reason": row.get("reason", "untranslated_dialogue")}
               for row in records if row.get("human_speech", True) and row["status"] != "translated"
               and not (allow_pending and row["status"] == "pending")]
    if missing:
        write_json(output / "incomplete.json", {"status": "incomplete", "audio_policy": "complete_dialogue",
                                                "stage": stage, "missing_dialogue": missing})
        raise RuntimeError(f"Incomplete translated dialogue at {stage}: {', '.join(row['id'] for row in missing)}")


def ask_translator(translator, prompt, output, *, system=None):
    started = time.perf_counter()
    try:
        return translator(prompt, system=system, allow_partial=True) if system else translator(prompt)
    finally:
        if output is not None:
            record_timing(output, "translation", time.perf_counter() - started, 1)


def translate(records, conversation, config, shorten=False, output=None):
    translator = Translator(config.translation_url, config.translation_model,
                            config.translation_api_key_env, config.translation_backend, config.translation_options)
    results = []
    try:
        for record in records:
            round_number = record.get("shortening_round", int(shorten))
            history = list(record.get("translation_history", []))
            prompt_record = record
            for attempt in range(2 if shorten else 1):
                prompt = translation_prompt(prompt_record, conversation, config.source_language,
                                            config.target_language, shorten)
                proposed = ask_translator(translator, prompt, output)
                review, feedback = "initial_translation", None
                if shorten:
                    if proposed == record["target_text"]:
                        review = "unchanged"
                    else:
                        review_prompt = semantic_review_prompt(record, proposed, config.source_language,
                                                               config.target_language)
                        answer = ask_translator(translator, review_prompt, output, system=REVIEW_SYSTEM)
                        verdict = answer.split("\n", 1)[0].strip().strip("*`\"").rstrip(".")
                        review = "passed" if verdict.upper() == "PASS" else "rejected"
                        if review == "rejected":
                            label, _, detail = verdict.partition(":")
                            if label.strip().upper() == "FAIL":
                                feedback = detail.strip()[:300]
                history.append({"round": round_number, "source_text": record["text"], "text": proposed,
                                "review": review, "review_feedback": feedback,
                                "available_seconds": available_seconds(record),
                                "word_budget": record.get("shortening_word_budget")})
                if shorten and review == "unchanged" and attempt == 0:
                    prompt_record = {**record, "try_different_wording": True}
                    continue
                if shorten and review == "rejected" and feedback and attempt == 0:
                    prompt_record = {**record, "rejected_translation": proposed, "review_feedback": feedback}
                    continue
                break
            text = record["target_text"] if review == "rejected" else proposed
            item = {**record, "target_text": text, "raw_model_text": proposed,
                    "translation_history": history, "translation_source": config.translation_model,
                    "translation_attempt": round_number, "translation_ready": True,
                    "translation_review": review, "time_compression_policy": TIMING_POLICY}
            if output is not None:
                save_record(output, item)
            results.append(item)
        return results
    finally:
        translator.close()


def prepared_tts(record, reference, path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError("invalid_tts_audio")
    item = {**record, "tts_generated_path": str(path)}
    audio, details = prepare_waveform(audio[:, 0], rate, reference["rms"])
    prepared = path.with_name(path.stem + "_prepared.wav")
    temporary = prepared.with_suffix(".tmp.wav")
    sf.write(temporary, audio, rate, subtype="FLOAT")
    temporary.replace(prepared)
    item.update(tts_preparation=details)
    path = prepared
    if not audio.size or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
        raise ValueError("invalid_tts_audio")
    duration = len(audio) / rate
    item.update(tts_raw_path=str(path), tts_sample_rate=rate, tts_duration=duration,
                tts_natural_duration=duration, tts_natural_path=str(path), tts_signature=signature(path))
    return item


def synthesis_settings(record, reference, config, backend, seed, index):
    return {"model": backend.identity, "text": record["target_text"],
            "language": config.target_language, "speaker_id": record["speaker_id"],
            "base_seed": config.seed + record.get("tts_seed_index", index), "seed": seed,
            "reference": audio_signature(reference["path"]) if reference.get("path") else None,
            "reference_text": reference.get("text", ""),
            "reference_rms": reference["rms"]}


def reusable_tts(record, settings):
    """Keep the previous sample for identical conditioning; a changed root seed invalidates it."""
    if record.get("tts_cache_settings") != settings:
        return False
    path = Path(record["tts_raw_path"])
    return path.is_file() and record["tts_signature"] == signature(path)


def synthesize(records, references, config, output, attempt=0):
    if not records:
        return []
    backend = create_speech_model(config)
    rows, pending, results = [], [], {}
    for index, record in enumerate(records):
        reference = references[record["speaker_id"]]
        existing_settings = record.get("tts_cache_settings")
        matching_conditioning = False
        if existing_settings:
            started = time.perf_counter()
            settings = synthesis_settings(record, reference, config, backend, record["tts_seed"], index)
            matching_conditioning = settings == existing_settings
            if reusable_tts(record, settings):
                item = {**record, "tts_reused": True}
                save_record(output, item)
                record_timing(output, "tts_cache", time.perf_counter() - started, 1)
                results[item["id"]] = item
                print(f"TTS cache {item['id']}: {item['tts_duration']:.3f}s", flush=True)
                continue
        current_attempt = max(attempt, record.get("tts_attempt", 0), record.get("translation_attempt", 0))
        if matching_conditioning:
            seed = record["tts_seed"]
        else:
            seed = (config.seed + record.get("tts_seed_index", index) + current_attempt * 1000
                    + record.get("tts_sample_offset", 0))
            settings = synthesis_settings(record, reference, config, backend, seed, index)
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:20]
        path = output / f"tts_{record['id']}_{digest}.wav"
        rows.append({"path": str(path), "text": record["target_text"], "reference": reference.get("path"),
                     "reference_text": reference.get("text", ""), "seed": seed,
                     "x_vector_only_mode": config.qwen_x_vector_only_mode})
        fresh = {key: value for key, value in record.items() if not key.startswith("tts_") or key == "tts_seed_index"}
        if not matching_conditioning:
            fresh.pop("natural_resample_attempted", None)
        pending.append({**fresh, "tts_seed": seed, "tts_attempt": current_attempt,
                        "tts_backend": config.tts_backend, "tts_model": backend.name,
                        "tts_reference_mode": backend.reference_mode,
                        "reference_speaker_id": record["speaker_id"] if config.tts_backend == "qwen3tts" else None,
                        "tts_voice": config.tts_voice if config.tts_backend == "openai_compatible" else None,
                        "tts_cache_settings": settings, "tts_reused": False})
    if rows:
        request = output / f"tts_request_{attempt}.json"
        write_json(request, {"model": backend.name, "device": config.device,
                             "language": config.target_language, "rows": rows})
        missing = sum(not Path(row["path"]).is_file() for row in rows)
        started = time.perf_counter()
        try:
            backend.generate(request)
        finally:
            record_timing(output, "tts", time.perf_counter() - started, missing)
        for item, row in zip(pending, rows):
            item = prepared_tts(item, references[item["speaker_id"]], Path(row["path"]))
            save_record(output, item)
            results[item["id"]] = item
            print(f"TTS {item['id']}: {item['tts_duration']:.3f}s", flush=True)
    return [results[record["id"]] for record in records]


def speech_attempt(record, round_number):
    result = {"round": round_number, "source_text": record["text"],
              "target_text": record.get("target_text"), "available_seconds": available_seconds(record),
              "word_budget": record.get("shortening_word_budget"), "reason": record.get("reason")}
    if record["status"] != "preserve_original":
        result.update(tts_duration=record["tts_duration"],
                      tts_natural_duration=record.get("tts_natural_duration", record["tts_duration"]),
                      effective_tempo=placement(record)[1], tts_attempt=record["tts_attempt"],
                      tts_seed=record.get("tts_seed"), post_tempo=placement(record)[1],
                      fits=placement(record)[1] == 1, tts_reused=record.get("tts_reused", False))
    return result


def shorten_to_fit(records, references, config, output):
    """Refine long translations using each voice's measured speaking rate."""
    calibration_path = output / "duration_calibration.json"
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text())
    else:
        calibration = calibrate_voices(records, references, config.target_language)
        write_json(calibration_path, calibration)

    best = {row["id"]: {**row, "speech_attempts": row.get("speech_attempts") or [speech_attempt(row, 0)]}
            for row in records}
    for round_number in range(1, SHORTENING_ROUNDS + 1):
        pending = []
        for item in best.values():
            if (item["status"] == "preserve_original" or item.get("shortening_stopped_reason")
                    or item.get("shortening_rounds_completed", 0) >= round_number
                    or placement(item)[1] <= SHORTENING_TRIGGER_TEMPO):
                continue
            pending.append({**item, "shortening_round": round_number,
                            "shortening_word_budget": shortening_budget(item, config.target_language,
                                                                         calibration, references)})
        if not pending:
            continue
        round_output = output / f"shortening_round_{round_number}"
        round_output.mkdir(exist_ok=True)
        shortened = translate(pending, list(best.values()), config, shorten=True, output=round_output)
        for candidate in synthesize(shortened, references, config, round_output, attempt=round_number):
            key = candidate["id"]
            previous = best[key]
            improved = candidate["tts_duration"] < previous["tts_duration"] - .02
            selected = {**(candidate if improved else previous),
                        "translation_history": candidate["translation_history"],
                        "speech_attempts": [*previous["speech_attempts"], speech_attempt(candidate, round_number)],
                        "shortening_rounds_completed": round_number,
                        "shortening_word_budget": candidate["shortening_word_budget"]}
            if candidate.get("translation_review") == "rejected":
                selected["shortening_stopped_reason"] = "meaning_review_rejected"
            elif candidate.get("tts_reused"):
                selected["shortening_stopped_reason"] = "unchanged_translation"
            elif not improved:
                selected["shortening_stopped_reason"] = "no_duration_improvement"
            best[key] = selected
            save_record(output, selected)
    return [best[record["id"]] for record in records]


def resample_short_utterances(records, references, config, output):
    """Sample a few natural takes for short reactions or near-fit speech."""
    if config.tts_backend != "qwen3tts":
        return records
    eligible = {row["id"]: row for row in records
                if row["status"] != "preserve_original" and placement(row)[1] > MAX_TEMPO
                and (text_units(row["target_text"], config.target_language) <= (3 if config.target_language in CJK else 1)
                     or placement(row)[1] <= 1.35
                     or row["tts_duration"] > 1.6 * estimated_seconds(row["target_text"], config.target_language))
                and not row.get("natural_resample_attempted")}
    if not eligible:
        return records
    attempts_dir = output / "short_utterance_resample"
    attempts_dir.mkdir(exist_ok=True)
    variants = [{**row, "id": f"{key}_sample_{sample}", "tts_sample_offset": sample * 71}
                for key, row in eligible.items() for sample in range(1, SHORT_SAMPLE_RETRIES + 1)]
    for variant in variants:
        variant.pop("tts_cache_settings", None)
    samples = synthesize(variants, references, config, attempts_dir)
    best = dict(eligible)
    for measured in samples:
        key = measured["id"].rsplit("_sample_", 1)[0]
        candidate = {**measured, "id": key}
        if candidate["tts_duration"] < best[key]["tts_duration"]:
            best[key] = candidate
    for key, selected in best.items():
        best[key] = {**selected, "natural_resample_attempted": True}
        save_record(output, best[key])
    return [best.get(row["id"], row) for row in records]


def duration_fit_issue(row, tempo, reason="natural_speech_exceeds_window"):
    return {"id": row["id"], "text": row["text"], "target_text": row["target_text"],
            "reason": reason, "natural_seconds": row["tts_duration"], "available_seconds": available_seconds(row),
            "required_tempo": tempo, "max_tempo": MAX_TEMPO}


def duration_fit_error(failures, output):
    write_json(output / "incomplete.json", {"status": "incomplete", "audio_policy": "complete_dialogue",
                                            "stage": "duration_fit", "missing_dialogue": failures})
    ratios = ", ".join(f"{row['id']} ({row['required_tempo']:.2f}x)" for row in failures)
    raise RuntimeError(f"Dialogue exceeds the {MAX_TEMPO:.2f}x tempo limit on the fixed timeline: {ratios}")


def required_tempo(record):
    return record["tts_duration"] / max(1 / 48000, available_seconds(record))


def require_duration_fit(records, output):
    failures = [duration_fit_issue(row, required_tempo(row)) for row in records
                if row["status"] != "preserve_original" and required_tempo(row) > MAX_TEMPO]
    if failures:
        duration_fit_error(failures, output)


def align_record(record, output):
    start, preferred = placement(record)
    if preferred > MAX_TEMPO:
        # Short clips can be rejected by the rounding guard even when the actual
        # atempo output fits. Measure at a few bounded speeds instead.
        minimum = min(required_tempo(record), MAX_TEMPO)
        tempos = (minimum, (minimum + MAX_TEMPO) / 2, MAX_TEMPO)
    else:
        tempos = (preferred, MAX_TEMPO)
    path = output / f"aligned_{record['id']}.wav"
    meta = path.with_suffix(".json")
    cached = json.loads(meta.read_text()) if meta.is_file() and path.is_file() else None
    temporary = path.with_suffix(".tmp.wav")
    source = signature(record["tts_raw_path"])
    try:
        for tempo in dict.fromkeys(tempos):
            settings = {"source": source, "start": start, "tempo": tempo}
            audio_path = path
            if settings != cached:
                audio_path = temporary
                command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", record["tts_raw_path"]]
                if tempo != 1:
                    command.extend(["-af", atempo_filter(tempo)])
                subprocess.run(command + ["-ar", "48000", "-ac", "2", "-c:a", "pcm_f32le", str(temporary)], check=True)
            audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
            if not np.isfinite(audio).all():
                raise ValueError(f"Non-finite retimed speech: {record['id']}")
            first, last = round(start * rate), round(start * rate) + len(audio)
            if last <= round(record["window_end"] * rate):
                if audio_path == temporary:
                    temporary.replace(path)
                    write_json(meta, settings)
                break
        else:
            fitted_seconds = len(audio) / rate
            estimated = MAX_TEMPO * fitted_seconds / max(1 / rate, record["window_end"] - start)
            issue = duration_fit_issue(record, estimated, "rendered_speech_exceeds_window")
            duration_fit_error([{**issue, "rendered_seconds_at_max_tempo": fitted_seconds}], output)
    finally:
        temporary.unlink(missing_ok=True)
    row = {**record, "status": "translated", "reason": None, "start": first / rate, "end": last / rate,
           "source_interval": {"start": record["start"], "end": record["end"]},
           "allowed_interval": {"start": record["window_start"], "end": record["window_end"]},
           "tts_aligned_path": str(path), "tempo_ratio": tempo,
           "borrowed_pause_seconds": max(0., record["start"] - first / rate) + max(0., last / rate - record["end"])}
    natural = record.get("tts_natural_duration", record["tts_duration"])
    row.update(max_tempo=MAX_TEMPO, time_compression={"natural_seconds": natural,
               "native_duration_factor": record.get("tts_native_duration_factor", 1.),
               "native_speed": natural / record["tts_duration"], "post_speed": tempo,
               "total_speed": natural / record["tts_duration"] * tempo,
               "output_seconds": len(audio) / rate})
    return row, audio


def assemble(records, references, plan, config, output):
    require_duration_fit(records, output)
    duration = plan["frames"] / plan["fps"]
    samples = round(duration * 48000)
    tracks = {speaker: np.zeros((samples, 2), np.float32) for speaker in references}
    utterances = []
    for record in records:
        if record["status"] == "preserve_original":
            row, aligned = record, None
        else:
            row, aligned = align_record(record, output)
        if aligned is None:
            save_record(output, preserve(record, row["reason"]))
            row = {**row, "attempted_target_text": row.get("target_text")}
            row.update(status="preserve_original", source_interval={"start": record["start"], "end": record["end"]},
                       start=record["start"], end=record["end"], target_text=None)
        else:
            first, last = round(row["start"] * 48000), round(row["end"] * 48000)
            tracks[record["speaker_id"]][first:last] += aligned
            save_record(output, {**record, "status": "translated"})
        utterances.append(row)
    require_complete_dialogue(utterances, output, "speech_generation")
    full = sum(tracks.values(), np.zeros((samples, 2), np.float32))
    peak = float(np.max(np.abs(full))) if samples else 0
    gain = min(1.0, .98 / peak) if peak else 1.0
    full *= gain
    target_path = output / "target_speech_48k.wav"
    sf.write(target_path, full, 48000, subtype="FLOAT")
    speaker_paths = {}
    for number, (speaker, audio) in enumerate(tracks.items()):
        native_path, drive_path = output / f"speaker_{number:02d}_48k.wav", output / f"speaker_{number:02d}_16k.wav"
        sf.write(native_path, audio * gain, 48000, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(native_path),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(drive_path)], check=True)
        speaker_paths[speaker] = str(drive_path)
    successful = [row for row in utterances if row["status"] == "translated"]
    result = {"source_plan": str(Path(config.plan).resolve()), "source_speech": str(Path(config.speech_source).resolve()),
              "source_language": config.source_language, "target_language": config.target_language,
              "duration": duration, "sample_rate": 48000, "global_gain": gain, "seed": config.seed,
              "utterances": utterances, "target_speech_path": str(target_path),
              "speaker_tracks": speaker_paths, "voice_references": references,
              "replace_intervals": [{"id": row["id"], **row["source_interval"]} for row in successful],
              "counts": {"translated": len(successful), "preserve_original": len(utterances) - len(successful)},
              "human_review_required": False}
    calibration_path = output / "duration_calibration.json"
    result.update(audio_policy="complete_dialogue", audio_complete=True,
                  timing_fit={"policy": TIMING_POLICY, "max_tempo": MAX_TEMPO,
                              "max_applied_tempo": max((row["tempo_ratio"] for row in successful), default=1.),
                              "accelerated_utterances": sum(row["tempo_ratio"] > 1 for row in successful),
                              "borrowed_pause_utterances": sum(row["borrowed_pause_seconds"] > 1 / 48000
                                                               for row in successful),
                              "calibration": json.loads(calibration_path.read_text()) if calibration_path.is_file() else {}})
    (output / "source_text.txt").write_text("\n".join(f"{row['id']} {row['text']}" for row in utterances) + "\n")
    (output / "translated.txt").write_text("\n".join(f"{row['id']} {row['target_text']}" for row in successful) + "\n")
    timings = summarize_timings(output)
    if timings is not None:
        result["timings"] = timings
    write_json(output / "dialogue.json", result)
    (output / "incomplete.json").unlink(missing_ok=True)
    return result


def run_speech(config):
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if config.duration_mode != "fixed":
        raise ValueError("Dubbing requires a fixed video timeline")
    request = {"plan": signature(config.plan), "source_speech": signature(config.speech_source),
               "source_language": config.source_language,
               "target_language": config.target_language, "duration_mode": "fixed",
               "seed": config.seed, "recipe": 12}
    backend = create_speech_model(config)
    request["backends"] = {
        "tts_identity": backend.identity,
        "translation_url": config.translation_url, "translation_model": config.translation_model,
        "translation_backend": config.translation_backend, "translation_api_key_env": config.translation_api_key_env,
        "translation_options": config.translation_options,
        "tts_backend": config.tts_backend,
        "tts_model": str(Path(config.tts_model).resolve()) if config.tts_backend == "qwen3tts" else config.tts_model,
        "tts_url": config.tts_url, "tts_voice": config.tts_voice, "tts_api_key_env": config.tts_api_key_env,
        "tts_python": config.tts_python, "qwen_x_vector_only_mode": config.qwen_x_vector_only_mode,
        "speech_adapter": signature(Path(__file__)),
        "adapter": signature(Path(__file__).with_name("model.py")),
        "utils": signature(Path(__file__).with_name("utils.py")),
        "timing": signature(Path(__file__).with_name("timing.py")),
    }
    request["backends"].update(translation_review="llm_semantic_review", shortening_rounds=SHORTENING_ROUNDS,
                               time_compression_policy=TIMING_POLICY, max_tempo=MAX_TEMPO, audio_policy="complete_dialogue")
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("Speech output belongs to another request; choose a new directory")
        if (output / "dialogue.json").exists():
            completed = json.loads((output / "dialogue.json").read_text())
            for index, record in enumerate(completed["utterances"]):
                if record["status"] == "preserve_original":
                    continue
                reference = completed["voice_references"][record["speaker_id"]]
                if reference.get("path") and not Path(reference["path"]).is_file():
                    break
                settings = synthesis_settings(record, reference, config, backend, record["tts_seed"], index)
                if not reusable_tts(record, settings):
                    break
            else:
                return completed
    else:
        if any(output.iterdir()):
            raise FileExistsError(output)
        write_json(request_path, request)
    plan = json.loads(Path(config.plan).read_text())
    audio, rate = sf.read(config.speech_source, dtype="float32", always_2d=True)
    if rate != 48000 or len(audio) != round(plan["frames"] / plan["fps"] * rate):
        raise ValueError("Source speech must share the plan's 48kHz timeline")
    waveform = audio.mean(1)
    records = speaking_windows(plan, waveform, rate)
    if config.tts_backend == "qwen3tts":
        references = voice_references(records, waveform, rate, output, speaker_bindings=plan.get("speakers"))
    else:
        references = loudness_references(records, waveform, rate, config.tts_voice)
    write_json(output / "voice_references.json", references)
    for index, record in enumerate(records):
        checkpoint = output / "items" / f"{record['id']}.json"
        if checkpoint.exists():
            records[index] = json.loads(checkpoint.read_text())
        elif record["status"] != "preserve_original" and record["speaker_id"] not in references:
            records[index] = preserve(record, "insufficient_clean_voice_reference")
    if any(row["status"] == "preserve_original" and row["human_speech"] for row in records):
        require_complete_dialogue(records, output, "voice_reference", allow_pending=True)
    pending = [record for record in records if record["status"] != "preserve_original"
               and not record.get("translation_ready")]
    if pending:
        translated = {record["id"]: record for record in translate(pending, records, config, output=output)}
        records = [translated.get(record["id"], record) for record in records]
    ready = [record for record in records if record["status"] != "preserve_original"]
    synthesized = {record["id"]: record for record in synthesize(ready, references, config, output)} if ready else {}
    records = [synthesized.get(record["id"], record) for record in records]
    records = shorten_to_fit(records, references, config, output)
    records = resample_short_utterances(records, references, config, output)
    for record in records:
        save_record(output, record)
    result = assemble(records, references, plan, config, output)
    timings = result.get("timings")
    if timings is not None:
        print(f"Speech timing: translation={timings['translation_seconds']:.3f}s "
              f"({timings['translation_requests']} requests), "
              f"TTS={timings['tts_seconds']:.3f}s "
              f"({timings['tts_batches']} batches, {timings['tts_requested_utterances']} utterance attempts, "
              f"{timings['tts_cache_hits']} cache hits)", flush=True)
    return result


def run_stage(*, plan, speech_source, output_dir, source_language, target_language,
              translation_url, translation_model, tts_model, device="cuda:0", seed=42,
              duration_mode="fixed", translation_backend="lightllm",
              translation_api_key_env="LIGHTDUBBER_LLM_API_KEY", translation_options=None, tts_backend="qwen3tts",
              tts_url="", tts_api_key_env="LIGHTDUBBER_TTS_API_KEY", tts_voice="alloy",
              tts_python=sys.executable, qwen_x_vector_only_mode=True):
    request = SpeechRequest(
        plan=plan, speech_source=speech_source, output_dir=output_dir,
        source_language=source_language, target_language=target_language,
        translation_url=translation_url, translation_model=translation_model, tts_model=tts_model,
        device=device, seed=seed, duration_mode=duration_mode,
        translation_backend=translation_backend, translation_api_key_env=translation_api_key_env,
        translation_options=translation_options or {},
        tts_backend=tts_backend, tts_url=tts_url, tts_api_key_env=tts_api_key_env,
        tts_voice=tts_voice, tts_python=tts_python, qwen_x_vector_only_mode=qwen_x_vector_only_mode,
    )
    return run_speech(request)
