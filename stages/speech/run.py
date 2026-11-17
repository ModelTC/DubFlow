"""翻译原片对白，按说话人克隆音色，并在同一时间轴安排目标语言语音。"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from runtime.clients import Translator
from runtime.execution import run_python
from runtime.languages import LANGUAGES

from stages.speech.models import prepare_waveform, translation_prompt

SHORTENING_ROUNDS = 3
SHORTENING_TRIGGER_RATIO = 1.1
SHORTENING_MIN_DURATION_RATIO = .9


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def signature(path):
    path = Path(path).resolve()
    info = path.stat()
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def save_record(output, record):
    directory = output / "items"
    directory.mkdir(exist_ok=True)
    write_json(directory / f"{record['id']}.json", record)


def preserve(record, reason):
    return {**record, "status": "preserve_original", "reason": reason}


def speaking_windows(plan):
    """只借相邻静默；共享静默在中点分开，不越过其他台词或新增切镜。"""
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
        left_cut = max(shot["start_frame"] / fps for shot in plan["shots"] if shot["start_frame"] / fps <= record["start"])
        right_cut = min(shot["end_frame"] / fps for shot in plan["shots"] if shot["end_frame"] / fps >= record["end"])
        left = max(0, left_cut, record["start"] - .25)
        right = min(duration, right_cut, record["end"] + .65)
        if index:
            left = max(left, (records[index - 1]["end"] + record["start"]) / 2)
        if index + 1 < len(records):
            right = min(right, (record["end"] + records[index + 1]["start"]) / 2)
        # 核心台词可见时，借用静默不能把口型推到人物已经转身/离画之后。
        for shot in plan["shots"]:
            if not shot["start_frame"] / fps <= record["start"] < record["end"] <= shot["end_frame"] / fps:
                continue
            visible = {frame for track in plan["tracks"]
                       if track["person_id"] == record["person_id"] and track["shot_id"] == shot["id"]
                       for frame in track["frames"]} if record["person_id"] else set()
            first = max(shot["start_frame"], math.floor((record["start"] - .04) * fps))
            last = min(shot["end_frame"], math.ceil((record["end"] + .04) * fps))
            if visible and all(frame in visible for frame in range(first, last)):
                while first > shot["start_frame"] and first - 1 in visible:
                    first -= 1
                while last < shot["end_frame"] and last in visible:
                    last += 1
                left, right = max(left, first / fps + .04), min(right, last / fps - .04)
            break
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


def adaptive_shrink_candidates(records, plan):
    """Only retime isolated speech wholly inside one source shot."""
    fps = plan["fps"]
    shots = plan["shots"]
    spans = {record["id"]: (math.floor(record["start"] * fps), math.ceil(record["end"] * fps))
             for record in records}
    candidates = set()
    for record in records:
        if record["status"] == "preserve_original" or record.get("audio_only"):
            continue
        first, last = spans[record["id"]]
        if last - first < 4:
            continue
        if not any(shot["start_frame"] <= first < last <= shot["end_frame"] for shot in shots):
            continue
        if any(other_id != record["id"] and first < other_last and other_first < last
               for other_id, (other_first, other_last) in spans.items()):
            continue
        candidates.add(record["id"])
    return candidates


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
        sf.write(path, audio, rate, subtype="FLOAT")
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


def translate(records, conversation, args, shorten=False, output=None):
    translator = Translator(args.translation_url, args.translation_model)
    results = []
    try:
        for record in records:
            text = translator(translation_prompt(record, conversation, args.source_language,
                                                 args.target_language, shorten))
            round_number = record.get("shortening_round", int(shorten))
            history = [*record.get("translation_history", []),
                       {"round": round_number, "source_text": record["text"], "text": text,
                        "available_seconds": available_seconds(record),
                        "word_budget": record.get("shortening_word_budget")}]
            item = {**record, "target_text": text, "raw_model_text": text,
                    "translation_history": history, "translation_source": args.translation_model,
                    "translation_attempt": round_number, "translation_ready": True,
                    "translation_review": "disabled", "time_compression_policy": "uncapped"}
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


def synthesize(records, references, args, output, attempt=0):
    rows, items = [], []
    for index, record in enumerate(records):
        current_attempt = max(attempt, record.get("tts_attempt", 0), record.get("translation_attempt", 0))
        digest = hashlib.sha256(record["target_text"].encode()).hexdigest()[:12]
        path = output / f"qwen_{record['id']}_{current_attempt}_{digest}.wav"
        seed = args.seed + record.get("tts_seed_index", index) + current_attempt * 1000
        reference = references[record["speaker_id"]]
        rows.append({"path": str(path), "text": record["target_text"], "reference": reference["path"],
                     "reference_text": reference["text"], "seed": seed,
                     "x_vector_only_mode": args.qwen_x_vector_only_mode})
        fresh = {key: value for key, value in record.items() if not key.startswith("tts_") or key == "tts_seed_index"}
        items.append({**fresh, "tts_seed": seed, "tts_attempt": current_attempt,
                      "tts_backend": "qwen3tts", "tts_model": args.tts_model,
                      "tts_reference_mode": "x_vector_only" if args.qwen_x_vector_only_mode else "in_context",
                      "reference_speaker_id": record["speaker_id"]})
    if not rows:
        return []
    request = output / f"qwen_request_{attempt}.json"
    write_json(request, {"model_dir": str(Path(args.tts_model).resolve()),
                         "device": args.device, "language": args.target_language, "rows": rows})
    run_python(args.tts_python, "stages.speech.models", ["--qwen-request", str(request)],
               env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, check=True)
    results = []
    for item, row in zip(items, rows):
        item = prepared_tts(item, references[item["speaker_id"]], Path(row["path"]))
        save_record(output, item)
        results.append(item)
        print(f"Qwen3-TTS {item['id']}: {item['tts_duration']:.3f}s", flush=True)
    return results


def placement(record, duration_mode="fixed"):
    natural = record["tts_duration"]
    start = max(record["window_start"], record["start"])
    capacity = record["window_end"] - start
    source_duration = record["end"] - start
    if (duration_mode == "adaptive_video" and record.get("adaptive_shrink_eligible")
            and natural < source_duration - .08):
        # Render the mouth against slow speech first. Export then contracts this
        # same source interval, restoring the TTS pace while retiming the video.
        return start, natural / max(.01, source_duration - .04)
    # atempo 的块长舍入留 40ms 余量；不裁掉任何已生成的语音。
    tempo = 1.0 if natural <= capacity + 1 / 48000 else natural / max(.01, capacity - .04)
    return start, tempo


def effective_tempo(record):
    native_speed = record.get("tts_natural_duration", record["tts_duration"]) / record["tts_duration"]
    return placement(record)[1] * native_speed


def available_seconds(record):
    return max(0., record["window_end"] - max(record["window_start"], record["start"]))


def speech_attempt(record, round_number):
    result = {"round": round_number, "source_text": record["text"],
              "target_text": record.get("target_text"), "available_seconds": available_seconds(record),
              "word_budget": record.get("shortening_word_budget"), "reason": record.get("reason")}
    if record["status"] != "preserve_original":
        result.update(tts_duration=record["tts_duration"],
                      tts_natural_duration=record.get("tts_natural_duration", record["tts_duration"]),
                      effective_tempo=effective_tempo(record), tts_attempt=record["tts_attempt"],
                      tts_seed=record.get("tts_seed"), post_tempo=placement(record)[1],
                      fits=placement(record)[1] == 1)
    return result


def shortening_budget(record, language):
    text = record["target_text"]
    units = (sum(character.isalnum() for character in text) if language in ("zh", "ja", "ko")
             else len(re.findall(r"\b[^\W\d_]+(?:['’][^\W\d_]+)*\b", text)))
    duration = record["tts_duration"]
    target = .95 * available_seconds(record)
    estimate = max(1, round(units * target / max(duration, .01)))
    if duration > available_seconds(record):
        return min(max(1, units - 1), estimate)
    return max(units + 1, estimate)


def shorten_to_fit(records, references, args, output):
    """Use measured TTS duration for up to three revisions, keeping the best dub."""
    if getattr(args, "duration_mode", "fixed") == "adaptive_video":
        return records  # The picture follows natural speech; do not shorten the translation as well.
    best, probe, attempts, histories = {}, {}, {}, {}
    for record in records:
        item = dict(record)
        if item["status"] != "preserve_original" and not item.get("speech_attempts"):
            item["speech_attempts"] = [speech_attempt(item, 0)]
        key = item["id"]
        best[key] = probe[key] = item
        attempts[key] = item.get("speech_attempts", [])
        histories[key] = item.get("translation_history", [])
        completed = item.get("shortening_rounds_completed", 0)
        if completed:
            checkpoint = output / f"shortening_round_{completed}" / "items" / f"{key}.json"
            if checkpoint.is_file():
                previous_probe = json.loads(checkpoint.read_text())
                if previous_probe["status"] != "preserve_original" and "tts_duration" in previous_probe:
                    probe[key] = previous_probe

    for round_number in range(1, SHORTENING_ROUNDS + 1):
        pending = []
        for record in records:
            key = record["id"]
            selected = best[key]
            if selected["status"] == "preserve_original":
                continue
            completed = selected.get("shortening_rounds_completed", 0)
            capacity = available_seconds(selected)
            duration = selected["tts_duration"]
            if (completed >= round_number or capacity <= 0 or
                    (not completed and duration <= SHORTENING_TRIGGER_RATIO * capacity) or
                    SHORTENING_MIN_DURATION_RATIO * capacity <= duration <= capacity):
                continue
            previous_probe = probe[key]
            pending.append({**previous_probe, "translation_history": histories[key],
                            "speech_attempts": attempts[key], "shortening_round": round_number,
                            "available_seconds": capacity,
                            "shortening_word_budget": shortening_budget(previous_probe, args.target_language)})
        if not pending:
            continue
        round_output = output / f"shortening_round_{round_number}"
        round_output.mkdir(exist_ok=True)
        shortened = translate(pending, records, args, shorten=True, output=round_output)
        for candidate in synthesize(shortened, references, args, round_output, attempt=round_number):
            key = candidate["id"]
            selected = best[key]
            capacity = available_seconds(selected)
            histories[key] = candidate["translation_history"]
            attempts[key] = [*attempts[key], speech_attempt(candidate, round_number)]
            probe[key] = candidate
            if (candidate["tts_duration"] >= SHORTENING_MIN_DURATION_RATIO * capacity
                    and abs(candidate["tts_duration"] - capacity) < abs(selected["tts_duration"] - capacity)):
                selected = candidate
            selected = {**selected, "translation_history": histories[key], "speech_attempts": attempts[key],
                        "shortening_rounds_completed": round_number,
                        "shortening_word_budget": candidate["shortening_word_budget"]}
            best[key] = selected
            save_record(output, selected)
    return [best[record["id"]] for record in records]


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


def align_record(record, output, duration_mode="fixed"):
    start, tempo = placement(record, duration_mode)
    path = output / f"aligned_{record['id']}.wav"
    settings = {"source": signature(record["tts_raw_path"]), "start": start, "tempo": tempo}
    meta = path.with_suffix(".json")
    if not (meta.is_file() and json.loads(meta.read_text()) == settings and path.is_file()):
        temporary = path.with_suffix(".tmp.wav")
        command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", record["tts_raw_path"]]
        if tempo != 1:
            command.extend(["-af", atempo_filter(tempo)])
        subprocess.run(command + ["-ar", "48000", "-ac", "2", "-c:a", "pcm_f32le", str(temporary)], check=True)
        temporary.replace(path)
        write_json(meta, settings)
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    first, last = round(start * rate), round(start * rate) + len(audio)
    if last > round(record["window_end"] * rate) or not np.isfinite(audio).all():
        raise ValueError(f"Retimed speech is invalid or exceeds its window: {record['id']}")
    row = {**record, "status": "translated", "reason": None, "start": first / rate, "end": last / rate,
           "source_interval": {"start": record["start"], "end": record["end"]},
           "allowed_interval": {"start": record["window_start"], "end": record["window_end"]},
           "tts_aligned_path": str(path), "tempo_ratio": tempo,
           "adaptive_shortening_prepared": duration_mode == "adaptive_video" and tempo < 1}
    natural = record.get("tts_natural_duration", record["tts_duration"])
    row.update(max_tempo=None, time_compression={"natural_seconds": natural,
               "native_duration_factor": record.get("tts_native_duration_factor", 1.),
               "native_speed": natural / record["tts_duration"], "post_speed": tempo,
               "total_speed": natural / record["tts_duration"] * tempo,
               "output_seconds": len(audio) / rate})
    return row, audio


def assemble(records, references, plan, args, output):
    duration = plan["frames"] / plan["fps"]
    samples = round(duration * 48000)
    tracks = {speaker: np.zeros((samples, 2), np.float32) for speaker in references}
    utterances = []
    for record in records:
        if record["status"] == "preserve_original":
            row, aligned = record, None
        else:
            row, aligned = align_record(record, output, getattr(args, "duration_mode", "fixed"))
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
    result = {"source_plan": str(Path(args.plan).resolve()), "source_speech": str(Path(args.speech_source).resolve()),
              "source_language": args.source_language, "target_language": args.target_language,
              "duration": duration, "sample_rate": 48000, "global_gain": gain, "seed": args.seed,
              "utterances": utterances, "target_speech_path": str(target_path),
              "speaker_tracks": speaker_paths, "voice_references": references,
              "replace_intervals": [{"id": row["id"], **row["source_interval"]} for row in successful],
              "counts": {"translated": len(successful), "preserve_original": len(utterances) - len(successful)},
              "human_review_required": False}
    result.update(audio_policy="complete_dialogue", audio_complete=True)
    (output / "source_text.txt").write_text("\n".join(f"{row['id']} {row['text']}" for row in utterances) + "\n")
    (output / "translated.txt").write_text("\n".join(f"{row['id']} {row['target_text']}" for row in successful) + "\n")
    write_json(output / "dialogue.json", result)
    return result


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    duration_mode = getattr(args, "duration_mode", "fixed")
    request = {"plan": signature(args.plan), "source_speech": signature(args.speech_source),
               "source_language": args.source_language,
               "target_language": args.target_language, "duration_mode": duration_mode,
               "seed": args.seed, "recipe": 10}
    request["backends"] = {
        "translation_url": args.translation_url, "translation_model": args.translation_model,
        "tts_backend": "qwen3tts", "tts_model": str(Path(args.tts_model).resolve()),
        "tts_python": args.tts_python, "qwen_x_vector_only_mode": args.qwen_x_vector_only_mode,
        "speech_adapter": signature(Path(__file__)),
        "adapter": signature(Path(__file__).with_name("models.py")),
    }
    request["backends"].update(translation_review="disabled", shortening_rounds=0 if duration_mode == "adaptive_video" else SHORTENING_ROUNDS,
                               time_compression_policy="uncapped", max_tempo=None, audio_policy="complete_dialogue")
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("Speech output belongs to another request; choose a new directory")
        if (output / "dialogue.json").exists():
            return json.loads((output / "dialogue.json").read_text())
    else:
        if any(output.iterdir()):
            raise FileExistsError(output)
        write_json(request_path, request)
    plan = json.loads(Path(args.plan).read_text())
    records = speaking_windows(plan)
    if duration_mode == "adaptive_video":
        eligible = adaptive_shrink_candidates(records, plan)
        records = [{**record, "adaptive_shrink_eligible": record["id"] in eligible} for record in records]
    audio, rate = sf.read(args.speech_source, dtype="float32", always_2d=True)
    if rate != 48000 or len(audio) != round(plan["frames"] / plan["fps"] * rate):
        raise ValueError("Source speech must share the plan's 48kHz timeline")
    references = voice_references(records, audio.mean(1), rate, output, speaker_bindings=plan.get("speakers"))
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
        translated = {record["id"]: record for record in translate(pending, records, args, output=output)}
        records = [translated.get(record["id"], record) for record in records]
    ready = [record for record in records if record["status"] != "preserve_original"
             and not (record.get("tts_raw_path") and Path(record["tts_raw_path"]).is_file())]
    synthesized = {record["id"]: record for record in synthesize(ready, references, args, output)} if ready else {}
    records = [synthesized.get(record["id"], record) for record in records]
    records = shorten_to_fit(records, references, args, output)
    for record in records:
        save_record(output, record)
    return assemble(records, references, plan, args, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--speech-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-language", choices=LANGUAGES, default="zh")
    parser.add_argument("--target-language", choices=LANGUAGES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--translation-url", required=True, help="OpenAI 兼容翻译接口地址，包含 /v1")
    parser.add_argument("--translation-model", required=True, help="翻译服务中的模型名称")
    parser.add_argument("--tts-model", required=True, help="本地 TTS 权重目录")
    parser.add_argument("--tts-python", default=sys.executable, help="TTS 解释器")
    parser.add_argument("--qwen-x-vector-only-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--duration-mode", choices=("fixed", "adaptive_video"), default="fixed")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"dialogue": str(Path(args.output_dir).resolve() / "dialogue.json"), **result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
