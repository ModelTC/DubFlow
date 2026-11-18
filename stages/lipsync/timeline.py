"""同一时间轴的口型任务、人物裁框和配音驱动辅助函数。"""

import json
import math
import wave
from pathlib import Path

import numpy as np

SPEECH_GUARD = 0.04


def roi_mode(plan, person, shot_id, track_ids=()):
    if person is None:
        return "landmarks"
    bound_tracks = [track for track in plan["tracks"]
                    if track["person_id"] == person and track["shot_id"] == shot_id]
    matching_tracks = [track for track in bound_tracks if track["id"] in track_ids]
    tracks = matching_tracks or bound_tracks
    if tracks and all(track.get("identity_quality", {}).get("match_eligible") is False for track in tracks):
        return "track_bbox"
    return "landmarks"


def roi_records(plan, person, shot_id, first, last):
    """只使用真实检测；最多插值同人同镜头内的 10 帧缺口，不外推丢失的人脸。"""
    if person is None:
        return []
    observed = {}
    for track in plan["tracks"]:
        if track["person_id"] != person or track["shot_id"] != shot_id:
            continue
        for frame, box, confidence in zip(track["frames"], track["bboxes"], track["confidence"]):
            if frame not in observed or confidence > observed[frame]["confidence"]:
                observed[frame] = {"frame": frame, "bbox": list(map(float, box)), "observed": True,
                                   "track_id": track["id"], "confidence": float(confidence)}
    frames = sorted(observed)
    records = []
    for frame in range(first, last):
        if frame in observed:
            records.append(observed[frame])
            continue
        right = int(np.searchsorted(frames, frame))
        if right == 0 or right == len(frames):
            continue
        left_frame, right_frame = frames[right - 1], frames[right]
        if right_frame - left_frame - 1 <= 10:
            fraction = (frame - left_frame) / (right_frame - left_frame)
            box = (1 - fraction) * np.asarray(observed[left_frame]["bbox"]) + fraction * np.asarray(observed[right_frame]["bbox"])
            records.append({"frame": frame, "bbox": box.tolist(), "observed": False,
                            "track_id": observed[left_frame]["track_id"]})
    return records


def supported_context(records, core_first, core_last, first, last):
    by_frame = {record["frame"]: record for record in records}
    if not all(frame in by_frame for frame in range(core_first, core_last)):
        return first, last, records
    left, right = core_first, core_last
    while left > first and left - 1 in by_frame:
        left -= 1
    while right < last and right in by_frame:
        right += 1
    return left, right, [by_frame[frame] for frame in range(left, right)]


def split_supported_jobs(job, shot, fps):
    """Keep observed/interpolated runs; leave unsupported speech frames untouched."""
    if job["reason"] != "missing_core_face_track":
        return [job]
    runs = []
    for record in job["roi_records"]:
        if not runs or record["frame"] != runs[-1][-1]["frame"] + 1:
            runs.append([])
        runs[-1].append(record)
    regions, cursor = [], job["start_frame"]
    for records in runs:
        first, last = records[0]["frame"], records[-1]["frame"] + 1
        if cursor < first:
            regions.append((cursor, first, []))
        regions.append((first, last, records))
        cursor = last
    if cursor < job["end_frame"]:
        regions.append((cursor, job["end_frame"], []))

    def clipped(interval, first, last):
        if interval is None:
            return None
        start, end = max(interval["start"], first / fps), min(interval["end"], last / fps)
        return {"start": start, "end": end} if start < end else None

    result = []
    for first, last, records in regions:
        speech = {name: [part for interval in job[f"{name}_speech_intervals"]
                         if (part := clipped(interval, first, last)) is not None]
                  for name in ("source", "target")}
        if not speech["source"] and not speech["target"]:
            continue
        edit_first, edit_last = max(first, job["edit_start_frame"]), min(last, job["edit_end_frame"])
        part = {**job, "start_frame": first, "end_frame": last,
                "edit_start_frame": edit_first, "edit_end_frame": edit_last,
                "edit_starts_at_shot_cut": edit_first == shot["start_frame"],
                "edit_ends_at_shot_cut": edit_last == shot["end_frame"], "roi_records": records}
        if records:
            part.update(action="tbdub", reason="ready")
        else:
            # Only the missing span may block another supported visual job.
            for name in ("source", "target"):
                part[f"{name}_interval"] = clipped(job[f"{name}_interval"], first, last)
                part[f"{name}_speech_intervals"] = speech[name]
        result.append(part)
    return result or [job]


def merge_adjacent_jobs(plan, jobs):
    merged, fps = [], plan["fps"]
    shots = {shot["id"]: shot for shot in plan["shots"]}
    for job in jobs:
        previous = merged[-1] if merged else None
        if (previous is None or previous["action"] != "tbdub" or job["action"] != "tbdub" or
                any(previous[key] != job[key] for key in ("shot_id", "person_id", "speaker_id", "roi_mode")) or
                max(previous["edit_start_frame"], job["edit_start_frame"]) >=
                min(previous["edit_end_frame"], job["edit_end_frame"])):
            merged.append(job)
            continue
        source = {"start": min(previous["source_interval"]["start"], job["source_interval"]["start"]),
                  "end": max(previous["source_interval"]["end"], job["source_interval"]["end"])}
        target = {"start": min(previous["target_interval"]["start"], job["target_interval"]["start"]),
                  "end": max(previous["target_interval"]["end"], job["target_interval"]["end"])}
        first, last = min(previous["start_frame"], job["start_frame"]), max(previous["end_frame"], job["end_frame"])
        records = roi_records(plan, job["person_id"], job["shot_id"], first, last)
        shot = shots[job["shot_id"]]
        core_first = max(shot["start_frame"], math.floor((min(source["start"], target["start"]) - SPEECH_GUARD) * fps))
        core_last = min(shot["end_frame"], math.ceil((max(source["end"], target["end"]) + SPEECH_GUARD) * fps))
        left, right, records = supported_context(records, core_first, core_last, first, last)
        if (left, right) != (first, last) or len(records) != last - first:
            merged.append(job)
            continue
        previous.update(start_frame=first, end_frame=last, roi_records=records,
                        edit_start_frame=min(previous["edit_start_frame"], job["edit_start_frame"]),
                        edit_end_frame=max(previous["edit_end_frame"], job["edit_end_frame"]),
                        edit_starts_at_shot_cut=previous["edit_starts_at_shot_cut"] or job["edit_starts_at_shot_cut"],
                        edit_ends_at_shot_cut=previous["edit_ends_at_shot_cut"] or job["edit_ends_at_shot_cut"],
                        source_interval=source, target_interval=target,
                        source_speech_intervals=(previous.get("source_speech_intervals", [previous["source_interval"]])
                                                 + job.get("source_speech_intervals", [job["source_interval"]])),
                        source_text=f"{previous['source_text']} {job['source_text']}",
                        target_text=f"{previous['target_text']} {job['target_text']}",
                        utterance_ids=previous["utterance_ids"] + job["utterance_ids"],
                        target_speech_intervals=previous["target_speech_intervals"] + job["target_speech_intervals"])
    for index, job in enumerate(merged):
        job["id"] = f"job_{index:03d}"
    return merged


def plan_jobs(plan, targets, matches, context, edit_context):
    if not 0 <= edit_context <= context:
        raise ValueError("Require 0 <= edit context <= model context")
    by_source = {match["source_index"]: match for match in matches}
    jobs, fps = [], plan["fps"]
    for index, source in enumerate(plan["utterances"]):
        match = by_source.get(index)
        selected = [] if match is None else [targets[number] for number in match["target_indices"]]
        speech_start = min([source["start"], *[segment["start"] for segment in selected]])
        speech_end = max([source["end"], *[segment["end"] for segment in selected]])
        core_start = math.floor((speech_start - SPEECH_GUARD) * fps)
        core_end = math.ceil((speech_end + SPEECH_GUARD) * fps)
        for shot in plan["shots"]:
            if math.ceil(speech_end * fps) <= shot["start_frame"] or math.floor(speech_start * fps) >= shot["end_frame"]:
                continue
            first = max(shot["start_frame"], math.floor((speech_start - context) * fps))
            last = min(shot["end_frame"], math.ceil((speech_end + context) * fps))
            records = roi_records(plan, source["person_id"], shot["id"], first, last)
            first, last, records = supported_context(records, max(core_start, shot["start_frame"]),
                                                     min(core_end, shot["end_frame"]), first, last)
            edit_first = max(first, math.floor((speech_start - edit_context) * fps))
            edit_last = min(last, math.ceil((speech_end + edit_context) * fps))
            if match is None:
                reason = "no_target_match"
            elif source["person_id"] is None:
                reason = "unbound_speaker"
            elif len(records) != last - first:
                reason = "missing_core_face_track"
            else:
                reason = "ready"
            job = {"id": f"job_{len(jobs):03d}", "utterance_id": source["id"], "utterance_ids": [source["id"]],
                         "person_id": source["person_id"], "speaker_id": source["speaker_id"], "shot_id": shot["id"],
                         "roi_mode": roi_mode(plan, source["person_id"], shot["id"], source.get("track_ids") or ()),
                         "action": "tbdub" if reason == "ready" else "audio_only", "reason": reason,
                         "start_frame": first, "end_frame": last,
                         "edit_start_frame": edit_first, "edit_end_frame": edit_last,
                         "edit_starts_at_shot_cut": edit_first == shot["start_frame"],
                         "edit_ends_at_shot_cut": edit_last == shot["end_frame"],
                         "source_text": source["text"], "source_interval": {"start": source["start"], "end": source["end"]},
                         "source_speech_intervals": [{"start": part["start"], "end": part["end"]}
                                                     for part in source.get("source_parts", [source])],
                         "target_text": " ".join(segment["text"].strip() for segment in selected),
                         "target_interval": None if not selected else {"start": selected[0]["start"], "end": selected[-1]["end"]},
                         "target_speech_intervals": [{"start": segment["start"], "end": segment["end"]} for segment in selected],
                         "roi_records": records, "drive_guard_seconds": SPEECH_GUARD}
            jobs.extend(split_supported_jobs(job, shot, fps))
    return merge_adjacent_jobs(plan, jobs)


def prepared_audio(manifest_path, target_video, duration):
    manifest = json.loads(Path(manifest_path).read_text())
    if Path(manifest["target_video"]).resolve() != Path(target_video).resolve():
        raise ValueError("Audio manifest belongs to another target video/audio")
    mixed_path, speech_path = Path(manifest["mixed_audio_path"]), Path(manifest["speech_drive_path"])
    with wave.open(str(mixed_path)) as mixed:
        if (mixed.getframerate(), mixed.getnchannels(), mixed.getsampwidth(), mixed.getnframes()) != (48000, 2, 2, round(duration * 48000)):
            raise ValueError("Final mix must be timeline-aligned 48kHz stereo PCM16")
    with wave.open(str(speech_path)) as speech:
        if (speech.getframerate(), speech.getnchannels(), speech.getsampwidth(), speech.getnframes()) != (16000, 1, 2, round(duration * 16000)):
            raise ValueError("Speech drive must be timeline-aligned 16kHz mono PCM16")
        mono = np.frombuffer(speech.readframes(speech.getnframes()), dtype="<i2")
    return mixed_path.resolve(), speech_path.resolve(), mono


def write_drive(path, mono, job, fps):
    first = round(job["start_frame"] * 16000 / fps)
    last = round(job["end_frame"] * 16000 / fps)
    drive = np.zeros(last - first, dtype=np.int16)
    for interval in job["target_speech_intervals"]:
        left = max(first, round((interval["start"] - SPEECH_GUARD) * 16000))
        right = min(last, round((interval["end"] + SPEECH_GUARD) * 16000), len(mono))
        if right > left:
            drive[left - first:right - first] = mono[left:right]
    with path.open("xb") as output, wave.open(output, "wb") as waveform:
        waveform.setnchannels(1)
        waveform.setsampwidth(2)
        waveform.setframerate(16000)
        waveform.writeframes(drive.astype("<i2").tobytes())
