"""用保留下来的对白 ID 直接规划口型，不重新 ASR 或做跨语言语义匹配。"""

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np

from stages.lipsync.timeline import plan_jobs, prepared_audio, roi_records, write_drive

VISUAL_WINDOW_POLICY = "continuous_visible_track_v1"
MAX_VISUAL_WINDOW_SECONDS = 15.0


def read_track(path, frames, fps):
    with wave.open(str(path)) as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes()) != (16000, 1, 2, round(frames / fps * 16000)):
            raise ValueError(f"Speaker track does not share the source timeline: {path}")
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")


def protect_original_frames(jobs, source, accepted):
    protected = [row for row in source["utterances"] if row["id"] not in accepted]
    for job in jobs:
        if job["action"] != "tbdub":
            continue
        start = min(job["source_interval"]["start"], job["target_interval"]["start"])
        end = max(job["source_interval"]["end"], job["target_interval"]["end"])
        for row in protected:
            if row["end"] <= start:
                job["edit_start_frame"] = max(job["edit_start_frame"], math.ceil(row["end"] * source["fps"]))
            elif row["start"] >= end:
                job["edit_end_frame"] = min(job["edit_end_frame"], math.floor(row["start"] * source["fps"]))
        if job["edit_start_frame"] >= job["edit_end_frame"]:
            job.update(action="audio_only", reason="no_safe_edit_interval")
        shot = next(shot for shot in source["shots"] if shot["id"] == job["shot_id"])
        job["edit_starts_at_shot_cut"] = job["edit_start_frame"] == shot["start_frame"]
        job["edit_ends_at_shot_cut"] = job["edit_end_frame"] == shot["end_frame"]


def continuous_visual_jobs(jobs, source, accepted):
    """Keep one generated appearance across a visible person's silent pauses.

    Shot cuts, unsupported face gaps and preserved original dialogue are hard
    boundaries. Driving audio remains independently gated to accepted speech.
    """
    fps = source["fps"]
    shots = {shot["id"]: shot for shot in source["shots"]}
    protected = [(math.floor(row["start"] * fps), math.ceil(row["end"] * fps))
                 for row in source["utterances"] if row["id"] not in accepted]
    bindings = {}
    for job in jobs:
        if job["action"] == "tbdub":
            bindings.setdefault((job["shot_id"], job["person_id"]), set()).add(
                (job["speaker_id"], job["roi_mode"]))
    components, oversized, merged, result = {}, {}, {}, []

    def bounded_fallback(job, blocked, reason):
        fallback = {**job, "visual_window_fallback": reason,
                    "visual_window_limit_seconds": MAX_VISUAL_WINDOW_SECONDS}
        if any(max(first, job["edit_start_frame"]) < min(last, job["edit_end_frame"])
               for first, last in blocked):
            fallback.update(action="audio_only", reason="protected_dialogue_overlap")
        return fallback

    for job in jobs:
        identity = (job["shot_id"], job["person_id"])
        if job["action"] != "tbdub":
            result.append(job)
            continue
        blocked = list(protected)
        for other in jobs:
            if other["action"] != "audio_only" or (other["shot_id"], other["person_id"]) != identity:
                continue
            for interval in (other["source_interval"], other.get("target_interval")):
                if interval is None:
                    continue
                first, last = math.floor(interval["start"] * fps), math.ceil(interval["end"] * fps)
                if other["reason"] == "missing_core_face_track":
                    first, last = max(first, other["start_frame"]), min(last, other["end_frame"])
                blocked.append((first, last))
        # Conflicting speaker/ROI bindings must not expand two drives over the
        # same face. Retain the existing bounded plan for that ambiguous track.
        if len(bindings[identity]) != 1:
            result.append(bounded_fallback(job, blocked, "conflicting_visual_bindings"))
            continue
        shot = shots[job["shot_id"]]
        if identity not in components:
            runs = []
            first, last = shot["start_frame"], shot["end_frame"]
            limit = math.floor(MAX_VISUAL_WINDOW_SECONDS * fps)
            if last - first > limit:
                related = [other for other in jobs if other["action"] == "tbdub" and
                           (other["shot_id"], other["person_id"]) == identity]
                first = max(first, min(other["start_frame"] for other in related))
                last = min(last, max(other["end_frame"] for other in related))
            # Bound the ROI query itself as well as the later VAE decode/PNGs.
            # Oversized groups keep the previous short-job policy, explicitly.
            oversized[identity] = last - first > limit
            records = [] if oversized[identity] else roi_records(
                source, job["person_id"], job["shot_id"], first, last)
            for record in records:
                frame = record["frame"]
                if any(first <= frame < last for first, last in blocked):
                    continue
                if not runs or frame != runs[-1][-1]["frame"] + 1:
                    runs.append([])
                runs[-1].append(record)
            components[identity] = runs
        if oversized[identity]:
            result.append(bounded_fallback(job, blocked, "continuous_window_exceeds_limit"))
            continue
        speech = job.get("source_speech_intervals", [job["source_interval"]]) + job["target_speech_intervals"]
        found = False
        for records in components[identity]:
            first, last = records[0]["frame"], records[-1]["frame"] + 1
            # Do not activate another visible component using only silent
            # context, or a different shot's portion of a shared utterance.
            if (max(first, job["edit_start_frame"]) >= min(last, job["edit_end_frame"]) or
                    not any(math.floor(part["start"] * fps) < last and
                            math.ceil(part["end"] * fps) > first for part in speech)):
                continue
            found = True
            key = (*identity, job["speaker_id"], job["roi_mode"], first, last)
            previous = merged.get(key)
            if previous is None:
                previous = {**job, "utterance_ids": list(job["utterance_ids"]),
                            "source_speech_intervals": list(job.get("source_speech_intervals", [job["source_interval"]])),
                            "target_speech_intervals": list(job["target_speech_intervals"]),
                            "start_frame": first, "end_frame": last,
                            "edit_start_frame": first, "edit_end_frame": last,
                            "edit_starts_at_shot_cut": first == shot["start_frame"],
                            "edit_ends_at_shot_cut": last == shot["end_frame"],
                            "roi_records": records, "temporal_policy": VISUAL_WINDOW_POLICY}
                merged[key] = previous
                result.append(previous)
                continue
            previous["utterance_ids"] = list(dict.fromkeys(previous["utterance_ids"] + job["utterance_ids"]))
            for name in ("source", "target"):
                previous[f"{name}_interval"] = {
                    "start": min(previous[f"{name}_interval"]["start"], job[f"{name}_interval"]["start"]),
                    "end": max(previous[f"{name}_interval"]["end"], job[f"{name}_interval"]["end"])}
                previous[f"{name}_speech_intervals"] += job.get(f"{name}_speech_intervals", [job[f"{name}_interval"]])
                previous[f"{name}_text"] += " " + job[f"{name}_text"]
        if not found:
            result.append({**job, "action": "audio_only", "reason": "no_safe_continuous_track"})
    for index, job in enumerate(result):
        job["id"] = f"job_{index:03d}"
    return result


def build_plan(args):
    source_path, dialogue_path = Path(args.plan).resolve(), Path(args.dialogue).resolve()
    source, dialogue = json.loads(source_path.read_text()), json.loads(dialogue_path.read_text())
    if Path(dialogue["source_plan"]).resolve() != source_path:
        raise ValueError("Translated dialogue belongs to another source plan")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "render_plan.json"
    if plan_path.exists():
        raise FileExistsError(plan_path)
    originals = {row["id"]: row for row in source["utterances"]}
    translated = {row["id"]: row for row in dialogue["utterances"]}
    if len(translated) != len(dialogue["utterances"]) or set(translated) != set(originals):
        raise ValueError("Translation must preserve every source utterance ID exactly once")
    duration = source["frames"] / source["fps"]
    audio_manifest = json.loads(Path(args.audio_manifest).read_text())
    accepted = set(audio_manifest["metadata"]["accepted_replacement_ids"])
    skipped = {row["id"]: row["reason"] for row in audio_manifest["metadata"]["skipped_replacements"]}
    gain = audio_manifest["metadata"]["target_speech_gain"]
    targets, matches = [], []
    selected_rows = []
    for index, original in enumerate(source["utterances"]):
        row = translated[original["id"]]
        source_speaker = row.get("source_speaker_id", row["speaker_id"])
        if source_speaker != original["speaker_id"] or row["person_id"] != original["person_id"]:
            raise ValueError(f"Translation changed the speaker binding: {row['id']}")
        if row["id"] not in accepted:
            continue
        if row["status"] != "translated" or not row["human_speech"]:
            raise ValueError(f"Mix accepted a non-translated utterance: {row['id']}")
        if not 0 <= row["start"] < row["end"] <= duration:
            raise ValueError(f"Translated speech is outside the video: {row['id']}")
        if not row.get("audio_only"):
            matches.append({"source_index": index, "target_indices": [len(targets)]})
        targets.append({"start": row["start"], "end": row["end"], "text": row["target_text"]})
        selected_rows.append(row)
    final_audio, speech_path, _ = prepared_audio(args.audio_manifest, dialogue["target_speech_path"], duration)
    jobs = plan_jobs(source, targets, matches, args.context_seconds, args.edit_context_seconds)
    for job in jobs:
        if translated[job["utterance_id"]].get("audio_only") and job["utterance_id"] in accepted:
            row = translated[job["utterance_id"]]
            job.update(action="audio_only", reason="unresolved_speaker_audio_translated",
                       target_interval={"start": row["start"], "end": row["end"]},
                       target_speech_intervals=[{"start": row["start"], "end": row["end"]}],
                       target_text=row["target_text"])
    visual_accepted = accepted - {row["id"] for row in selected_rows if row.get("audio_only")}
    protect_original_frames(jobs, source, visual_accepted)
    jobs = continuous_visual_jobs(jobs, source, accepted)
    tracks = {}
    for speaker in {row["speaker_id"] for row in selected_rows}:
        raw = read_track(dialogue["speaker_tracks"][speaker], source["frames"], source["fps"])
        clean = np.zeros_like(raw)
        for row in selected_rows:
            if row["speaker_id"] == speaker:
                first, last = round(row["start"] * 16000), round(row["end"] * 16000)
                clean[first:last] = np.clip(np.rint(raw[first:last].astype(np.float32) * gain), -32768, 32767).astype(np.int16)
        tracks[speaker] = clean
    drives = output / "drives"
    drives.mkdir(exist_ok=True)
    for job in jobs:
        if job["action"] == "audio_only":
            job["drive_audio_path"] = None
            if job["utterance_id"] not in accepted:
                job["reason"] = skipped.get(job["utterance_id"]) or translated[job["utterance_id"]].get("reason") or "preserve_original_audio"
            continue
        path = drives / f"{job['id']}.wav"
        temporary = path.with_suffix(".partial.wav")
        if temporary.exists():
            temporary.unlink()
        write_drive(temporary, tracks[job["speaker_id"]], job, source["fps"])
        temporary.replace(path)
        job["drive_audio_path"] = str(path)
    result = {"source_plan": str(source_path), "dialogue": str(dialogue_path),
              "source_video": source["source_video"], "target_video": dialogue["target_speech_path"],
              "audio_manifest": str(Path(args.audio_manifest).resolve()), "target_audio_path": str(final_audio),
              "speech_drive_source": str(speech_path), "fps": source["fps"], "frames": source["frames"],
              "matching": {"method": "preserved_utterance_ids", "accepted_replacement_ids": sorted(accepted), "context_seconds": args.context_seconds,
                           "edit_context_seconds": args.edit_context_seconds,
                           "visual_window_policy": VISUAL_WINDOW_POLICY,
                           "max_visual_window_seconds": MAX_VISUAL_WINDOW_SECONDS}, "jobs": jobs}
    with plan_path.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"Direct ID mapping: {len(originals)} utterances, {len(jobs)} render jobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--dialogue", required=True)
    parser.add_argument("--audio-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-seconds", type=float, default=0.7)
    parser.add_argument("--edit-context-seconds", type=float, default=0.6)
    build_plan(parser.parse_args())
