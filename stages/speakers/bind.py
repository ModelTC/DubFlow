"""把声学说话人、跨镜头人脸和原声 ASD 证据关联起来；分数不是准确率。"""

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

SPEAKER_TURN_MERGE_POLICY = "continuous_same_speaker_v2"
VISUAL_BINDING_POLICY = "word_anchored_visual_v7_primary_turns"
NONLINGUISTIC_CALL = re.compile(
    r"(?:汪{2,}|喵{2,}|咩{2,}|呱{2,}|嘎{2,}|嗷{2,}|哈{2,}|呵{2,}|咳{2,}|"
    r"(?:woof){2,}|(?:meow){2,}|(?:ha){2,}|(?:cough){2,})"
)


def pure_vocalization(text):
    # Match the entire turn: a speaking animal may also utter real words.
    compact = "".join(character for character in unicodedata.normalize("NFKC", text).casefold()
                      if character.isalnum())
    return bool(NONLINGUISTIC_CALL.fullmatch(compact))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def identity_map(identity):
    return identity["track_to_person"], {track["id"]: track for track in identity["tracks"]}


def frame_evidence(vision, identity):
    mapping, _ = identity_map(identity)
    result = defaultdict(dict)
    for track in vision["tracks"]:
        person = mapping[track["id"]]
        scores = track["asd_scores"]
        for frame, score in zip(track["frames"], scores):
            if score is not None and math.isfinite(score):
                # Duplicate detections of one identity do not get extra votes.
                result[person][frame] = max(float(score), result[person].get(frame, -math.inf))
    return result


def evidence_for_turn(turn, frames, fps):
    first = math.floor(turn["start"] * fps)
    last = math.ceil(turn["end"] * fps)
    width = max(1, last - first)
    candidates = []
    for person, scores in frames.items():
        samples = [score for frame, score in scores.items() if first <= frame < last]
        if not samples:
            continue
        mean = statistics.fmean(samples)
        positive = sum(score > 0 for score in samples) / len(samples)
        coverage = len(samples) / width
        support = max(0.0, mean) * positive * coverage
        candidates.append({"person_id": person, "mean_asd_logit": mean,
                           "positive_fraction": positive, "frame_coverage": coverage,
                           "support": support, "scored_frames": len(samples)})
    return sorted(candidates, key=lambda item: (-item["support"], item["person_id"]))


def human_speech_evidence(turn, mode="speaker_visual"):
    """Qwen 的语言文本参与翻译；标注和纯叫声留在原音轨。"""
    text = turn["text"].strip()
    labels = {"music", "bgm", "background music", "laughter", "applause", "silence", "noise",
              "barking", "dog barking", "音乐", "背景音乐", "笑声", "掌声", "狗叫", "犬吠", "静音", "音效"}
    marked = re.fullmatch(r"(?:\s*[\[（(【][^\]）)】]+[\]）)】]\s*)+", text)
    annotations = re.findall(r"[\[（(【]([^\]）)】]+)[\]）)】]", text)
    non_linguistic = marked and all(annotation.lower().strip() in labels for annotation in annotations)
    if not any(character.isalnum() for character in text) or non_linguistic or "♪" in text or "♫" in text:
        return False, "non_linguistic_annotation"
    if pure_vocalization(text):
        return False, "nonlinguistic_vocalization"
    return True, "linguistic_qwen_asr_no_logprob"


def speech_turns(audio):
    """Split an ASR segment on acoustic speaker changes, retaining overlaps."""
    result = []
    for segment in audio["segments"]:
        words = segment["words"]
        timed = [word for word in words if word.get("start") is not None and word.get("end") is not None]
        # Secondary overlap labels may flicker inside a word or clause. They
        # still prohibit unsafe lip edits, but are not new translation turns.
        keys = [word.get("speaker_id") or tuple(sorted(word["speaker_ids"])) for word in timed]
        if not timed or len(set(keys)) <= 1:
            result.append(deepcopy(segment))
            continue
        groups = []
        for word, key in zip(timed, keys):
            if not groups or groups[-1][0] != key:
                groups.append((key, []))
            groups[-1][1].append(word)
        for key, group in groups:
            speakers = sorted({speaker for word in group for speaker in word["speaker_ids"]})
            primary = key if isinstance(key, str) else speakers[0] if len(speakers) == 1 else None
            text = words_text(group)
            result.append({"start": group[0]["start"], "end": group[-1]["end"], "text": text,
                           "speaker_id": primary, "speaker_ids": speakers, "words": deepcopy(group),
                           "source_asr_text": segment["text"],
                           "asr_confidence": segment.get("asr_confidence", {}),
                           "untimed_words": deepcopy([word for word in words if word not in timed])})
    return result


def assess_audio(audio, frames, fps, mode="speaker_visual"):
    votes = defaultdict(lambda: defaultdict(float))
    turns = []
    speech_duration = supported_duration = 0.0
    for segment in speech_turns(audio):
        turn = deepcopy(segment)
        turn["human_speech"], turn["human_speech_reason"] = human_speech_evidence(turn, mode)
        speakers = turn["speaker_ids"]
        turn["evidence"] = evidence_for_turn(turn, frames, fps)
        duration = max(0.0, turn["end"] - turn["start"])
        speech_duration += duration if turn["human_speech"] else 0
        # 重叠对白需要单人音轨，不能把整条混音交给一张脸。
        if turn["human_speech"] and len(speakers) == 1:
            supported_fraction = max((candidate["frame_coverage"] * candidate["positive_fraction"]
                                      for candidate in turn["evidence"] if candidate["support"] > 0), default=0.0)
            supported_duration += duration * min(1.0, supported_fraction)
            for candidate in turn["evidence"]:
                votes[speakers[0]][candidate["person_id"]] += candidate["support"] * duration
        turns.append(turn)
    speaker_results = {}
    total = matched = 0.0
    for speaker, person_votes in votes.items():
        ranked = sorted(person_votes.items(), key=lambda item: (-item[1], item[0]))
        weight = sum(person_votes.values())
        first_person, first_weight = ranked[0]
        purity = first_weight / weight if weight > 0 else 0.0
        accepted = weight >= 0.15 and purity >= 0.75
        speaker_results[speaker] = {
            "person_id": first_person if accepted else None,
            "status": "bound" if accepted else "ambiguous_visual_identity",
            "visual_consistency": purity, "support_weight": weight,
            "person_votes": dict(ranked),
        }
        total += weight
        matched += first_weight
    consistency = matched / total if total > 0 else 0.0
    coverage = supported_duration / speech_duration if speech_duration > 0 else 0.0
    # 近平局时偏好较少碎片的声学结果，但不强制等于画面人数。
    score = consistency * coverage - 0.005 * max(0, len(votes) - 1)
    return {"audio": audio, "turns": turns, "speakers": speaker_results,
            "visual_consistency": consistency, "selection_score": score,
            "evidence_coverage": coverage, "asr_speech_seconds": speech_duration,
            "single_speaker_asd_supported_seconds": supported_duration,
            "total_visual_support": total}


def words_text(words):
    texts = [word["text"] for word in words]
    unspaced = any(re.search(r"[\u3400-\u9fff\u3040-\u30ff]", text) for text in texts)
    return re.sub(r"[ \t]+", " ", ("" if unspaced or any(text.startswith(" ") for text in texts) else " ").join(texts)).strip()


def independent_visual_reference(turn, speakers, eligible_people):
    if len(turn["speaker_ids"]) != 1:
        return None
    speaker = speakers.get(turn["speaker_ids"][0])
    if speaker is None or speaker["person_id"] not in eligible_people:
        return None
    # The turn being recovered must not supply its own reference votes.
    votes = dict(speaker["person_votes"])
    duration = turn["end"] - turn["start"]
    for candidate in turn["evidence"]:
        person = candidate["person_id"]
        votes[person] = max(0.0, votes[person] - candidate["support"] * duration)
    weight = sum(votes.values())
    if weight < .15:
        return None
    person = max(votes, key=votes.get)
    consistency = votes[person] / weight
    if person != speaker["person_id"] or consistency < .75:
        return None
    return {"person_id": person, "visual_consistency": consistency, "support_weight": weight}


def sustained_visual_anchor(piece, person, frames, fps):
    first, last = math.floor(piece["start"] * fps), math.ceil(piece["end"] * fps)
    runs = []
    for frame in sorted(frame for frame in frames[person] if first <= frame < last):
        if not runs or frame != runs[-1][1]:
            runs.append([frame, frame + 1])
        else:
            runs[-1][1] = frame + 1
    for first, last in runs:
        # At least 0.4 seconds of continuous evidence, not a few isolated detections.
        if last - first < math.ceil(.4 * fps):
            continue
        interval = {"start": first / fps, "end": last / fps}
        evidence = evidence_for_turn(interval, frames, fps)
        top = evidence[0]
        runner_up = evidence[1]["support"] if len(evidence) > 1 else 0
        if (top["person_id"] == person and top["support"] >= .25 and
                top["positive_fraction"] >= .6 and top["support"] - runner_up >= .25):
            return {"interval": interval, "evidence": evidence}
    return None


def visual_speech_turns(turns, frames, shots, fps, speakers, eligible_people):
    """A voice cluster may belong to several on-screen characters; bind locally."""
    result = []
    for turn in turns:
        words = turn.get("words", [])
        if (len(turn["speaker_ids"]) > 1 or not turn["human_speech"] or not words or
                any(word.get("start") is None or word.get("end") is None for word in words)):
            result.append({**turn, "source_asr_text": turn.get("source_asr_text", turn["text"]),
                           "visual_person_id": None, "visual_assignment": "unresolved_word_or_speaker_evidence"})
            continue
        groups = []
        current_person = None
        for word in words:
            center = (word["start"] + word["end"]) / 2
            shot = next((shot for shot in shots if shot["start_frame"] / fps <= center < shot["end_frame"] / fps), None)
            evidence = []
            if shot is not None:
                window = {"start": max(shot["start_frame"] / fps, word["start"] - .2),
                          "end": min(shot["end_frame"] / fps, word["end"] + .2)}
                evidence = evidence_for_turn(window, frames, fps)
            top = evidence[0] if evidence else None
            runner_up = evidence[1]["support"] if len(evidence) > 1 else 0
            strong = (top is not None and top["support"] >= .25 and top["frame_coverage"] >= .5
                      and top["positive_fraction"] >= .6 and top["support"] - runner_up >= .25)
            person = top["person_id"] if strong else None
            # Brief weak ASD evidence must not split a sentence into isolated characters.
            if not groups or person is not None and current_person is not None and person != current_person:
                groups.append({"person": person, "words": []})
                current_person = person
            elif person is not None:
                current_person = person
                groups[-1]["person"] = person
            groups[-1]["words"].append(word)
        for group in groups:
            items = group["words"]
            piece = {**turn, "start": items[0]["start"], "end": items[-1]["end"],
                     "text": words_text(items), "words": items, "source_asr_text": turn.get("source_asr_text", turn["text"]),
                     "visual_person_id": None, "visual_assignment": "unresolved_local_visual_evidence"}
            piece["evidence"] = evidence_for_turn(piece, frames, fps)
            ranked = piece["evidence"]
            top = ranked[0] if ranked else None
            runner_up = ranked[1]["support"] if len(ranked) > 1 else 0
            if top is not None and group["person"] in (None, top["person_id"]):
                strong = (top["support"] >= .25 and top["frame_coverage"] >= .25 and
                          top["positive_fraction"] >= .6 and top["support"] - runner_up >= .25)
                # 句尾停顿会稀释整句 ASD；保留已经通过逐词竞争检查的同一人锚点。
                anchored = (group["person"] == top["person_id"] and top["frame_coverage"] >= .25 and
                            top["positive_fraction"] >= .5 and top["support"] > 0 and
                            top["support"] >= 2 * runner_up)
                if strong or anchored:
                    piece.update(visual_person_id=top["person_id"], visual_assignment=(
                        "word_local_visual_evidence" if strong else "consistent_word_visual_evidence"))
                elif (top["frame_coverage"] < .25 and top["mean_asd_logit"] >= .25 and
                      top["positive_fraction"] >= .6):
                    reference = independent_visual_reference(turn, speakers, eligible_people)
                    if reference is not None and reference["person_id"] == top["person_id"]:
                        anchor = sustained_visual_anchor(piece, top["person_id"], frames, fps)
                        if anchor is not None:
                            piece.update(visual_person_id=top["person_id"],
                                         visual_assignment="global_voice_local_visual_anchor",
                                         binding_recoveries=[{"reference": reference, "anchor": anchor}])
            result.append(piece)

    merged = []
    for turn in result:
        previous = merged[-1] if merged else None
        compatible_voice = previous is not None and (previous["speaker_ids"] == turn["speaker_ids"] or (
            len(previous["speaker_ids"]) <= 1 and len(turn["speaker_ids"]) <= 1 and
            (not previous["speaker_ids"] or not turn["speaker_ids"]) and
            previous["source_asr_text"] == turn["source_asr_text"]))
        merge = (previous is not None and turn["visual_person_id"] is not None and
                 previous["visual_person_id"] == turn["visual_person_id"] and
                 compatible_voice and len(turn["speaker_ids"]) <= 1 and
                 previous["human_speech"] and turn["human_speech"] and
                 0 <= turn["start"] - previous["end"] <= .5 and turn["end"] - previous["start"] <= 12 and
                 not re.search(r"[。！？.!?][\"'”’）)]*\s*$", previous["text"]))
        if merge:
            previous["words"].extend(turn["words"])
            speakers = previous["speaker_ids"] or turn["speaker_ids"]
            previous.update(end=turn["end"], text=words_text(previous["words"]), speaker_ids=speakers,
                            speaker_id=speakers[0] if len(speakers) == 1 else None)
            previous["evidence"] = evidence_for_turn(previous, frames, fps)
            if "binding_recoveries" in turn:
                previous.setdefault("binding_recoveries", []).extend(turn["binding_recoveries"])
        else:
            merged.append(turn)
    return merged


def speaker_turn_merge_policy(max_gap=.8, max_duration=15.):
    if not math.isfinite(max_gap) or not math.isfinite(max_duration) or max_gap < 0 or max_duration <= 0:
        raise ValueError("Speaker-turn gap must be nonnegative and duration must be positive")
    return {"policy": SPEAKER_TURN_MERGE_POLICY, "max_gap_seconds": max_gap,
            "max_duration_seconds": max_duration, "cross_shot": "unfinished_bound_visual_only"}


def merge_speaker_turns(plan, max_gap=.8, max_duration=15.):
    """Group continuous same-speaker speech, retaining the original timed parts."""
    policy = speaker_turn_merge_policy(max_gap, max_duration)
    result = deepcopy(plan)
    rows = sorted(result["utterances"], key=lambda row: (row["start"], row["end"]))
    fps = plan["fps"]

    def speaker(row):
        acoustic = row.get("acoustic_speaker_ids", row["speaker_ids"])
        if not row["human_speech"] or len(row["speaker_ids"]) != 1 or len(acoustic) > 1:
            return None
        if row["person_id"] is None and row["speaker_id"] not in acoustic:
            return None
        return row["speaker_id"], row["person_id"]

    def shot(row):
        return next((item["id"] for item in plan["shots"] if
                     item["start_frame"] / fps <= row["start"] < row["end"] <= item["end_frame"] / fps), None)

    def bound_visual(row):
        return (row.get("status") == "bound" and row["person_id"] is not None and
                row["speaker_id"] == f"visual__{row['person_id']}")

    def acoustic_ids(row):
        return set(row.get("acoustic_speaker_ids", row["speaker_ids"]))

    overlapping = {row["id"] for row in rows if row["human_speech"] and any(
        other["id"] != row["id"] and other["human_speech"] and
        min(row["end"], other["end"]) - max(row["start"], other["start"]) > 1 / fps for other in rows)}
    groups = []
    for row in rows:
        previous = groups[-1] if groups else []
        key, shot_id = speaker(row), shot(row)
        same_shot = previous and shot_id is not None and shot(previous[-1]) == shot_id
        known_voices = set().union(*(acoustic_ids(part) for part in previous)) if previous else set()
        incoming_voices = acoustic_ids(row)
        cross_shot_continuation = (previous and not same_shot and bound_visual(previous[-1]) and
                                   bound_visual(row) and len(known_voices) <= 1 and
                                   (not known_voices or not incoming_voices or known_voices == incoming_voices) and
                                   not re.search(r"[。！？.!?][\"'”’）)]*\s*$", previous[-1]["text"]))
        if (previous and key is not None and speaker(previous[-1]) == key and
                (same_shot or cross_shot_continuation) and
                row["id"] not in overlapping and previous[-1]["id"] not in overlapping and
                0 <= row["start"] - previous[-1]["end"] <= max_gap + 1e-9 and
                row["end"] - previous[0]["start"] <= max_duration + 1e-9):
            previous.append(row)
        else:
            groups.append([row])
    merged = []
    for parts in groups:
        row = deepcopy(parts[0])
        row["source_parts"] = parts
        if len(parts) > 1:
            row.update(end=parts[-1]["end"], text=" ".join(part["text"].strip() for part in parts),
                       words=deepcopy([word for part in parts for word in part["words"]]),
                       source_asr_text=" ".join(dict.fromkeys(part["source_asr_text"] for part in parts)),
                       untimed_words=deepcopy([word for part in parts for word in part.get("untimed_words", [])]),
                       track_ids=list(dict.fromkeys(track for part in parts for track in part["track_ids"])),
                       asr_confidence={"backend": "source_turn_group", "source_ids": [part["id"] for part in parts]},
                       binding_evidence=[], assignment_source="continuous_same_speaker_from_source_parts")
            acoustic = list(dict.fromkeys(speaker for part in parts for speaker in part.get("acoustic_speaker_ids", part["speaker_ids"])))
            row.update(acoustic_speaker_ids=acoustic, acoustic_speaker_id=acoustic[0] if len(acoustic) == 1 else None)
            if len({shot(part) for part in parts}) > 1 or any(shot(part) is None for part in parts):
                row.update(cross_shot_translation_unit=True,
                           acoustic_identity_confirmed=(len(acoustic) == 1 and all(
                               part.get("acoustic_speaker_ids", part["speaker_ids"]) == acoustic for part in parts)))
            first, last = math.floor(row["start"] * fps), math.ceil(row["end"] * fps)
            visible = {frame for track in plan["tracks"] if row["person_id"] is not None and track["person_id"] == row["person_id"]
                       for frame in track["frames"] if first <= frame < last}
            row["face_detection_frame_coverage"] = len(visible) / max(1, last - first)
        merged.append(row)
    result["utterances"] = merged
    result["diagnostics"]["speaker_turn_merging"] = {
        **policy, "input_utterances": len(rows), "output_turns": len(merged)}
    return result


def build_plan(vision, identity, audio_candidates, mode="speaker_visual", merge_turns=False, merge_gap=.8, merge_duration=15.):
    if mode != "speaker_visual":
        raise ValueError("Only speaker_visual fusion is supported")
    frames = frame_evidence(vision, identity)
    assessed = [assess_audio(audio, frames, vision["fps"], mode) for audio in audio_candidates]
    best = max(assessed, key=lambda candidate: candidate["selection_score"])
    mapping, identities = identity_map(identity)
    tracks = [{**track,
               "person_id": mapping[track["id"]],
               "identity_candidate_id": mapping[track["id"]],
               "identity_quality": identities[track["id"]]["quality"]} for track in vision["tracks"]]
    utterances = []
    eligible_people = {mapping[track["id"]] for track in identity["tracks"] if track["quality"]["match_eligible"]}
    turns = visual_speech_turns(best["turns"], frames, vision["shots"], vision["fps"],
                               best["speakers"], eligible_people)
    for index, turn in enumerate(turns):
        speakers = turn["speaker_ids"]
        speaker = speakers[0] if len(speakers) == 1 else None
        local = turn["evidence"]
        person, source = turn["visual_person_id"], turn["visual_assignment"]
        if not turn["human_speech"]:
            person, source = None, "not_accepted_human_speech"
        first, last = math.floor(turn["start"] * vision["fps"]), math.ceil(turn["end"] * vision["fps"])
        visible = {frame for track in tracks if track["person_id"] == person
                   for frame in track["frames"] if first <= frame < last} if person else set()
        utterances.append({"id": f"utterance_{index + 1:03d}", "start": turn["start"], "end": turn["end"],
                           "text": turn["text"], "speaker_id": speaker, "speaker_ids": speakers,
                           "human_speech": turn["human_speech"], "human_speech_reason": turn["human_speech_reason"],
                           "asr_confidence": turn.get("asr_confidence", {}),
                           "person_id": person, "status": "bound" if person else "unresolved",
                           "assignment_source": source, "binding_evidence": local,
                           "track_ids": [track["id"] for track in tracks if track["person_id"] == person
                                         and any(first <= frame < last for frame in track["frames"])],
                           "face_detection_frame_coverage": len(visible) / max(1, last - first),
                           "source_asr_text": turn.get("source_asr_text", turn["text"]),
                           "untimed_words": turn.get("untimed_words", []),
                           "words": turn.get("words", [])})
        row = utterances[-1]
        if "binding_recoveries" in turn:
            row["binding_recoveries"] = turn["binding_recoveries"]
        row.update(acoustic_speaker_id=speaker, acoustic_speaker_ids=list(speakers))
        if person is not None and len(speakers) <= 1:
            visual_speaker = f"visual__{person}"
            row.update(speaker_id=visual_speaker, speaker_ids=[visual_speaker],
                       speaker_identity_source="bound_visual_person")
        elif not speakers and turn["human_speech"]:
            unknown_speaker = f"unknown_{row['id']}"
            row.update(speaker_id=unknown_speaker, speaker_ids=[unknown_speaker],
                       speaker_identity_source="unknown_voice")
    audio = best["audio"]
    backend = audio["models"]["diarization"]["backend"]
    diagnostics = []
    for candidate in assessed:
        diagnostics.append({"audio_file": candidate["audio"].get("_result_path"),
                            "backend": candidate["audio"]["models"]["diarization"]["backend"],
                            "speaker_count": len(candidate["speakers"]),
                            "visual_consistency": candidate["visual_consistency"],
                            "evidence_coverage": candidate["evidence_coverage"],
                            "asr_speech_seconds": candidate["asr_speech_seconds"],
                            "single_speaker_asd_supported_seconds": candidate["single_speaker_asd_supported_seconds"],
                            "selection_score": candidate["selection_score"],
                            "speaker_bindings": candidate["speakers"]})
    plan = {"schema_version": 1, "fps": vision["fps"], "frames": vision["frames"],
            "width": vision["width"], "height": vision["height"], "shots": vision["shots"],
            "source_video": vision["video_path"], "source_audio": audio["source_audio"],
            "selected_audio_backend": backend, "selected_audio_result": audio.get("_result_path"),
            "models": {"audio": audio["models"], "vision": vision["models"], "identity": identity["models"]},
            "tracks": tracks, "speakers": best["speakers"], "utterances": utterances,
            "diagnostics": {"audio_candidates": diagnostics, "fusion_mode": mode,
                            "selection_rule": "visual_consistency * evidence_coverage - 0.005 * extra_clusters",
                            "human_speech_filter": "ASR evidence; nonlinguistic sounds and mixed sources are not perfectly classified"}}
    plan["diagnostics"]["visual_binding_policy"] = VISUAL_BINDING_POLICY
    return merge_speaker_turns(plan, merge_gap, merge_duration) if merge_turns else plan



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--audio", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("speaker_visual",), default="speaker_visual")
    parser.add_argument("--merge-speaker-turns", action="store_true")
    parser.add_argument("--speaker-turn-max-gap-seconds", type=float, default=.8)
    parser.add_argument("--speaker-turn-max-seconds", type=float, default=15.)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace this generated plan")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    candidates = []
    for path in args.audio:
        audio = json.loads(Path(path).read_text())
        audio["_result_path"] = str(Path(path).resolve())
        candidates.append(audio)
    if len({audio.get("audio_sha256") for audio in candidates}) > 1:
        raise ValueError("Audio candidates must analyze the same waveform")
    plan = build_plan(json.loads(Path(args.vision).read_text()), json.loads(Path(args.identity).read_text()), candidates, args.mode,
                      args.merge_speaker_turns, args.speaker_turn_max_gap_seconds, args.speaker_turn_max_seconds)
    write_json(output, plan)
    summary = {"selected_audio_backend": plan["selected_audio_backend"],
               "utterances": [{key: item[key] for key in ("id", "start", "end", "text", "speaker_id", "person_id", "face_detection_frame_coverage")} for item in plan["utterances"]]}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

