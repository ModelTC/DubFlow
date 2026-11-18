"""Recognition language, waveform, alignment, and transcript helpers."""

import gc
import json
import re
import subprocess
import wave
from difflib import SequenceMatcher
from pathlib import Path

from runtime.languages import LANGUAGES

SAMPLE_RATE = 16000
ASR_POLICY = "full_audio_qwen_alignment_v9"
WINDOW_SECONDS = 25
OVERLAP_SECONDS = 2


def detected_language(value):
    for code, name in LANGUAGES.items():
        if isinstance(value, str) and value.strip().casefold() in (code, name.casefold()):
            return code
    raise ValueError(f"Automatically detected source language {value!r} is not supported by this dubbing route")


def transcript_language(text, detected):
    """Correct contradictory language tags only when the writing system is decisive."""
    letters = sum(char.isalpha() for char in text)
    if not letters:
        return detected
    kana = len(re.findall(r"[\u3040-\u30ff]", text))
    han = len(re.findall(r"[\u3400-\u9fff]", text))
    hangul = len(re.findall(r"[\uac00-\ud7af]", text))
    if kana >= 3 and (kana + han) / letters > .5:
        return "ja"
    if hangul >= 3 and hangul / letters > .5:
        return "ko"
    if detected not in {"ja", "ko"} and han >= 8 and han / letters > .8:
        return "zh"
    return detected


def dominant_language(clips):
    counts = {code: 0 for code in LANGUAGES}
    for _, _, transcript, language in clips:
        units = (sum(character.isalnum() for character in transcript) if language in {"zh", "ja", "ko"}
                 else len(re.findall(r"\w+(?:['’]\w+)*", transcript, flags=re.UNICODE)))
        counts[language] += max(1, units)
    if not any(counts.values()):
        raise ValueError("Could not detect a source language: no speech was recognized")
    selected = max(counts, key=counts.get)
    if sum(count for code, count in counts.items() if code != selected) > .2 * sum(counts.values()):
        raise ValueError("Mixed source languages were detected; this route needs one source language per video")
    return selected, counts


def release(torch, device):
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def read_mono(path):
    import soundfile as sf

    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or waveform.shape[1] != 1:
        raise ValueError("Modern audio analysis expects 16kHz mono WAV")
    return waveform[:, 0]


def punctuated_words(words, transcript):
    # The native aligner strips punctuation. Restore it from the actual ASR text.
    positions = [index for index, character in enumerate(transcript) if character.isalnum() or character == "'"]
    compact = "".join(transcript[index] for index in positions).casefold()
    tokens = ["".join(character for character in row["word"] if character.isalnum() or character == "'")
              for row in words]
    if "".join(tokens).casefold() != compact:
        raise ValueError("Forced alignment does not cover the recognized transcript")
    result, cursor, boundary = [], 0, 0
    for row, token in zip(words, tokens):
        if not token:
            continue
        cursor += len(token)
        end = positions[cursor] if cursor < len(positions) else len(transcript)
        result.append({**row, "word": transcript[boundary:end]})
        boundary = end
    return result


def anchored_words(words, transcript):
    words = punctuated_words(words, transcript)
    anchors = [index for index, word in enumerate(words) if word["end"] > word["start"]]
    if words and not anchors:
        raise ValueError("Forced alignment contains no positive-duration speech anchor")
    anchored = [dict(words[index]) for index in anchors]
    if not anchored:
        return anchored

    anchored[0]["word"] = "".join(word["word"] for word in words[:anchors[0]]) + anchored[0]["word"]
    for position, (left, right) in enumerate(zip(anchors, anchors[1:])):
        zero_words = words[left + 1:right]
        cost = sum(abs(words[right]["start"] - word["start"]) for word in zero_words)
        best_cost, best_cut = cost, 0
        for cut, word in enumerate(zero_words, 1):
            cost += abs(word["start"] - words[left]["end"]) - abs(words[right]["start"] - word["start"])
            if cost <= best_cost:  # At equal distances, retain the previous anchor.
                best_cost, best_cut = cost, cut
        anchored[position]["word"] += "".join(word["word"] for word in zero_words[:best_cut])
        anchored[position + 1]["word"] = ("".join(word["word"] for word in zero_words[best_cut:]) +
                                          anchored[position + 1]["word"])
    anchored[-1]["word"] += "".join(word["word"] for word in words[anchors[-1] + 1:])
    return anchored


def sentence_segments(words, transcript):
    anchored = anchored_words(words, transcript)
    groups = []
    for word in anchored:
        if not groups or groups[-1]["closed"] or word["start"] - groups[-1]["words"][-1]["end"] > .7:
            groups.append({"words": [], "closed": False})
        groups[-1]["words"].append(word)
        groups[-1]["closed"] = bool(re.search(r"[。！？.!?][\"'”’）)]*\s*$", word["word"]))
    segments = []
    for group in groups:
        items = group["words"]
        start, end = items[0]["start"], items[-1]["end"]
        segments.append({"start": start, "end": end, "text": "".join(row["word"] for row in items).strip(),
                         "words": items, "asr_confidence": {"backend": "qwen3", "confidence_available": False}})
    return segments


def alignment_quality(words, transcript):
    clauses = [{"text": "", "characters": 0, "zero_duration_characters": 0}]
    for row in punctuated_words(words, transcript):
        for character in row["word"]:
            clause = clauses[-1]
            clause["text"] += character
            if character.isalnum() or character == "'":
                clause["characters"] += 1
                clause["zero_duration_characters"] += row["start"] == row["end"]
            if character in "，,。.!?！？；;：:\n" and clause["characters"]:
                clauses.append({"text": "", "characters": 0, "zero_duration_characters": 0})
    clauses = [row for row in clauses if row["characters"]]
    for clause in clauses:
        clause["text"] = clause["text"].strip()
        clause["zero_duration_fraction"] = clause["zero_duration_characters"] / clause["characters"]
    total = sum(row["characters"] for row in clauses)
    zero = sum(row["zero_duration_characters"] for row in clauses)
    return {"characters": total, "zero_duration_characters": zero,
            "zero_duration_fraction": zero / total if total else 0.0, "clause_count": len(clauses),
            "worst_clause": max(clauses, key=lambda row: (row["zero_duration_fraction"], row["characters"])) if clauses else None}


def transcription_windows(duration):
    start = 0.0
    while start < duration:
        end = min(duration, start + WINDOW_SECONDS)
        yield {"start": start, "end": end}
        if end == duration:
            break
        start += WINDOW_SECONDS - OVERLAP_SECONDS


def merge_window_words(previous, incoming, window_start, previous_window_end):
    """Join aligned overlapping windows at a shared word, retaining each word once."""
    if not previous or not incoming:
        return previous + incoming, "no_shared_speech"
    left = [(index, word) for index, word in enumerate(previous) if word["end"] > window_start]
    right = [(index, word) for index, word in enumerate(incoming) if word["start"] < previous_window_end]
    midpoint = (window_start + previous_window_end) / 2

    def token(word):
        return "".join(character for character in word["word"].casefold() if character.isalnum())

    matches = SequenceMatcher(None, [token(word) for _, word in left],
                              [token(word) for _, word in right], autojunk=False)
    anchors = []
    for block in matches.get_matching_blocks():
        for offset in range(block.size):
            old_index, old = left[block.a + offset]
            new_index, new = right[block.b + offset]
            old_center = (old["start"] + old["end"]) / 2
            new_center = (new["start"] + new["end"]) / 2
            if abs(old_center - new_center) <= .8:
                anchors.append((abs((old_center + new_center) / 2 - midpoint), old_index, new_index))
    if anchors:
        _, old_index, new_index = min(anchors)
        return previous[:old_index + 1] + incoming[new_index + 1:], "shared_aligned_word"
    # Disagreeing transcripts have no lexical seam; keep whole words on either side.
    return ([word for word in previous if (word["start"] + word["end"]) / 2 <= midpoint] +
            [word for word in incoming if (word["start"] + word["end"]) / 2 > midpoint]), "timestamp_boundary"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_audio(*, audio, output, video=None):
    if audio:
        path = Path(audio).resolve()
    else:
        path = Path(output).resolve().parent / "work/audio_16k.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-n", "-i", video,
                            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(path)], check=True)
    with wave.open(str(path)) as waveform:
        if (waveform.getframerate(), waveform.getnchannels(), waveform.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Source audio must be 16kHz mono PCM16 WAV")
        duration = waveform.getnframes() / 16000
    return path, duration


def label_words(asr, diarization):
    """保留真实同时说话；粗词时间跨换人时，用交集和片段边界启发式赋 ID。"""
    output = []
    for segment in asr["aligned_segments"]:
        candidates, totals = [], {}
        for source in segment.get("words", []):
            word = dict(source)
            word["text"] = word.pop("word")
            scores, active = {}, []
            if word.get("start") is not None and word.get("end") is not None:
                for turn in diarization:
                    start, end = max(word["start"], turn["start"]), min(word["end"], turn["end"])
                    if end > start:
                        speaker = turn["speaker_id"]
                        overlap = end - start
                        scores[speaker] = scores.get(speaker, 0.0) + overlap
                        totals[speaker] = totals.get(speaker, 0.0) + overlap
                        active.append((start, end, speaker))
            simultaneous = any(left[2] != right[2] and max(left[0], right[0]) < min(left[1], right[1])
                               for index, left in enumerate(active) for right in active[index + 1:])
            candidates.append((word, scores, simultaneous))

        dominant = max(totals, key=totals.get) if totals else None
        words, assigned_totals = [], {}
        for index, (word, scores, simultaneous) in enumerate(candidates):
            speakers = sorted(scores, key=scores.get, reverse=True)
            if len(speakers) > 1:
                word["speaker_overlap_seconds"] = scores
                if simultaneous:
                    word["speaker_assignment"] = "simultaneous_turn_overlap"
                elif index in (0, len(candidates) - 1) and dominant in scores:
                    speakers = [dominant]
                    word["speaker_assignment"] = "segment_boundary_dominant"
                else:
                    speakers = speakers[:1]
                    word["speaker_assignment"] = "sequential_turn_max_overlap"
            word["speaker_ids"] = speakers
            word["speaker_id"] = speakers[0] if speakers else None
            for speaker in speakers:
                assigned_totals[speaker] = assigned_totals.get(speaker, 0.0) + scores[speaker]
            words.append(word)
        output.append({"start": segment["start"], "end": segment["end"], "text": segment["text"],
                       "speaker_id": max(assigned_totals, key=assigned_totals.get) if assigned_totals else None,
                       "speaker_ids": sorted(assigned_totals, key=assigned_totals.get, reverse=True),
                       "asr_confidence": segment.get("asr_confidence", {}), "words": words})
    return output
