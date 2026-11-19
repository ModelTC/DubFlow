"""Repair locally collapsed word times while preserving the recognized words."""

import math
import re

from stages.recognition.utils import SAMPLE_RATE, punctuated_words

TIMESTAMP_STEP = .08


def collapsed_ranges(words):
    """Find compressed runs; one untimed function word is common at 80 ms resolution."""
    ranges = []
    first = 0
    while first < len(words):
        if words[first]["end"] > words[first]["start"]:
            first += 1
            continue
        last = first + 1
        while last < len(words) and words[last]["end"] <= words[last]["start"]:
            last += 1
        text = "".join(word["word"] for word in words[first:last])
        standalone = first == 0 or re.search(r"[。！？.!?][\"'”’）)]*\s*$", words[first - 1]["word"])
        if last - first > 1 or standalone and re.search(r"[。！？.!?]", text):
            # Re-align an anchor on each side so the local model hears the words in context.
            start, end = max(0, first - 1), min(len(words), last + 1)
            if ranges and start < ranges[-1][1]:
                ranges[-1] = ranges[-1][0], end
            else:
                ranges.append((start, end))
        first = last
    if words and not any(word["end"] > word["start"] for word in words):
        return [(0, len(words))]
    return ranges


def matching_turn(first, last, turns):
    """Use a single acoustic turn to bound a repair only when no rival turn shares it."""
    overlapping = [turn for turn in turns if min(last, turn["end"]) - max(first, turn["start"]) > .08]
    if len(overlapping) != 1:
        return None
    turn = overlapping[0]
    if min(last, turn["end"]) - max(first, turn["start"]) >= .6 * (last - first):
        return turn
    return None


def estimate_missing_times(words, turn, first, last):
    """Spread missing times only through unclaimed space in a confirmed speaker turn."""
    if turn is None:
        return words
    first, last = max(first, turn["start"]), min(last, turn["end"])
    repaired = list(words)
    start = 0
    while start < len(words):
        if words[start]["end"] > words[start]["start"]:
            start += 1
            continue
        end = start + 1
        while end < len(words) and words[end]["end"] <= words[end]["start"]:
            end += 1
        left = max(first, words[start - 1]["end"]) if start else first
        right = min(last, words[end]["start"]) if end < len(words) else last
        if right - left >= TIMESTAMP_STEP * (end - start):
            # Prefer word length to a flat split, with a minimum for single-letter words.
            weights = [max(2, sum(character.isalnum() for character in row["word"]))
                       for row in words[start:end]]
            duration, total = right - left, sum(weights)
            position = left
            for index, weight in zip(range(start, end), weights):
                stop = right if index == end - 1 else position + duration * weight / total
                repaired[index] = {**words[index], "start": position, "end": stop,
                                   "alignment_method": "diarization_estimate"}
                position = stop
        start = end
    return repaired


def clip_word_times(words, duration):
    """Allow one decoder timestamp tick at the edge of a cropped waveform."""
    clipped = []
    for word in words:
        start, end = word["start"], word["end"]
        if not (math.isfinite(start) and math.isfinite(end) and
                0 <= start <= end <= duration + TIMESTAMP_STEP):
            raise ValueError("Forced-alignment word timestamp is outside the source audio")
        clipped.append({**word, "start": min(start, duration), "end": min(end, duration)})
    return clipped


def read_local_alignment(align, clip, language, text, first, last):
    begin_sample, end_sample = round(first * SAMPLE_RATE), round(last * SAMPLE_RATE)
    if end_sample - begin_sample < round(TIMESTAMP_STEP * SAMPLE_RATE):
        return []
    start = begin_sample / SAMPLE_RATE
    region = clip[begin_sample:end_sample]
    try:
        words = clip_word_times(punctuated_words(align(region, text, language), text), len(region) / SAMPLE_RATE)
    except ValueError:
        return []
    if any(word["start"] < previous["end"] - 1e-6 for previous, word in zip(words, words[1:])):
        return []
    return [{**word, "start": word["start"] + start, "end": word["end"] + start,
             "alignment_method": "local_forced_alignment"} for word in words]


def missing_units(words):
    return sum(sum(character.isalnum() for character in word["word"]) for word in words
               if word["end"] <= word["start"])


def repair_word_times(raw, transcript, clip, language, align, turns=()):
    duration = len(clip) / SAMPLE_RATE
    words = clip_word_times(punctuated_words(raw, transcript), duration)
    ranges = collapsed_ranges(words)
    result, diagnostics = [], []
    previous = 0
    for first, last in ranges:
        original = words[first:last]
        left = words[first - 1]["end"] if first else 0.
        right = words[last]["start"] if last < len(words) else duration
        start, end = original[0]["start"], original[-1]["end"]
        evidence_start, evidence_end = (start, end) if end > start else (
            max(left, start - .25), min(right, end + .35))
        turn = matching_turn(evidence_start, evidence_end, turns)
        crop_start = max(left, min(start - .25, turn["start"] - .16 if turn else start))
        crop_end = min(right, max(end + .35, turn["end"] + .16 if turn else end))
        text = "".join(word["word"] for word in original)
        local = read_local_alignment(align, clip, language, text, crop_start, crop_end)
        candidates = [original, estimate_missing_times(original, turn, crop_start, crop_end)]
        if local:
            candidates.append(estimate_missing_times(local, turn, crop_start, crop_end))
        # Keep the model result when it recovers at least as much as the fallback.
        best = min(reversed(candidates), key=missing_units)
        if missing_units(best) >= missing_units(original):
            best = original
        if best is not original and turn is not None:
            # Keep a continuation attached when its recovered time is just outside
            # the detected turn. Real per-word overlap still takes precedence.
            best = [{**word, "alignment_repair": first, "alignment_speaker_hint": turn["speaker_id"]}
                    if max(turn["start"] - word["end"], word["start"] - turn["end"]) <= .2 else word
                    for word in best]
        methods = sorted({word["alignment_method"] for word in best if "alignment_method" in word})
        diagnostics.append({"text": text.strip(), "crop_start": crop_start, "crop_end": crop_end,
                            "missing_characters_before": missing_units(original), "missing_characters_after": missing_units(best),
                            "methods": methods})
        result.extend(words[previous:first])
        result.extend(best)
        previous = last
    result.extend(words[previous:])
    return result, diagnostics
