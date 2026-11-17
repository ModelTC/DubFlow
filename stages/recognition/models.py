"""Local Transformers adapters for Qwen3 speech and Nemotron 3 diarization."""

import gc
import re
from difflib import SequenceMatcher
from pathlib import Path

from runtime.languages import LANGUAGES
from runtime.models import get

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


def qwen_asr(model_path, device):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    path = str(Path(model_path).resolve())
    dtype = "bfloat16" if device.startswith("cuda") else "float32"

    def load():
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            path, local_files_only=True, dtype=getattr(torch, dtype),
            attn_implementation="sdpa").to(device).eval()
        return model, processor

    return get(("qwen3_asr", path, device, dtype), load)


def nemotron3_model(model_path, device):
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    path = str(Path(model_path).resolve())
    dtype = "bfloat16" if device.startswith("cuda") else "float32"

    def load():
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModelForAudioFrameClassification.from_pretrained(
            path, local_files_only=True, dtype=getattr(torch, dtype)).to(device).eval()
        return model, processor

    return get(("nemotron3", path, device, dtype), load)


class QwenForcedAligner:
    def __init__(self, model_path, device="cuda:0"):
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor

        self.torch, self.device = torch, device
        path = str(Path(model_path).resolve())
        dtype = "bfloat16" if device.startswith("cuda") else "float32"

        def load():
            processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            model = AutoModelForTokenClassification.from_pretrained(
                path, local_files_only=True, dtype=getattr(torch, dtype),
                attn_implementation="sdpa").to(device).eval()
            return model, processor

        self.model, self.processor = get(("qwen3_aligner", path, device, dtype), load)

    def align(self, audio, text, language):
        inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio=str(audio) if isinstance(audio, Path) else audio, transcript=text, language=LANGUAGES[language])
        inputs = inputs.to(self.model.device, self.model.dtype)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        stamps = self.processor.decode_forced_alignment(
            logits=outputs.logits, input_ids=inputs["input_ids"], word_lists=word_lists,
            timestamp_token_id=self.model.config.timestamp_token_id)[0]
        return [{"word": row["text"], "start": float(row["start_time"]), "end": float(row["end_time"])}
                for row in stamps]

    def close(self):
        del self.model, self.processor
        release(self.torch, self.device)


def recognize_qwen3(model, processor, clip, language):
    import torch

    request = {"audio": clip}
    if language != "auto":
        request["language"] = LANGUAGES[language]
    inputs = processor.apply_transcription_request(**request).to(model.device, model.dtype)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    generated = generated[:, inputs["input_ids"].shape[1]:]
    if generated.shape[1] >= 1024:
        raise RuntimeError("Qwen3-ASR reached its output limit; refusing a truncated transcript")
    return processor.decode(generated, return_format="parsed")[0]

def transcribe_qwen3(path, args):
    import torch

    waveform = read_mono(path)
    device = args.device
    model, processor = qwen_asr(args.asr_model, device)

    clips = []
    try:
        for region in transcription_windows(len(waveform) / SAMPLE_RATE):
            clip = waveform[round(region["start"] * SAMPLE_RATE):round(region["end"] * SAMPLE_RATE)]
            parsed = recognize_qwen3(model, processor, clip, args.language)
            if parsed["transcription"].strip():
                language = detected_language(parsed.get("language")) if args.language == "auto" else args.language
                if args.language == "auto":
                    corrected = transcript_language(parsed["transcription"], language)
                    if corrected != language:
                        # One explicit-language recognition, only for a contradictory auto tag.
                        print(f"ASR language tag {language} conflicts with transcript script; retrying as {corrected}", flush=True)
                        corrected_text = recognize_qwen3(model, processor, clip, corrected)["transcription"].strip()
                        if corrected_text:
                            parsed["transcription"] = corrected_text
                        language = corrected
                clips.append((region, clip, parsed["transcription"].strip(), language))
    finally:
        del model, processor
        release(torch, device)

    language, language_counts = (dominant_language(clips) if args.language == "auto"
                                 else (args.language, {args.language: sum(len(row[2]) for row in clips)}))
    merged, recognized, joins, alignment_windows = [], [], [], []
    if clips:
        aligner = QwenForcedAligner(args.align_model, device)
        try:
            previous_end = 0
            for region, clip, transcript, clip_language in clips:
                raw = aligner.align(clip, transcript, clip_language)
                quality = alignment_quality(raw, transcript)
                diagnostic = {**region, "text": transcript, "raw_time_origin_seconds": region["start"],
                              "primary_raw_words": raw, "quality": quality, "backend": "qwen3_forced_aligner",
                              "detected_language": clip_language}
                words = punctuated_words(raw, transcript)
                clip_duration = len(clip) / SAMPLE_RATE
                absolute = []
                for word in words:
                    if not 0 <= word["start"] <= word["end"] <= clip_duration + .08:
                        raise ValueError("Forced-alignment word timestamp is outside the source audio")
                    absolute.append({**word, "start": min(word["start"], clip_duration) + region["start"],
                                     "end": min(word["end"], clip_duration) + region["start"]})
                words = anchored_words(absolute, transcript)
                alignment_windows.append({**diagnostic, "final_words": words})
                if region["start"] < previous_end:
                    merged, method = merge_window_words(merged, words, region["start"], previous_end)
                    joins.append({"start": region["start"], "end": previous_end, "method": method})
                else:
                    merged.extend(words)
                previous_end = region["end"]
                recognized.append({"start": region["start"], "end": region["end"], "text": transcript,
                                   "language": clip_language})
        finally:
            aligner.close()
    aligned = sentence_segments(merged, "".join(word["word"] for word in merged))
    return {"language": language, "language_probability": None, "detected_language_counts": language_counts,
            "asr_segments": recognized,
            "aligned_segments": aligned, "window_joins": joins, "alignment_windows": alignment_windows,
            "models": {"asr": {"backend": "qwen3", "model": args.asr_model, "device": device,
                               "recognition_policy": ASR_POLICY, "window_seconds": WINDOW_SECONDS,
                               "overlap_seconds": OVERLAP_SECONDS, "vad_filter": False,
                               "confidence_available": False},
                       "alignment": {"backend": "qwen3_forced_aligner",
                                     "model": args.align_model, "device": device}}}



def diarize_nemotron3(path, args, duration):
    import torch

    waveform = read_mono(path)
    model, processor = nemotron3_model(args.diar_model, args.device)
    try:
        inputs = processor(waveform, sampling_rate=SAMPLE_RATE).to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            logits = model(**inputs).logits
        detected = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
        segments = [{"start": max(0.0, float(row["Start"])), "end": min(duration, float(row["End"])),
                     "speaker_id": f"speaker_{row['Speaker']}"} for row in detected
                    if min(duration, float(row["End"])) > max(0.0, float(row["Start"]))]
        segments.sort(key=lambda row: (row["start"], row["end"], row["speaker_id"]))
        return {"diarization": segments, "models": {"diarization": {"backend": "hf_nemotron3",
                "model": args.diar_model, "max_speakers": 8, "device": args.device}}}
    finally:
        del model, processor
        release(torch, args.device)
