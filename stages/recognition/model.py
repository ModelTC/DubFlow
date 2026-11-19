"""Local Transformers adapters for Qwen3 speech and Nemotron 3 diarization."""

from pathlib import Path

from runtime.models import get
from runtime.languages import LANGUAGES

from stages.recognition.alignment import repair_word_times
from stages.recognition.utils import (
    ASR_POLICY, OVERLAP_SECONDS, SAMPLE_RATE, WINDOW_SECONDS,
    alignment_quality, anchored_words, detected_language, dominant_language,
    merge_window_words, read_mono, release, sentence_segments,
    transcript_language, transcription_windows,
)


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

def transcribe_qwen3(path, *, asr_model, align_model, language, device="cuda:0", turns=()):
    import torch

    waveform = read_mono(path)
    source_language = language
    model, processor = qwen_asr(asr_model, device)

    clips = []
    try:
        for region in transcription_windows(len(waveform) / SAMPLE_RATE):
            clip = waveform[round(region["start"] * SAMPLE_RATE):round(region["end"] * SAMPLE_RATE)]
            parsed = recognize_qwen3(model, processor, clip, source_language)
            if parsed["transcription"].strip():
                language = detected_language(parsed.get("language")) if source_language == "auto" else source_language
                if source_language == "auto":
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

    language, language_counts = (dominant_language(clips) if source_language == "auto"
                                 else (source_language, {source_language: sum(len(row[2]) for row in clips)}))
    merged, recognized, joins, alignment_windows = [], [], [], []
    if clips:
        aligner = QwenForcedAligner(align_model, device)
        try:
            previous_end = 0
            for region, clip, transcript, clip_language in clips:
                raw = aligner.align(clip, transcript, clip_language)
                quality = alignment_quality(raw, transcript)
                diagnostic = {**region, "text": transcript, "raw_time_origin_seconds": region["start"],
                              "primary_raw_words": raw, "quality": quality, "backend": "qwen3_forced_aligner",
                              "detected_language": clip_language}
                offset = region["start"]
                clock = [{**turn, "start": turn["start"] - offset, "end": turn["end"] - offset}
                         for turn in turns if turn["end"] > offset and turn["start"] < region["end"]]
                words, repairs = repair_word_times(raw, transcript, clip, clip_language, aligner.align, clock)
                absolute = [{**word, "start": word["start"] + offset, "end": word["end"] + offset}
                            for word in words]
                for word in absolute:
                    if "alignment_repair" in word:
                        word["alignment_repair"] = [offset, word["alignment_repair"]]
                words = anchored_words(absolute, transcript)
                alignment_windows.append({**diagnostic, "timestamp_repairs": repairs, "final_words": words})
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
            "models": {"asr": {"backend": "qwen3", "model": asr_model, "device": device,
                               "recognition_policy": ASR_POLICY, "window_seconds": WINDOW_SECONDS,
                               "overlap_seconds": OVERLAP_SECONDS, "vad_filter": False,
                               "confidence_available": False},
                       "alignment": {"backend": "qwen3_forced_aligner",
                                     "model": align_model, "device": device}}}


def diarize_nemotron3(path, duration, *, model_path, device="cuda:0"):
    import torch

    waveform = read_mono(path)
    model, processor = nemotron3_model(model_path, device)
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
                "model": model_path, "max_speakers": 8, "device": device}}}
    finally:
        del model, processor
        release(torch, device)
