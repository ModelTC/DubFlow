"""Qwen3-ASR、强对齐和 Nemotron 声学说话人标注。"""

import argparse
import hashlib
import json
import subprocess
import wave
from pathlib import Path

from runtime.config import ROOT
from runtime.languages import LANGUAGES


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_audio(args):
    if args.audio:
        path = Path(args.audio).resolve()
    else:
        path = Path(args.output).resolve().parent / "work/audio_16k.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-n", "-i", args.video,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--audio")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asr-model", default=str(ROOT / "models/Qwen3-ASR-1.7B-hf"))
    parser.add_argument("--align-model", default=str(ROOT / "models/Qwen3-ForcedAligner-0.6B-hf"))
    parser.add_argument("--diar-model", default=str(ROOT / "models/Nemotron-3-Diarization"))
    parser.add_argument("--diarization-cache", help="复用相同原声、后端和模型的说话人结果，重新按当前词时间标注")
    parser.add_argument("--language", required=True, choices=("auto", *LANGUAGES))
    parser.add_argument("--stage", choices=("all", "asr", "diarization"), default="all")
    args = parser.parse_args()
    path, duration = source_audio(args)
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    output = Path(args.output).resolve()
    asr_path = output.with_name("audio_asr.json")
    if args.stage in ("asr", "all"):
        from stages.recognition.models import transcribe_qwen3
        asr = transcribe_qwen3(path, args)
        asr.update(audio_sha256=fingerprint, duration=duration, source_audio=str(path))
        write_json(asr_path, asr)
    else:
        asr = json.loads(asr_path.read_text())
        if asr["audio_sha256"] != fingerprint:
            raise ValueError("ASR cache belongs to another source audio")
        from stages.recognition.models import ASR_POLICY
        if asr["models"]["asr"].get("recognition_policy") != ASR_POLICY:
            raise ValueError("Qwen3 ASR cache uses an older recognition policy; regenerate ASR in a new output directory")
    if args.stage == "asr":
        print(f"ASR and alignment: {asr_path}", flush=True)
        return

    if args.diarization_cache:
        cache_path = Path(args.diarization_cache).resolve()
        cached = json.loads(cache_path.read_text())
        cached_model = cached["models"]["diarization"]
        expected_backend, model_path = "hf_nemotron3", args.diar_model
        if (cached["audio_sha256"] != fingerprint or cached_model["backend"] != expected_backend or
                Path(cached_model["model"]).resolve() != Path(model_path).resolve()):
            raise ValueError("Diarization cache belongs to another source waveform, backend or model")
        diarization = {"diarization": cached["diarization"],
                       "models": {"diarization": {**cached_model, "cached_from": str(cache_path),
                                                   "execution": "cache_reuse_no_model_inference",
                                                   "cache_audio_sha256": fingerprint}}}
    else:
        from stages.recognition.models import diarize_nemotron3
        diarization = diarize_nemotron3(path, args, duration)
    write_json(output.with_name(output.stem + "_diarization.json"), diarization)
    result = {"duration": duration, "language": asr["language"], "source_audio": str(path),
              "source_video": args.video, "audio_sha256": fingerprint,
              "segments": label_words(asr, diarization["diarization"]),
              "diarization": diarization["diarization"],
              "models": {**asr["models"], **diarization["models"]}}
    write_json(output, result)
    print(f"Audio analysis: {output}", flush=True)


if __name__ == "__main__":
    main()
