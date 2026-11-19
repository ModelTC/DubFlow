"""Qwen3-ASR、强对齐和 Nemotron 声学说话人标注。"""

import hashlib
import json
from pathlib import Path

from stages.recognition.utils import write_json, source_audio, label_words


def detect_speakers(path, duration, fingerprint, diar_model, device, cache):
    if cache:
        cache_path = Path(cache).resolve()
        cached = json.loads(cache_path.read_text())
        cached_model = cached["models"]["diarization"]
        expected_backend, model_path = "hf_nemotron3", diar_model
        if (cached["audio_sha256"] != fingerprint or cached_model["backend"] != expected_backend or
                Path(cached_model["model"]).resolve() != Path(model_path).resolve()):
            raise ValueError("Diarization cache belongs to another source waveform, backend or model")
        diarization = {"diarization": cached["diarization"],
                       "models": {"diarization": {**cached_model, "cached_from": str(cache_path),
                                                   "execution": "cache_reuse_no_model_inference",
                                                   "cache_audio_sha256": fingerprint}}}
    else:
        from stages.recognition.model import diarize_nemotron3
        diarization = diarize_nemotron3(path, duration, model_path=diar_model, device=device)
    return diarization


def run_stage(*, output, language, asr_model, align_model, diar_model,
              audio=None, video=None, device="cuda:0", diarization_cache=None, stage="all"):
    path, duration = source_audio(audio=audio, video=video, output=output)
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    output = Path(output)
    asr_path = output.with_name("audio_asr.json")
    if stage != "asr":
        diarization = detect_speakers(path, duration, fingerprint, diar_model, device, diarization_cache)
        write_json(output.with_name(output.stem + "_diarization.json"), diarization)
    if stage in ("asr", "all"):
        from stages.recognition.model import transcribe_qwen3
        asr = transcribe_qwen3(path, asr_model=asr_model, align_model=align_model, language=language,
                               device=device, turns=diarization["diarization"] if stage == "all" else ())
        asr.update(audio_sha256=fingerprint, duration=duration, source_audio=str(path))
        write_json(asr_path, asr)
    else:
        asr = json.loads(asr_path.read_text())
        if asr["audio_sha256"] != fingerprint:
            raise ValueError("ASR cache belongs to another source audio")
        from stages.recognition.model import ASR_POLICY
        if asr["models"]["asr"].get("recognition_policy") != ASR_POLICY:
            raise ValueError("Qwen3 ASR cache uses an older recognition policy; regenerate ASR in a new output directory")
    if stage == "asr":
        print(f"ASR and alignment: {asr_path}", flush=True)
        return

    result = {"duration": duration, "language": asr["language"], "source_audio": str(path),
              "source_video": video, "audio_sha256": fingerprint,
              "segments": label_words(asr, diarization["diarization"]),
              "diarization": diarization["diarization"],
              "models": {**asr["models"], **diarization["models"]}}
    write_json(output, result)
    print(f"Audio analysis: {output}", flush=True)
