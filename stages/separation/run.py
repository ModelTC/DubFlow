"""SAM-Audio human-dialogue separation on the original 48 kHz stereo timeline."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from stages.separation.model import SAMSession
from stages.separation.utils import (
    analysis_intervals, chunk_anchors, implementation_signature, json_hash,
    model_signature, window_weights, windows,
)

from stages.postprocess.utils import (
    SAMPLE_RATE,
    extract_audio,
    file_hash,
    fit_tail,
    read_stereo,
    resume_output,
    write_audio,
    write_json,
)

SAM_REVISION = "bb4c6999d2677c7402360e426afc01ddfad6dce0"
DESCRIPTION = "speech"
CHANNEL_POLICY = "shared_seed_left_right_dual_mono_reuse_v1"
ANCHOR_POLICY = "aligned_human_source_parts_positive_v1"


def separate_windows(source, session, output, description, chunk_seconds, overlap_seconds, seed,
                     speech_intervals=None):
    chunk, overlap = round(chunk_seconds * SAMPLE_RATE), round(overlap_seconds * SAMPLE_RATE)
    accumulated = [np.zeros(source.shape, dtype=np.float64) for _ in range(2)]
    normalization = np.zeros(len(source), dtype=np.float64)
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    records = []
    for index, (start, end) in enumerate(windows(len(source), chunk, overlap)):
        paths = [chunks / f"{index:05d}_{name}.wav" for name in ("target", "residual")]
        anchors = chunk_anchors(speech_intervals, start, end)
        anchor_hash = json_hash(anchors)
        cache_request = {"recipe": 3, "start_sample": start, "end_sample": end,
                         "source_sha256": hashlib.sha256(source[start:end].tobytes()).hexdigest(),
                         "description": description, "seed": seed + index * 2,
                         "anchors": anchors, "anchor_sha256": anchor_hash}
        marker = chunks / f"{index:05d}_request.json"
        reused = (all(path.is_file() for path in paths) and marker.is_file()
                  and json.loads(marker.read_text()) == cache_request)
        if reused:
            tracks = [read_stereo(path) for path in paths]
        elif anchors == []:
            # No accepted dialogue in this chunk: do not guess from "speech" alone.
            tracks = [np.zeros_like(source[start:end]), source[start:end].copy()]
        else:
            kwargs = {} if anchors is None else {"anchors": anchors}
            tracks = session.separate(source[start:end], description, seed + index * 2, **kwargs)
        if any(track.shape != source[start:end].shape or not np.isfinite(track).all() for track in tracks):
            raise ValueError("SAM chunk changed the source sample grid")
        weights = window_weights(start, end, len(source), overlap)
        for values, track, path in zip(accumulated, tracks, paths, strict=True):
            if not reused:
                write_audio(path, track, "FLOAT")
            values[start:end] += track * weights[:, None]
        if not reused:
            write_json(marker, cache_request)
        normalization[start:end] += weights
        records.append({"index": index, "start_sample": start, "end_sample": end,
                        "seed": seed + index * 2, "cache_reused": reused,
                        "anchors": anchors, "anchor_sha256": anchor_hash,
                        "separation_skipped": anchors == []})
        print(f"SAM-Audio {index + 1}: {start / SAMPLE_RATE:.2f}–{end / SAMPLE_RATE:.2f}s", flush=True)
    target, residual = [(track / normalization[:, None]).astype(np.float32) for track in accumulated]
    return target, residual, records


def prepare_source(source_video, output_dir, duration, model_dir, text_encoder_dir,
                   device="cuda:0", description=DESCRIPTION, chunk_seconds=20., overlap_seconds=2., seed=42,
                   analysis_plan=None):
    if (not all(math.isfinite(value) for value in (duration, chunk_seconds, overlap_seconds))
            or duration <= 0 or not 0 <= overlap_seconds < chunk_seconds):
        raise ValueError("Require positive duration and 0 <= overlap < chunk size")
    source_video = Path(source_video).resolve()
    model_dir, text_encoder_dir = Path(model_dir).resolve(), Path(text_encoder_dir).resolve()
    speech_intervals, plan_signature = None, None
    if analysis_plan is not None:
        analysis_plan = Path(analysis_plan).resolve()
        speech_intervals = analysis_intervals(json.loads(analysis_plan.read_text()), duration)
        plan_signature = {"path": str(analysis_plan), "sha256": file_hash(analysis_plan)}
    anchors = {"analysis_plan": plan_signature, "anchor_policy": ANCHOR_POLICY if analysis_plan else "text_only",
               "no_dialogue_chunk_policy": "original_residual_zero_target" if analysis_plan else None,
               "global_speech_intervals": speech_intervals, "anchor_sha256": json_hash(speech_intervals)}
    request = {"recipe": 3, "source_video": str(source_video), "source_sha256": file_hash(source_video),
               "duration": duration, "models": model_signature(model_dir, text_encoder_dir),
               "description": description, "chunk_seconds": chunk_seconds, "overlap_seconds": overlap_seconds,
               "seed": seed, "device": device, "sam_revision": SAM_REVISION,
               "adapter": implementation_signature(),
               "predict_spans": False, "reranking_candidates": 1,
               "channels": CHANNEL_POLICY, "dtype": "float32", "background": "sam_predicted_residual", **anchors}
    names = ("source_mix_48k.wav", "source_original_48k.wav", "source_speech_48k.wav",
             "source_background_48k.wav", "source_removal_48k.wav")
    output, manifest, completed = resume_output(output_dir, "source_audio.json", duration, request, names)
    if completed is not None:
        for record in completed["metadata"]["outputs"].values():
            if file_hash(record["path"]) != record["sha256"]:
                raise ValueError("Completed SAM separation audio changed")
        return completed
    source = extract_audio(source_video, output / names[0], reuse=True)
    source, tail = fit_tail(source, round(duration * SAMPLE_RATE))
    session = None if speech_intervals == [] else SAMSession(model_dir, text_encoder_dir, device)
    try:
        speech, background, chunks = separate_windows(source, session, output, description, chunk_seconds, overlap_seconds, seed,
                                                     speech_intervals=speech_intervals)
    finally:
        if session is not None:
            session.close()
    # Native target is for voice cloning; replacement removes original - native residual.
    removal = source - background
    paths = {"original": output / names[1], "speech": output / names[2],
             "background": output / names[3], "removal": output / names[4]}
    for name, audio in (("original", source), ("speech", speech), ("background", background), ("removal", removal)):
        write_audio(paths[name], audio, "FLOAT")
    metadata = {"separation_model": model_dir.name, "separation_recipe": 3, **anchors,
                "checkpoint": str(model_dir / "checkpoint.pt"),
                "source_sha256": request["source_sha256"], "model_signature": request["models"],
                "description": description, "predict_spans": False, "reranking_candidates": 1,
                "source_mix_method": "decoded_original_mix", "source_background_method": "sam_predicted_residual",
                "source_removal_method": "original_mix_minus_sam_predicted_residual",
                "model_predicted_residual_used": session is not None, "channels": CHANNEL_POLICY,
                "chunk_seconds": chunk_seconds, "overlap_seconds": overlap_seconds, "chunks": chunks,
                "time_origin_seconds": 0., "tail": tail, "model_loaded": session is not None,
                "outputs": {name: {"path": str(path), "sha256": file_hash(path)} for name, path in paths.items()},
                "limitations": ["SAM residual may retain faint dialogue or separation artifacts; it is not guaranteed speech-free.",
                                "With an analysis plan, chunks without ASR human-dialogue anchors preserve the original audio; other chunks use the predicted residual.",
                                "Positive anchors cover recognized human dialogue only; ASR omissions remain possible."]}
    result = {"source_video": str(source_video), "duration": duration, "sample_rate": SAMPLE_RATE,
              "source_mix_path": str(paths["original"]), "source_speech_path": str(paths["speech"]),
              "source_background_path": str(paths["background"]), "source_removal_path": str(paths["removal"]),
              "metadata": metadata}
    write_json(manifest, result)
    return result


def run_stage(*, source_video, output_dir, duration, model_dir, text_encoder_dir,
              device="cuda:0", description=DESCRIPTION, chunk_seconds=20., overlap_seconds=2.,
              seed=42, analysis_plan=None):
    return prepare_source(source_video, output_dir, duration, model_dir, text_encoder_dir,
                          device, description, chunk_seconds, overlap_seconds, seed,
                          analysis_plan=analysis_plan)

