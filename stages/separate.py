"""SAM-Audio human-dialogue separation on the original 48 kHz stereo timeline."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from runtime.models import get

from stages.output.audio import (
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


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def analysis_intervals(plan, duration):
    """Use original ASR parts, not a merged turn spanning intervening pauses."""
    fps = plan.get("fps")
    if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or
            not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0):
        raise ValueError("Analysis plan requires finite positive fps and source duration")
    intervals = []
    for utterance in plan["utterances"]:
        text = utterance.get("text")
        if utterance.get("human_speech") is not True or not isinstance(text, str) or not text.strip():
            continue
        for part in utterance.get("source_parts") or [utterance]:
            part_text = part.get("text", text)
            if part.get("human_speech", True) is not True or not isinstance(part_text, str) or not part_text.strip():
                continue
            interval = part.get("source_interval", part)
            start, end = interval.get("start"), interval.get("end")
            if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in (start, end)) or not 0 <= start < end or start >= duration or
                    end > duration + 1 / fps + 1e-9):
                raise ValueError(f"ASR speech interval is outside the source timeline: {interval}")
            intervals.append({"start": float(start), "end": min(float(end), duration)})
    merged = []
    for interval in sorted(intervals, key=lambda row: (row["start"], row["end"])):
        if merged and interval["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], interval["end"])
        else:
            merged.append(interval.copy())
    return merged


def chunk_anchors(intervals, start, end):
    """Clip source-time intervals to a chunk and express seconds from its start."""
    if intervals is None:
        return None  # Standalone separation can use a text prompt without a transcript.
    first, last = start / SAMPLE_RATE, end / SAMPLE_RATE
    return [["+", max(first, row["start"]) - first, min(last, row["end"]) - first]
            for row in intervals if max(first, row["start"]) < min(last, row["end"])]


def file_signature(path):
    path = Path(path).resolve()
    info = path.stat()
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def model_signature(model_dir, text_encoder_dir):
    for name in ("config.json", "checkpoint.pt"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    if not (text_encoder_dir / "config.json").is_file():
        raise FileNotFoundError(text_encoder_dir / "config.json")
    files = [model_dir / "config.json", model_dir / "checkpoint.pt"]
    files += [path for path in sorted(text_encoder_dir.rglob("*"))
              if path.is_file() and ".cache" not in path.parts
              and path.suffix in (".json", ".safetensors", ".bin", ".model")]
    if not any(path.suffix in (".safetensors", ".bin") for path in files[2:]):
        raise FileNotFoundError(f"Missing local T5 weights: {text_encoder_dir}")
    if not any((text_encoder_dir / name).is_file() for name in ("tokenizer.json", "spiece.model")):
        raise FileNotFoundError(f"Missing local T5 tokenizer: {text_encoder_dir}")
    return [file_signature(path) for path in files]


def windows(samples, chunk_samples, overlap_samples):
    if not 0 <= overlap_samples < chunk_samples:
        raise ValueError("Require 0 <= overlap < chunk size")
    start = 0
    while start < samples:
        end = min(start + chunk_samples, samples)
        yield start, end
        if end == samples:
            break
        start += chunk_samples - overlap_samples


def window_weights(start, end, samples, overlap):
    weights = np.ones(end - start, dtype=np.float64)
    edge = min(overlap, len(weights))
    if edge:
        ramp = np.sin(np.linspace(0, np.pi / 2, edge + 2)[1:-1]) ** 2
        if start:
            weights[:edge] *= ramp
        if end < samples:
            weights[-edge:] *= ramp[::-1]
    return weights


def load_sam(model_dir, text_encoder_dir, device):
    model_dir, text_encoder_dir = Path(model_dir).resolve(), Path(text_encoder_dir).resolve()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MODELSCOPE_OFFLINE="1")
    import torch
    from sam_audio import SAMAudio, SAMAudioProcessor
    from sam_audio.model.config import SAMAudioConfig

    def load():
        config = json.loads((model_dir / "config.json").read_text())
        config["text_encoder"] = {**config.get("text_encoder", {}), "name": str(text_encoder_dir)}
        config.update(span_predictor=None, visual_ranker=None, text_ranker=None)
        model = SAMAudio(SAMAudioConfig(**config))
        model.load_state_dict(torch.load(model_dir / "checkpoint.pt", map_location="cpu", weights_only=True), strict=True)
        return model.eval().to(device), SAMAudioProcessor.from_pretrained(str(model_dir))

    return get(("sam_audio", str(model_dir), str(text_encoder_dir), device), load)


class SAMSession:
    def __init__(self, model_dir, text_encoder_dir, device):
        import torch

        self.torch, self.device = torch, device
        self.model, self.processor = load_sam(model_dir, text_encoder_dir, device)
        if self.processor.audio_sampling_rate != SAMPLE_RATE:
            raise ValueError("SAM checkpoint must use the 48 kHz source timeline")

    def separate(self, audio, description, seed, anchors=None):
        if anchors == []:
            return np.zeros_like(audio), audio.copy()
        torch = self.torch
        target, residual = np.empty_like(audio), np.empty_like(audio)
        hop = self.processor.audio_hop_length
        device = torch.device(self.device)
        devices = [device.index or 0] if device.type == "cuda" else []
        # SAM averages input channels. Preserve stereo, reusing exact dual-mono input.
        dual_mono = np.array_equal(audio[:, 0], audio[:, 1])
        for channel in range(1 if dual_mono else 2):
            values = torch.from_numpy(audio[:, channel].copy()).unsqueeze(0)
            if values.shape[-1] < hop:
                values = torch.nn.functional.pad(values, (0, hop - values.shape[-1]))
            kwargs = {} if anchors is None else {"anchors": [anchors]}
            batch = self.processor(audios=[values], descriptions=[description], **kwargs).to(self.device)
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    torch.cuda.default_generators[devices[0]].manual_seed(seed)
                separated = self.model.separate(batch, predict_spans=False, reranking_candidates=1)
            for track, prediction in ((target, separated.target[0]), (residual, separated.residual[0])):
                values = prediction.detach().float().cpu().numpy().reshape(-1)
                if len(values) < len(audio) or not np.isfinite(values).all():
                    raise ValueError("SAM returned incomplete or nonfinite target/residual")
                # Codec output is hop-padded; remove only its tail padding, never retime speech.
                track[:, channel] = values[:len(audio)]
        if dual_mono:
            target[:, 1], residual[:, 1] = target[:, 0], residual[:, 0]
        return target, residual

    def close(self):
        del self.model, self.processor
        gc.collect()
        if self.device.startswith("cuda"):
            with self.torch.cuda.device(self.device):
                self.torch.cuda.empty_cache()


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
               "adapter": file_signature(__file__), "predict_spans": False, "reranking_candidates": 1,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--analysis-plan", help="Aligned ASR plan supplying positive speech time anchors")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text-encoder-dir", help="Defaults to model-dir/t5-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--description", default=DESCRIPTION)
    parser.add_argument("--chunk-seconds", type=float, default=20.)
    parser.add_argument("--overlap-seconds", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    text_encoder = args.text_encoder_dir or str(Path(args.model_dir) / "t5-base")
    prepare_source(args.source_video, args.output_dir, args.duration, args.model_dir, text_encoder,
                   args.device, args.description, args.chunk_seconds, args.overlap_seconds, args.seed,
                   analysis_plan=args.analysis_plan)
    print(Path(args.output_dir).resolve() / "source_audio.json")


if __name__ == "__main__":
    main()
