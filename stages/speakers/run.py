"""同镜头人脸跟踪与 UniTalk-LoCoNet 发言评分；frames/bboxes 只记录真实检测。"""

from __future__ import annotations

import json
import logging
import warnings
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
from scenedetect import ContentDetector, detect
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment

from stages.speakers.active_speaker import ASD_POLICY, CONTEXT_FRAMES, MAX_WINDOW_FRAMES, score_track
from stages.speakers.model import load_grounding_detector, load_loconet, load_s3fd
from stages.speakers.utils import AnalysisFrames, box_iou, file_signature, save_detections, track_signature, write_json

LOG = logging.getLogger(__name__)


def _scene_list(video_path, total_frames, threshold, min_scene_len):
    scenes = detect(str(video_path), ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    boundaries = sorted({0, total_frames, *[int(start.get_frames()) for start, _ in scenes]})
    return [{"id": f"shot_{index:03d}", "start_frame": start, "end_frame": end}
            for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])) if end > start]


def detect_faces(video_path, frame_dir, device, scale, confidence, minimum_size,
                 detections=None, checkpoint=None, *, weights):
    detector = load_s3fd(weights, device)
    detections = list(detections or [])
    try:
        with AnalysisFrames(video_path, frame_dir, start=len(detections)) as frames:
            for index, image in frames:
                height, width = image.shape[:2]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="An output with one or more elements was resized.*",
                                            category=UserWarning, module=r"stages\.speakers\.models\.s3fd\.utils")
                    boxes = detector.detect_faces(image, conf_th=confidence, scales=[scale])
                current = []
                for box in boxes:
                    x1, y1, x2, y2, score = map(float, box)
                    clipped = [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]
                    if min(clipped[2] - clipped[0], clipped[3] - clipped[1]) >= minimum_size:
                        current.append({"bbox": clipped, "confidence": score})
                detections.append(current)
                if checkpoint and len(detections) % 100 == 0:
                    checkpoint(detections)
                if index % 50 == 0:
                    LOG.info("Detected frame %d: %d faces", index, len(current))
    finally:
        if checkpoint:
            checkpoint(detections)
        del detector
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return detections


def _track_faces(detections, shots, max_gap, iou_threshold):
    """Associate actual detections within each shot; never track across a cut."""
    tracks = []
    for shot in shots:
        active = []
        for frame in range(shot["start_frame"], shot["end_frame"]):
            active = [track for track in active if frame - track["frames"][-1] <= max_gap + 1]
            faces = detections[frame]
            matched = set()
            if active and faces:
                similarities = np.asarray([[box_iou(track["bboxes"][-1], face["bbox"]) for face in faces] for track in active])
                rows, columns = linear_sum_assignment(1 - similarities)
                for row, column in zip(rows, columns):
                    if similarities[row, column] >= iou_threshold:
                        track = active[int(row)]
                        face = faces[int(column)]
                        track["frames"].append(frame)
                        track["bboxes"].append(face["bbox"])
                        track["confidence"].append(face["confidence"])
                        matched.add(int(column))
            for index, face in enumerate(faces):
                if index not in matched:
                    track = {"id": f"face_{len(tracks):04d}", "shot_id": shot["id"], "frames": [frame],
                             "bboxes": [face["bbox"]], "confidence": [face["confidence"]]}
                    tracks.append(track)
                    active.append(track)
    for track in tracks:
        frames = track["frames"]
        track.update(start_frame=frames[0], end_frame=frames[-1] + 1, detected=[True] * len(frames))
        track["gaps"] = [{"start_frame": first + 1, "end_frame": second}
                         for first, second in zip(frames, frames[1:]) if second > first + 1]
    return tracks


def grounded_faces(frame_dir, shots, detections, model_path, device, minimum_size=24):
    """Retry sparsely detected shots with open-vocabulary heads, not every frame twice."""
    from PIL import Image

    sparse = [shot for shot in shots if sum(bool(detections[i]) for i in
              range(shot["start_frame"], shot["end_frame"])) < .7 * (shot["end_frame"] - shot["start_frame"])]
    if not sparse:
        return detections, []

    model, processor = load_grounding_detector(model_path, device)
    result = list(detections)
    try:
        for shot in sparse:
            for index in range(shot["start_frame"], shot["end_frame"]):
                with Image.open(frame_dir / f"{index:07d}.jpg") as image:
                    image = image.convert("RGB")
                    inputs = processor(images=image, text="a human face. a cat face. a dog face. a cartoon face.",
                                       return_tensors="pt").to(device)
                    with torch.inference_mode():
                        outputs = model(**inputs)
                    found = processor.post_process_grounded_object_detection(
                        outputs, inputs.input_ids, threshold=.28, text_threshold=.25,
                        target_sizes=[image.size[::-1]])[0]
                    boxes = found["boxes"].cpu().numpy()
                    scores = found["scores"].cpu().numpy()
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image.width)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image.height)
                candidates = [{"bbox": box.tolist(), "confidence": float(score)}
                              for box, score in zip(boxes, scores)
                              if min(box[2] - box[0], box[3] - box[1]) >= minimum_size]
                selected = []
                for candidate in sorted(candidates, key=lambda row: -row["confidence"]):
                    a = candidate["bbox"]
                    duplicate = False
                    for previous in selected:
                        b = previous["bbox"]
                        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                        smaller = min((a[2]-a[0]) * (a[3]-a[1]), (b[2]-b[0]) * (b[3]-b[1]))
                        if box_iou(a, b) > .3 or intersection / smaller > .8:
                            duplicate = True
                            break
                    if not duplicate:
                        selected.append(candidate)
                result[index] = selected or detections[index]
            LOG.info("Open-vocabulary detection: %s", shot["id"])
    finally:
        del model, processor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return result, [shot["id"] for shot in sparse]


def analyze_video(video_path, output_json, work_dir, *, audio_path, detector_weights, asd_weights,
                  device="cuda:0",
                  scene_threshold=27.0, min_scene_len=15, detection_scale=1.0, confidence=0.5,
                  min_face_size=24, max_gap=10, iou_threshold=0.5, min_track_frames=10, grounding_model=None):
    """Analyze an already normalized 25fps video; intervals are half-open frames."""
    video_path, output_json = Path(video_path), Path(output_json)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = Path(audio_path)
    detector_weights, asd_weights = Path(detector_weights), Path(asd_weights)
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if abs(fps - 25) > 1e-3 or frame_count < 1:
        raise ValueError(f"Prepare an equal-duration 25fps working video first, got fps={fps}, frames={frame_count}")
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    cv2.setNumThreads(2)
    torch.set_num_threads(8)
    shots = _scene_list(video_path, frame_count, scene_threshold, min_scene_len)
    LOG.info("Scene detection: %d shots across %d frames", len(shots), frame_count)
    frame_dir = work_dir / "frames"
    cache_path = work_dir / "detections.json"
    settings = {"video": file_signature(video_path), "weights": file_signature(detector_weights),
                "scale": detection_scale, "confidence": confidence, "min_face_size": min_face_size}
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else None
    detections = cache["detections"] if cache is not None and cache["settings"] == settings else []
    if len(detections) > frame_count or any(not (frame_dir / f"{index:07d}.jpg").is_file()
                                            for index in range(len(detections))):
        raise ValueError("Detection checkpoint lost its source frames")
    if len(detections) == frame_count:
        LOG.info("Reusing exact-settings cached S3FD detections")
    else:
        LOG.info("Continuing S3FD at frame %d/%d", len(detections), frame_count)
        checkpoint = partial(save_detections, cache_path, settings)
        detections = detect_faces(video_path, frame_dir, device, detection_scale, confidence,
                                   min_face_size, detections, checkpoint, weights=detector_weights)
        if len(detections) != frame_count:
            raise ValueError(f"Decoded {len(detections)} frames, metadata says {frame_count}")
    grounded_shots = []
    if grounding_model:
        grounded_path = work_dir / "grounded_detections.json"
        grounded_settings = {**settings, "policy": "sparse_shot_heads_v1", "shots": shots,
                             "model": file_signature(Path(grounding_model) / "model.safetensors")}
        cached = json.loads(grounded_path.read_text()) if grounded_path.is_file() else None
        if cached is not None and cached["settings"] == grounded_settings:
            detections, grounded_shots = cached["detections"], cached["shots"]
        else:
            detections, grounded_shots = grounded_faces(frame_dir, shots, detections, grounding_model, device, min_face_size)
            write_json(grounded_path, {"settings": grounded_settings, "detections": detections, "shots": grounded_shots})
    tracks = _track_faces(detections, shots, max_gap, iou_threshold)
    result = {"fps": 25, "frames": frame_count, "width": width, "height": height,
              "video_path": str(video_path), "audio_path": str(audio_path), "shots": shots, "tracks": tracks,
              "models": {"detector": str(detector_weights), "asd": str(asd_weights),
                         "asd_backend": "unitalk_loconet", "asd_policy": ASD_POLICY,
                         "grounding_model": grounding_model, "grounded_shots": grounded_shots},
              "settings": {"scene_threshold": scene_threshold, "min_scene_len": min_scene_len,
                           "detection_scale": detection_scale, "confidence": confidence,
                           "min_face_size": min_face_size, "max_gap": max_gap,
                           "iou_threshold": iou_threshold, "min_track_frames": min_track_frames}}
    # ASD 必须配原片音频；译音不能用于判断原画面中的发言人。
    sample_rate, waveform = wavfile.read(audio_path)
    if sample_rate != 16000 or waveform.ndim != 1:
        raise ValueError("ASD expects the shared mono16kHz original waveform")
    asd_path = work_dir / "asd_cache.json"
    asd_settings = {"audio": file_signature(audio_path), "weights": file_signature(asd_weights),
                    "policy": ASD_POLICY, "max_window_frames": MAX_WINDOW_FRAMES, "context_frames": CONTEXT_FRAMES,
                    "min_track_frames": min_track_frames,
                    "tracks": [{"id": t["id"], "shot_id": t["shot_id"], "signature": track_signature(t)} for t in tracks]}
    asd = json.loads(asd_path.read_text()) if asd_path.is_file() else None
    scored = asd["tracks"] if asd is not None and asd["settings"] == asd_settings else {}
    contexts = [track for track in tracks if len(track["frames"]) >= min_track_frames]
    model = head = None
    try:
        for track in tracks:
            signature = track_signature(track)
            cached = scored.get(track["id"])
            if cached is not None and cached["signature"] == signature:
                if len(cached["scores"]) != len(track["frames"]):
                    raise ValueError(f"Incomplete ASD checkpoint: {track['id']}")
                track.update(asd_status=cached["status"], asd_scores=cached["scores"])
                if cached["status"] == "scored":
                    track.update(asd_probability=cached["probability"], asd_windows=cached["windows"])
                continue
            if len(track["frames"]) < min_track_frames:
                # 缺失评分是未知，不是静音。
                track.update(asd_status="insufficient_real_detections", asd_scores=[None] * len(track["frames"]))
            else:
                if model is None:
                    model, head = load_loconet(asd_weights, device)
                score_track(track, contexts, model, head, waveform, frame_dir, device)
                LOG.info("ASD %s: %d detections, %d gap frames, median score %.2f", track["id"], len(track["frames"]),
                         track["end_frame"] - track["start_frame"] - len(track["frames"]), float(np.median(track["asd_scores"])))
            scored[track["id"]] = {"signature": signature, "status": track["asd_status"], "scores": track["asd_scores"],
                                  "probability": track.get("asd_probability"), "windows": track.get("asd_windows")}
            write_json(asd_path, {"settings": asd_settings, "tracks": scored})
    finally:
        del model, head
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    write_json(output_json, result)
    LOG.info("Wrote %s: %d shots, %d tracks", output_json, len(shots), len(tracks))
    return result


def run_stage(*, video, audio, output, work_dir, s3fd_model, loconet_model,
              device="cuda:0", detection_scale=1., confidence=.5, grounding_model=None):
    if detection_scale <= 0 or not 0 < confidence <= 1:
        raise ValueError("Require positive detection scale and confidence in (0, 1]")
    return analyze_video(video, output, work_dir, audio_path=audio, device=device,
                         detector_weights=s3fd_model, asd_weights=loconet_model,
                         detection_scale=detection_scale, confidence=confidence, grounding_model=grounding_model)
