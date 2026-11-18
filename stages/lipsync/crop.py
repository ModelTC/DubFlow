"""直接用已绑定的 S3FD 框裁剪，复用原生关键点、平滑与贴回。"""

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from runtime.utils import write_json
from types import SimpleNamespace

from runtime.models import is_enabled
from runtime.execution import call_function
from stages.lipsync.model import load_landmarker
from stages.lipsync.utils import fingerprint, file_signature


def bound_face_box(record, width, height):
    x1, y1, x2, y2 = record["bbox"]
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError("Bound face box is outside the source frame")
    box = SimpleNamespace(origin_x=int(x1), origin_y=int(y1), width=int(x2 - x1), height=int(y2 - y1))
    if min(box.width, box.height) == 0:
        raise ValueError("Bound face box is smaller than one pixel")
    return box


def track_boxes(anchors, width, height):
    """Use a stable square crop around a tracked face without human landmarks."""
    import numpy as np

    if min(width, height) < 2:
        raise ValueError("Source frame is too small for a tracked-face crop")
    samples = []
    for anchor in anchors:
        bound_face_box(anchor, width, height)
        x1, y1, x2, y2 = anchor["bbox"]
        samples.append(((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)))
    samples = np.asarray(samples, dtype=np.float64)
    for reducer in (np.median, np.mean):
        samples = np.asarray([reducer(samples[max(0, i - 3):i + 4], axis=0)
                              for i in range(len(samples))])
    boxes = []
    for center_x, center_y, face_side in samples:
        side = min(max(2, math.ceil(face_side * 1.45)), width, height)
        left = max(0, min(round(center_x - side / 2), width - side))
        top = max(0, min(round(center_y - side / 2), height - side))
        boxes.append([left, top, left + side, top + side])
    return boxes


def track_bbox_worker(video, job, report_path):
    import cv2
    import numpy as np

    anchors = job["roi_records"]
    started = time.perf_counter()
    report = {
        "backend": "bound_s3fd_track_bbox", "bbox_source": "bound_s3fd_track",
        "model_path": None, "detector_model_loaded": False,
        "provided_bbox_frames": len(anchors),
        "observed_bbox_frames": sum(record["observed"] for record in anchors),
        "request_signature": job.get("roi_cache_signature"),
        "frames": job["end_frame"] - job["start_frame"], "decoded_frames": 0,
        "frame_diagnostics": [],
    }
    capture = cv2.VideoCapture(video)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video frame rate: {fps}")
        report["source_fps"] = fps
        if report["frames"] <= 0 or len(anchors) != report["frames"]:
            raise ValueError("Bound face track does not cover the model window")
        capture.set(cv2.CAP_PROP_POS_FRAMES, job["start_frame"])
        size = None
        for index, anchor in enumerate(anchors):
            if anchor["frame"] != job["start_frame"] + index:
                raise ValueError("Bound face track is not aligned with the model window")
            valid, pixels = capture.read()
            if not valid:
                raise ValueError("Source ended before the model window")
            report["decoded_frames"] += 1
            height, width = pixels.shape[:2]
            if size is not None and size != (width, height):
                raise ValueError("Source dimensions changed within the model window")
            size = (width, height)
            bound_face_box(anchor, width, height)
            report["frame_diagnostics"].append({
                "frame": index, "source_frame": anchor["frame"],
                "bbox_observed": anchor["observed"], "bbox_confidence": anchor.get("confidence"),
            })
        report.update(success=True, bboxes=track_boxes(anchors, *size), crop_mode="track_bbox",
                      interpolated_frames=[i for i, anchor in enumerate(anchors) if not anchor["observed"]],
                      landmark_frames=0)
    except Exception as error:
        report.update(success=False, error=str(error))
    finally:
        capture.release()
        report["preprocessing_seconds"] = time.perf_counter() - started
        write_json(report_path, report)
    return report


def run_worker(*, video, job, report, landmarker):
    report_path = Path(report)
    job = json.loads(Path(job).read_text())
    if job.get("roi_mode", "landmarks") == "track_bbox":
        return track_bbox_worker(video, job, report_path)
    import cv2
    import mediapipe as mp
    import numpy as np

    from stages.lipsync import geometry as native

    anchors = job["roi_records"]
    started = time.perf_counter()
    report = {
        "backend": "bound_s3fd_bbox_mediapipe_landmarker", "bbox_source": "bound_s3fd_track",
        "version": mp.__version__, "delegate": "CPU", "model_path": str(Path(landmarker).resolve()),
        "detector_model_loaded": False, "provided_bbox_frames": len(anchors),
        "observed_bbox_frames": sum(record["observed"] for record in anchors),
        "request_signature": job.get("roi_cache_signature"),
        "frames": job["end_frame"] - job["start_frame"], "decoded_frames": 0,
        "confidence_thresholds": {"presence": 0.5, "tracking": 0.5}, "frame_diagnostics": [],
    }
    capture = cv2.VideoCapture(video)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video frame rate: {fps}")
        report["source_fps"] = fps
        if report["frames"] <= 0 or len(anchors) != report["frames"]:
            raise ValueError("Bound face track does not cover the model window")
        capture.set(cv2.CAP_PROP_POS_FRAMES, job["start_frame"])
        landmark_started = time.perf_counter()
        landmark_model = load_landmarker(landmarker)
        landmarks = []
        size = None
        # IMAGE 模式没有跨视频时间状态；常驻模型只在进程退出时关闭。
        with nullcontext(landmark_model) if is_enabled() else landmark_model as landmarker:
            for index, anchor in enumerate(anchors):
                if anchor["frame"] != job["start_frame"] + index:
                    raise ValueError("Bound face track is not aligned with the model window")
                valid, pixels = capture.read()
                if not valid:
                    raise ValueError("Source ended before the model window")
                report["decoded_frames"] += 1
                height, width = pixels.shape[:2]
                if size is not None and size != (width, height):
                    raise ValueError("Source dimensions changed within the model window")
                size = (width, height)
                box = bound_face_box(anchor, width, height)
                crop, matrix = native.crop_detected_face(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB), box)
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
                report["frame_diagnostics"].append({
                    "frame": index, "source_frame": anchor["frame"], "landmark_faces": len(result.face_landmarks),
                    "bbox_observed": anchor["observed"], "bbox_confidence": anchor.get("confidence"),
                    "detection_box": [box.origin_x, box.origin_y, box.width, box.height],
                })
                points = None
                if result.face_landmarks:
                    local = np.array([(point.x * 256, point.y * 256, 1)
                                      for point in result.face_landmarks[0]], dtype=np.float32)
                    points = (local @ matrix.T) / np.array(size, dtype=np.float32)
                landmarks.append(points)
        report.update(landmark_seconds=time.perf_counter() - landmark_started,
                      landmark_frames=sum(item is not None for item in landmarks),
                      missing_frames=[index for index, points in enumerate(landmarks) if points is None])
        max_gap = max(1, round(fps * 0.4))
        missing = report["missing_frames"]
        gaps = np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1)
        if not report["landmark_frames"] or max(map(len, gaps)) > max_gap:
            attempt = {key: report[key] for key in ("backend", "landmark_frames", "missing_frames",
                                                    "frame_diagnostics", "landmark_seconds")}
            attempt.update(reason="no_landmarks" if not report["landmark_frames"] else "landmark_gap_exceeds_limit",
                           max_gap_frames=max_gap)
            report = track_bbox_worker(video, job, report_path)
            report["landmark_fallback"] = attempt
        else:
            points, interpolated = native.fill_short_gaps(landmarks, max_gap_frames=max_gap)
            boxes = native.build_face_bbox_from_landmarks(points, *size)
            boxes, case_flag = native.process_bbox(boxes, *size, force_fix=False)
            report.update(success=True, interpolated_frames=interpolated, crop_mode=case_flag,
                          bboxes=boxes, landmark_count=points.shape[1])
    except Exception as error:
        report.update(success=False, error=str(error))
    finally:
        capture.release()
        report["preprocessing_seconds"] = time.perf_counter() - started
        write_json(report_path, report)
    return report


def run_stage(*, plan, output, landmarker):
    output = Path(output)
    plan = json.loads(Path(plan).read_text())
    landmarker = Path(landmarker)
    needs_landmarker = any(job["action"] == "tbdub" and job.get("roi_mode", "landmarks") == "landmarks"
                           for job in plan["jobs"])
    if needs_landmarker and not landmarker.is_file():
        raise FileNotFoundError(landmarker)
    context = {"source_video": file_signature(plan["source_video"]),
               "landmarker": file_signature(landmarker) if needs_landmarker else None,
               "native_preprocessor": file_signature(Path(__file__).with_name("geometry.py")) if needs_landmarker else None,
               "adapter_version": 4}
    signature = fingerprint({"context": context, "plan": plan})
    if output.exists():
        if json.loads(output.read_text()).get("roi_preparation_signature") != signature:
            raise ValueError("Existing native ROI output belongs to another request")
        print(f"Retaining complete native ROI plan: {output}")
        return
    directory = output.parent / "rois"
    directory.mkdir(parents=True, exist_ok=True)
    for job in plan["jobs"]:
        if job["action"] == "audio_only":
            continue
        input_path, report_path = directory / f"{job['id']}_input.json", directory / f"{job['id']}.json"
        job_signature = fingerprint({"context": context, "job": job})
        report = json.loads(report_path.read_text()) if report_path.is_file() else None
        if report is None or report.get("request_signature") != job_signature or not report["success"]:
            write_json(input_path, {**job, "roi_cache_signature": job_signature})
            parameters = dict(video=plan["source_video"], job=str(input_path),
                              report=str(report_path), landmarker=str(landmarker))
            with (directory / f"{job['id']}.log").open("a") as log:
                if is_enabled():
                    run_worker(**parameters)
                else:
                    result = call_function(sys.executable, "stages.lipsync.crop", "run_worker", parameters,
                                           capture=True, check=False, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                    log.write((result.stdout or "") + (result.stderr or ""))
        if not report_path.exists():
            raise RuntimeError(f"ROI worker failed before writing a report: {job['id']}")
        report = json.loads(report_path.read_text())
        if report.get("request_signature") != job_signature:
            raise RuntimeError(f"ROI worker did not commit a matching report: {job['id']}")
        job["native_roi_report"] = str(report_path)
        if not report["success"]:
            raise RuntimeError(f"ROI preparation failed for {job['id']}: {report['error']}")
        if report.get("landmark_fallback"):
            job.update(roi_mode="track_bbox", roi_fallback_reason=report["landmark_fallback"]["reason"])
        if len(report["bboxes"]) != job["end_frame"] - job["start_frame"]:
            raise ValueError(f"Incomplete native ROI result: {job['id']}")
        interpolated = set(report["interpolated_frames"])
        job["roi_records"] = [{"frame": anchor["frame"], "bbox": box, "face_bbox": anchor["bbox"],
                               "observed": index not in interpolated}
                              for index, (anchor, box) in enumerate(zip(job["roi_records"], report["bboxes"]))]
    plan["roi_preparation_signature"] = signature
    write_json(output, plan)
