"""Speaker-analysis geometry, checkpoint I/O, and bounded frame prefetch."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def file_signature(path):
    path = Path(path).resolve()
    info = path.stat()
    return {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def track_signature(track):
    values = [track[key] for key in ("frames", "bboxes", "confidence")]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-8)


def numpy_nms(boxes, threshold):
    """Official final cross-scale NMS, without its removed NumPy np.int alias."""
    order = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while len(order):
        first = int(order[0])
        keep.append(first)
        order = np.asarray([index for index in order[1:] if box_iou(boxes[first], boxes[index]) <= threshold], dtype=np.int64)
    return np.asarray(keep, dtype=np.int64)


def save_detections(path, settings, detections):
    write_json(path, {"settings": settings, "detections": detections})


class AnalysisFrames:
    """Decode and save the next JPEG while the GPU analyzes the current frame."""

    def __init__(self, video, frame_dir, start=0):
        self.video = video
        self.frame_dir = Path(frame_dir)
        self.index = start

    def __enter__(self):
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.capture = cv2.VideoCapture(str(self.video))
        if not self.capture.isOpened():
            raise OSError(f"Cannot open analysis video: {self.video}")
        if self.index and not self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.index):
            self.capture.release()
            raise RuntimeError("Cannot seek to the detection checkpoint")
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis-frames")
        self.pending = self.executor.submit(self.read)
        return self

    def read(self):
        valid, image = self.capture.read()
        if not valid:
            return None
        index = self.index
        path = self.frame_dir / f"{index:07d}.jpg"
        temporary = path.with_suffix(".tmp.jpg")
        if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Cannot save analysis frame {index}")
        temporary.replace(path)
        self.index += 1
        return index, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __iter__(self):
        return self

    def __next__(self):
        result = self.pending.result()
        if result is None:
            raise StopIteration
        self.pending = self.executor.submit(self.read)
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.capture.release()


ARCFACE_LANDMARKS = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32,
)


def unit_vector(value):
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if vector.size == 0 or not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Face embedding must be finite and nonzero")
    return vector / norm


def box_overlaps(box, boxes):
    boxes = np.asarray(boxes)
    lower = np.maximum(box[:2], boxes[:, :2])
    upper = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(upper - lower, 0), axis=1)
    area = np.prod(np.maximum(box[2:] - box[:2], 0))
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-8)


def align_landmarks(landmarks):
    """Least-squares similarity transform, without a stochastic RANSAC fit."""
    source = np.asarray(landmarks, dtype=np.float64)
    target = ARCFACE_LANDMARKS.astype(np.float64)
    source_center, target_center = source.mean(0), target.mean(0)
    source_zero, target_zero = source - source_center, target - target_center
    u, singular, vt = np.linalg.svd(target_zero.T @ source_zero / len(source))
    sign = np.diag([1.0, np.linalg.det(u @ vt)])
    rotation = u @ sign @ vt
    variance = np.sum(source_zero**2) / len(source)
    if variance < 1e-8:
        raise ValueError("Degenerate facial landmarks")
    scale = np.sum(singular * np.diag(sign)) / variance
    translation = target_center - scale * rotation @ source_center
    return np.column_stack((scale * rotation, translation)).astype(np.float32)

