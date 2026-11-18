"""Media transport, ROI bounds, and cache metadata helpers."""

import hashlib
import json
import math
import subprocess
from pathlib import Path



def model_boxes(job, width, height):
    first, last = job["start_frame"], job["end_frame"]
    records = job["roi_records"]
    if [row["frame"] for row in records] != list(range(first, last)):
        raise ValueError(f"Incomplete ROI timeline: {job['id']}")
    boxes = []
    for row in records:
        x1, y1, x2, y2 = row["bbox"]
        box = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
        if not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height:
            raise ValueError(f"ROI outside image: {job['id']}, frame {row['frame']}")
        boxes.append(box)
    return boxes


def load_source_frames(video_path, expected_frames, start_frame=0, end_frame=None):
    """Decode only a job's frames on the OpenCV timeline used by ROI and composition."""
    import cv2
    from PIL import Image

    end_frame = expected_frames if end_frame is None else end_frame
    if not 0 <= start_frame < end_frame <= expected_frames:
        raise ValueError(f"Invalid source frame range [{start_frame}, {end_frame}) of {expected_frames}")
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open source video: {video_path}")
        position = start_frame if start_frame and capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame) else 0
        while position < end_frame:
            valid, bgr = capture.read()
            if not valid:
                break
            if position >= start_frame:
                frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            position += 1
        if start_frame == 0 and end_frame == expected_frames and capture.read()[0]:
            raise ValueError(f"Source frames changed since analysis: expected {expected_frames}, decoded more")
    finally:
        capture.release()
    if len(frames) != end_frame - start_frame:
        raise ValueError(f"Source frames changed since analysis: expected {end_frame - start_frame}, decoded {len(frames)}")
    return frames


def write_crop_video(frames, path):
    # RGB lossless transport avoids another chroma-subsampling pass at the face.
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "512x512", "-r", "25",
               "-i", "pipe:0", "-an", "-c:v", "libx264rgb", "-crf", "0",
               "-preset", "fast", "-threads", "4", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame.convert("RGB").tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f"Crop encoding failed: {path}")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_signature(path):
    path = Path(path).resolve()
    info = path.stat()
    return {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


