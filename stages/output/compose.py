"""按连续可见人物窗口贴回原生 ROI，保留窗口外原画，最终混音封装一次。"""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

COMPOSITION_POLICY = "continuous_track_roi_blend_v5"


def speech_windows(job, fps, join_gap=0):
    """Separate model context from the source/translated speech that needs editing."""
    first, last = job["edit_start_frame"], job["edit_end_frame"]
    guard = job["drive_guard_seconds"]
    source = job.get("source_speech_intervals") or [job["source_interval"]]
    target = job.get("target_speech_intervals") or [job["target_interval"]]
    windows = sorted((max(first, math.floor((item["start"] - guard) * fps)),
                      min(last, math.ceil((item["end"] + guard) * fps)))
                     for item in (*source, *target) if item is not None)
    merged = []
    for start, end in windows:
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= join_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def temporal_weight(job, frame, fps, seconds):
    if seconds < 0:
        raise ValueError("Transition seconds cannot be negative")
    transition = round(seconds * fps)
    if job.get("temporal_policy") == "continuous_visible_track_v1":
        start, end = job["edit_start_frame"], job["edit_end_frame"]
        if not start <= frame < end:
            return 0.0
        if not transition:
            return 1.0
        # Only fade at a track/protected-dialogue boundary, never at every
        # utterance. Keep fades wholly inside the supported visible interval.
        width = min(transition, max(1, (end - start) // 4))
        onset = (1.0 if job["edit_starts_at_shot_cut"] else
                 math.sin(math.pi / 2 * min(1.0, (frame - start + .5) / width)) ** 2)
        offset = (1.0 if job["edit_ends_at_shot_cut"] else
                  math.sin(math.pi / 2 * min(1.0, (end - frame - .5) / width)) ** 2)
        return min(onset, offset)
    # Conflicting or oversized tracks use bounded speech windows instead.
    for start, end in speech_windows(job, fps, join_gap=transition):
        if not transition:
            if start <= frame < end:
                return 1.0
            continue
        # A short utterance still needs a fully edited middle; fades must not
        # occupy most of its duration.
        width = min(transition, max(1, (end - start) // 4))
        half = width / 2
        if not start - half <= frame < end + half:
            continue
        onset = (1.0 if job["edit_starts_at_shot_cut"] and start == job["edit_start_frame"] else math.sin(
                     math.pi / 2 * min(1.0, max(0.0, (frame - start + half) / width))) ** 2)
        offset = (1.0 if job["edit_ends_at_shot_cut"] and end == job["edit_end_frame"] else math.sin(
                      math.pi / 2 * min(1.0, max(0.0, (end - frame + half) / width))) ** 2)
        return min(onset, offset)
    return 0.0


def index_jobs(plan, manifest):
    if manifest["status"] != "complete":
        raise ValueError("Render is incomplete")
    if any(manifest[key] != plan[key] for key in ("source_video", "target_audio_path", "fps", "frames")):
        raise ValueError("Render and plan belong to different timelines")
    results = {job["id"]: job for job in manifest["jobs"]}
    indexed = [[] for _ in range(plan["frames"])]
    for job in plan["jobs"]:
        if job["action"] == "audio_only":
            continue
        result = results[job["id"]]
        first, last = job["start_frame"], job["end_frame"]
        edit_first, edit_last = job["edit_start_frame"], job["edit_end_frame"]
        if (result["status"] != "complete" or result["frames_rendered"] != last - first or
                (result["start_frame"], result["end_frame"]) != (first, last)):
            raise ValueError(f"Incomplete frames: {job['id']}")
        if not 0 <= first <= edit_first < edit_last <= last <= plan["frames"]:
            raise ValueError(f"Edit window is outside model window: {job['id']}")
        records = {row["frame"]: row for row in result["roi_records"]}
        for frame in range(edit_first, edit_last):
            record = records[frame]
            box = record["bbox"]
            # PNG 的零点是模型窗口，不是最终贴回窗口。
            path = Path(result["frame_directory"]) / f"{frame - first:06d}.png"
            indexed[frame].append((job, path, box, record.get("face_bbox", box)))
    return indexed


def blend_patches(source, patches):
    """原生结果已经羽化；只在实际改动重叠处按离各自人脸的距离分配增量。"""
    if not patches:
        return source.copy()
    frame = source.astype(np.float32)
    height, width = source.shape[:2]
    total = np.zeros(source.shape, np.float32) if len(patches) > 1 else None
    priorities = np.zeros((height, width), np.float32) if len(patches) > 1 else None
    coverage = np.zeros((height, width), np.uint16) if len(patches) > 1 else None
    for pasted, box, face, weight in patches:
        if weight == 0:
            continue
        x1, y1, x2, y2 = box
        original = source[y1:y2, x1:x2].astype(np.float32)
        delta = (pasted[y1:y2, x1:x2].astype(np.float32) - original) * weight
        changed = np.any(delta != 0, axis=2)
        region = frame[y1:y2, x1:x2]
        region[changed] = (original + delta)[changed]
        if total is not None:
            fx1, fy1, fx2, fy2 = face
            x = (np.arange(x1, x2, dtype=np.float32) + .5 - (fx1 + fx2) / 2) / (fx2 - fx1)
            y = (np.arange(y1, y2, dtype=np.float32) + .5 - (fy1 + fy2) / 2) / (fy2 - fy1)
            priority = (1 + x[None, :] ** 2 + y[:, None] ** 2) ** -4 * changed * weight
            total[y1:y2, x1:x2] += delta * priority[..., None]
            priorities[y1:y2, x1:x2] += priority
            coverage[y1:y2, x1:x2] += changed
    if total is not None:
        overlap = coverage > 1
        # 增量保留时间渐变；淡入任务的空间占比也逐渐增加，不突然压低已有编辑。
        frame[overlap] = source[overlap] + total[overlap] / priorities[overlap, None]
    return np.clip(np.rint(frame), 0, 255).astype(np.uint8)


def compose(plan, manifest, output, transition):
    audio_only = all(job["action"] == "audio_only" for job in plan["jobs"])
    policy = COMPOSITION_POLICY
    signature = hashlib.sha256(json.dumps({"plan": plan, "manifest": manifest}, sort_keys=True).encode()).hexdigest()
    if output.exists():
        saved = json.loads(output.with_suffix(".json").read_text())
        expected = {"composition_policy": policy, "composition_signature": signature,
                    "audio_only": audio_only, "source_video": plan["source_video"],
                    "audio": plan["target_audio_path"], "fps": plan["fps"], "frames": plan["frames"],
                    "transition_seconds": transition}
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing composition belongs to another request")
        return saved
    if transition < 0:
        raise ValueError("Transition seconds cannot be negative")
    indexed = index_jobs(plan, manifest)
    capture = cv2.VideoCapture(plan["source_video"])
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != plan["frames"] or abs(capture.get(cv2.CAP_PROP_FPS) - plan["fps"]) > 1e-3:
        capture.release()
        raise ValueError("Working video no longer matches the plan")
    fps, count = plan["fps"], plan["frames"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.mp4")
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-i", plan["target_audio_path"],
               "-map", "0:v:0", "-map", "1:a:0", "-frames:v", str(count), "-t", str(count / fps),
               "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-threads", "8", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(temporary)]
    written = 0
    overlap_frames = 0
    with output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            while True:
                valid, source = capture.read()
                if not valid:
                    break
                if written == count:
                    raise ValueError("Working video contains more frames than the plan")
                patches = []
                boxes = []
                for job, path, box, face in indexed[written]:
                    weight = temporal_weight(job, written, fps, transition)
                    if weight == 0:
                        continue
                    pasted = cv2.imread(str(path))
                    if pasted is None or pasted.shape != source.shape:
                        raise ValueError(f"Missing/invalid native output: {path}")
                    x1, y1, x2, y2 = box
                    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
                        raise ValueError(f"ROI outside frame: {path}")
                    if face[2] <= face[0] or face[3] <= face[1]:
                        raise ValueError(f"Empty face box: {path}")
                    patches.append((pasted, box, face, weight))
                    boxes.append(box)
                overlap_frames += any(max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])
                                      for index, a in enumerate(boxes) for b in boxes[index + 1:])
                frame = blend_patches(source, patches)
                process.stdin.write(frame.tobytes())
                written += 1
        finally:
            capture.release()
            process.stdin.close()
            returncode = process.wait()
    if returncode or written != count:
        raise RuntimeError(f"Video encoding failed: returncode={returncode}, frames={written}/{count}")
    result = {"video": str(output), "source_video": plan["source_video"], "audio": plan["target_audio_path"],
              "fps": fps, "frames": count, "duration": count / fps, "transition_seconds": transition, "audio_only": audio_only,
              "composition_policy": policy, "composition_signature": signature,
              "overlap_frames": overlap_frames}
    marker = output.with_suffix(".tmp")
    marker.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    marker.replace(output.with_suffix(".json"))
    temporary.replace(output)
    return result


def comparison(source, final, output, fps, count):
    output = Path(output)
    if output.exists():
        return
    temporary = output.with_name(output.stem + ".partial.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", source, "-i", str(final),
                    "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-map", "1:a:0",
                    "-frames:v", str(count), "-t", str(count / fps), "-c:v", "libx264", "-crf", "20",
                    "-preset", "fast", "-threads", "8", "-pix_fmt", "yuv420p", "-c:a", "copy", str(temporary)], check=True)
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--comparison")
    parser.add_argument("--transition-seconds", type=float, default=0.3)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    result = compose(plan, manifest, Path(args.output).resolve(), args.transition_seconds)
    if args.comparison:
        comparison(plan["source_video"], Path(args.output), Path(args.comparison), plan["fps"], plan["frames"])
    marker = Path(args.output).parent / "compose.tmp"
    marker.write_text(json.dumps({"status": "complete", **result, "comparison": args.comparison}, ensure_ascii=False, indent=2) + "\n")
    marker.replace(marker.with_suffix(".json"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
