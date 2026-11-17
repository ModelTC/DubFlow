"""Send lossless face crops to LightX2V; feather and paste returned crops locally."""

import argparse
import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path

from runtime.clients import LipSync

from stages.lipsync.geometry import (
    blend_crop_video_with_ref,
    crop_and_resize_frames,
    paste_video_back,
)


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def audio_frames(path, fps=25):
    # 避免 int(seconds * fps) 的浮点截断导致 29 帧被算成 28 帧。
    with wave.open(str(path)) as audio:
        return audio.getnframes() * fps // audio.getframerate()


def restore_manifest(plan, path, request):
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["request"] != request:
            raise ValueError("Render checkpoint belongs to another plan or model path")
        return manifest
    return {"status": "running", "request": request, "source_video": plan["source_video"],
            "target_audio_path": plan["target_audio_path"], "fps": 25, "frames": plan["frames"], "jobs": []}


def completed_result(result):
    if result["status"] == "audio_only":
        return True
    directory = Path(result["frame_directory"])
    return (result["status"] == "complete" and
            result["frames_rendered"] == result["end_frame"] - result["start_frame"] and
            all((directory / f"{index:06d}.png").is_file() for index in range(result["frames_rendered"])))


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


def render_job(job, frames, client, output):
    first, last = job["start_frame"], job["end_frame"]
    boxes = model_boxes(job, *frames[0].size)
    with wave.open(job["drive_audio_path"]) as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getnframes()) != (16000, 1, (last - first) * 640):
            raise ValueError(f"Driving audio does not match model frames: {job['id']}")
    directory = output / job["id"]
    directory.mkdir(parents=True, exist_ok=True)
    crops = crop_and_resize_frames(frames, boxes)
    source_path, generated_path = directory / "input.mp4", directory / "generated.mp4"
    receipt = directory / "task.json"
    if receipt.exists():
        task = json.loads(receipt.read_text())
    else:
        write_crop_video(crops, source_path)
        payload = {"task": "v2v", "seed": 42, "video_path": str(source_path),
                   "audio_path": job["drive_audio_path"], "num_frames": last - first,
                   "size": [512, 512], "save_result_path": str(generated_path)}
        task = client.submit(payload)
        save_json(receipt, task)
    state = client.wait(task["task_id"])
    save_json(directory / "status.json", state)
    generated = load_source_frames(generated_path, last - first)
    if any(frame.size != (512, 512) for frame in generated):
        raise ValueError("TBDub returned a different crop size")
    # Color correction is done once in LightX2V; these are the native spatial edge masks.
    generated = blend_crop_video_with_ref(generated, crops, 1)
    pasted = paste_video_back(generated, frames, boxes)
    for index, frame in enumerate(pasted):
        frame.save(directory / f"{index:06d}.png", compress_level=1)
    records = [{**record, "bbox": box} for record, box in zip(job["roi_records"], boxes)]
    return {**job, "roi_records": records, "status": "complete", "frames_rendered": len(pasted),
            "task_id": task["task_id"], "frame_directory": str(directory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server-url", required=True)
    args = parser.parse_args()
    plan_bytes = Path(args.plan).read_bytes()
    plan = json.loads(plan_bytes)
    if plan["fps"] != 25:
        raise ValueError("TBDub requires a shared 25 FPS timeline")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "render_manifest.json"
    request = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
               "server_url": args.server_url, "renderer": "lightx2v_tbdub_v1"}
    manifest = restore_manifest(plan, manifest_path, request)
    finished = {job["id"]: job for job in manifest["jobs"] if completed_result(job)}
    manifest.update(status="running", jobs=list(finished.values()))
    manifest.pop("error", None)
    save_json(manifest_path, manifest)
    client = LipSync(args.server_url)
    try:
        for job in plan["jobs"]:
            if job["id"] in finished:
                result = finished[job["id"]]
                print(f"Retaining rendered {job['id']}", flush=True)
            elif job["action"] == "audio_only":
                result = {**job, "status": "audio_only", "frame_directory": None}
            else:
                print(f"Rendering {job['id']} [{job['start_frame']}, {job['end_frame']})", flush=True)
                frames = load_source_frames(plan["source_video"], plan["frames"],
                                            job["start_frame"], job["end_frame"])
                result = render_job(job, frames, client, output)
            finished[job["id"]] = result
            manifest["jobs"] = [finished[item["id"]] for item in plan["jobs"] if item["id"] in finished]
            save_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        save_json(manifest_path, manifest)
        raise
    finally:
        client.close()
    manifest["status"] = "complete"
    save_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
