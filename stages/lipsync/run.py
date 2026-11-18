"""Send lossless face crops to LightX2V; feather and paste returned crops locally."""

import hashlib
import json
import wave
from pathlib import Path

from runtime.utils import write_json

from stages.lipsync.model import TBDubModel
from stages.lipsync.utils import model_boxes, load_source_frames

from stages.lipsync.geometry import (
    blend_crop_video_with_ref,
    crop_and_resize_frames,
    paste_video_back,
)


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


def render_job(job, frames, client, output):
    first, last = job["start_frame"], job["end_frame"]
    boxes = model_boxes(job, *frames[0].size)
    with wave.open(job["drive_audio_path"]) as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getnframes()) != (16000, 1, (last - first) * 640):
            raise ValueError(f"Driving audio does not match model frames: {job['id']}")
    directory = output / job["id"]
    directory.mkdir(parents=True, exist_ok=True)
    crops = crop_and_resize_frames(frames, boxes)
    generated, task_id = client.generate(crops, job["drive_audio_path"], directory)
    if any(frame.size != (512, 512) for frame in generated):
        raise ValueError("TBDub returned a different crop size")
    # Color correction is done once in LightX2V; these are the native spatial edge masks.
    generated = blend_crop_video_with_ref(generated, crops, 1)
    pasted = paste_video_back(generated, frames, boxes)
    for index, frame in enumerate(pasted):
        frame.save(directory / f"{index:06d}.png", compress_level=1)
    records = [{**record, "bbox": box} for record, box in zip(job["roi_records"], boxes)]
    return {**job, "roi_records": records, "status": "complete", "frames_rendered": len(pasted),
            "task_id": task_id, "frame_directory": str(directory)}


def run_stage(*, plan, output_dir, server_url):
    plan_bytes = Path(plan).read_bytes()
    plan = json.loads(plan_bytes)
    if plan["fps"] != 25:
        raise ValueError("TBDub requires a shared 25 FPS timeline")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "render_manifest.json"
    request = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
               "server_url": server_url, "renderer": "lightx2v_tbdub_v1"}
    manifest = restore_manifest(plan, manifest_path, request)
    finished = {job["id"]: job for job in manifest["jobs"] if completed_result(job)}
    manifest.update(status="running", jobs=list(finished.values()))
    manifest.pop("error", None)
    write_json(manifest_path, manifest)
    client = TBDubModel(server_url)
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
            write_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(manifest_path, manifest)
        raise
    finally:
        client.close()
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
