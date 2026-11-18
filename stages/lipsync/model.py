"""MediaPipe landmarks and the LightX2V lip-sync service adapter."""

import json
from pathlib import Path

from runtime.utils import write_json

from runtime.clients import LipSync
from runtime.models import get
from stages.lipsync.utils import load_source_frames, write_crop_video


def load_landmarker(path, options=None):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    path = Path(path).resolve()
    options = options or vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(path), delegate=python.BaseOptions.Delegate.CPU), num_faces=1)
    return get(("mediapipe_landmarker", str(path), "cpu"), lambda: vision.FaceLandmarker.create_from_options(options))


class TBDubModel:
    def __init__(self, server_url):
        self.client = LipSync(server_url)

    def generate(self, frames, audio_path, directory):
        source = directory / "input.mp4"
        generated = directory / "generated.mp4"
        receipt = directory / "task.json"
        if receipt.exists():
            task = json.loads(receipt.read_text())
        else:
            write_crop_video(frames, source)
            task = self.client.submit({"task": "v2v", "seed": 42, "video_path": str(source),
                "audio_path": str(audio_path), "num_frames": len(frames), "size": [512, 512],
                "save_result_path": str(generated)})
            write_json(receipt, task)
        write_json(directory / "status.json", self.client.wait(task["task_id"]))
        return load_source_frames(generated, len(frames)), task["task_id"]

    def close(self):
        self.client.close()
