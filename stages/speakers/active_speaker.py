"""UniTalk-LoCoNet: score a tracked face against original audio and nearby faces."""

import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from runtime.models import get
from scipy.ndimage import uniform_filter1d
from third_party.loconet.model.loconet_encoder import locoencoder
from third_party.loconet.torchvggish import mel_features, vggish_params

ASD_POLICY = "unitalk_loconet_tight_context3_v1"
MAX_WINDOW_FRAMES = 500
CONTEXT_FRAMES = 25


def _create_loconet(checkpoint, device):
    cfg = SimpleNamespace(MODEL=SimpleNamespace(NUM_SPEAKERS=3, AV_layers=3, ADJUST_ATTENTION=0))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("model.module."): value for key, value in state.items()}
    model = locoencoder(cfg)
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()
                           if key.startswith("model.")}, strict=True)
    classifier = torch.nn.Linear(256, 2)
    classifier.load_state_dict({key.removeprefix("lossAV.FC."): value for key, value in state.items()
                                if key.startswith("lossAV.FC.")}, strict=True)
    return model.to(device).eval(), classifier.to(device).eval()


def load_loconet(checkpoint, device):
    checkpoint = Path(checkpoint).resolve()
    return get(("unitalk_loconet", str(checkpoint), device), lambda: _create_loconet(checkpoint, device))


def context_tracks(target, tracks, center):
    """Official validation layout: target first, then repeat context up to 3 faces."""
    candidates = [track for track in sorted(tracks, key=lambda item: item["id"])
                  if track["id"] != target["id"] and track["shot_id"] == target["shot_id"]
                  and center in track["frames"]]
    rng = random.Random(42)
    rng.shuffle(candidates)
    selected = [target, *candidates[:2]]
    while len(selected) < 3:
        selected.append(rng.choice(selected[1:] or selected))
    return selected


def context_pixels(selected, frames, frame_dir):
    """Resize entire boxes to 112px; absent context faces are zero, not stale crops."""
    boxes, visible = {}, {}
    for track in selected:
        key = track["id"]
        raw = np.asarray(track["bboxes"])
        boxes[key] = np.stack([np.interp(frames, track["frames"], raw[:, axis]) for axis in range(4)], axis=1)
        visible[key] = np.ones(len(frames), dtype=bool) if key == selected[0]["id"] else np.isin(frames, track["frames"])
    result = np.zeros((3, len(frames), 112, 112), dtype=np.uint8)
    for index, frame in enumerate(frames):
        image = cv2.imread(str(frame_dir / f"{frame:07d}.jpg"))
        if image is None:
            raise OSError(f"Missing analysis frame {frame}")
        crops = {}
        for slot, track in enumerate(selected):
            key = track["id"]
            if not visible[key][index]:
                continue
            if key not in crops:
                x1, y1, x2, y2 = np.rint(boxes[key][index]).astype(int)
                crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
                if not crop.size:
                    raise ValueError(f"Empty face crop: {key} at frame {frame}")
                crops[key] = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (112, 112))
            result[slot, index] = crops[key]
    return result


def audio_features(waveform, frames):
    """Match LoCoNet's PCM16 log-mel preprocessing, with 4 audio frames per frame."""
    if waveform.ndim != 1 or waveform.dtype != np.int16:
        raise ValueError("UniTalk-LoCoNet requires mono 16-kHz PCM16 original audio")
    start, needed = int(frames[0]) * 640, len(frames) * 640 + 240
    audio = waveform[start:start + needed]
    audio = np.pad(audio, (0, max(0, needed - len(audio))))
    features = mel_features.log_mel_spectrogram(
        audio, audio_sample_rate=16000, log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS, lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ)
    count = len(frames) * 4
    if len(features) < count:
        features = np.pad(features, ((0, count - len(features)), (0, 0)), mode="wrap")
    return features[:count]


def inference_windows(length, max_frames=MAX_WINDOW_FRAMES, context=CONTEXT_FRAMES):
    """Keep short tracks intact; bound long-track attention without dropping frames."""
    core = length if length <= max_frames else max_frames - 2 * context
    for first in range(0, length, core):
        last = min(length, first + core)
        yield first, last, max(0, first-context), min(length, last+context)


@torch.inference_mode()
def score_track(track, tracks, model, classifier, waveform, frame_dir, device):
    first_frame = track["frames"][0]
    length = track["frames"][-1] - first_frame + 1
    logits = np.empty((length, 2), dtype=np.float32)
    windows = []
    for first, last, left, right in inference_windows(length):
        frames = np.arange(first_frame + left, first_frame + right)
        selected = context_tracks(track, tracks, int(frames[len(frames)//2]))
        visual = context_pixels(selected, frames, frame_dir)
        audio = audio_features(waveform, frames)
        a = model.forward_audio_frontend(torch.as_tensor(audio, dtype=torch.float32, device=device)[None, None])
        v = model.forward_visual_frontend(torch.as_tensor(visual, dtype=torch.float32, device=device))
        a, v = model.forward_cross_attention(a.repeat(3, 1, 1), v)
        output = model.forward_audio_visual_backend(a, v, 1, 3)
        scores = classifier(output).reshape(3, len(frames), 2)[0].cpu().numpy()
        logits[first:last] = scores[first-left:last-left]
        windows.append({"first_frame": int(frames[0]), "end_frame": int(frames[-1])+1,
                        "context_tracks": [row["id"] for row in selected]})
    if not np.isfinite(logits).all():
        raise ValueError(f"Non-finite UniTalk-LoCoNet output: {track['id']}")
    smooth = uniform_filter1d(logits, size=5, axis=0, mode="nearest")
    positions = np.asarray(track["frames"]) - first_frame
    track.update(asd_scores=smooth[positions, 1].tolist(), asd_status="scored",
                 asd_probability=torch.from_numpy(smooth[positions]).softmax(-1)[:, 1].tolist(),
                 asd_windows=windows)
