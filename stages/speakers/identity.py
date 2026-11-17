"""用多帧 SCRFD/ArcFace 外观特征关联跨镜头人脸，不推断真实姓名。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from runtime.config import ROOT
from runtime.models import get, is_enabled

MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MODEL_LICENSE = "https://github.com/deepinsight/insightface/tree/master/model_zoo"
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


def aggregate_embeddings(embeddings, weights=None):
    """Medoid-trim a track before averaging; a bad crop cannot dominate it."""
    if len(embeddings) == 0:
        return None, {"samples": 0, "inliers": 0, "match_eligible": False}
    vectors = np.stack([unit_vector(value) for value in embeddings])
    weights = np.ones(len(vectors)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(vectors),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Sample weights must be finite, positive, and match embeddings")
    similarities = vectors @ vectors.T
    medoid = int(np.argmax(np.median(similarities, axis=1)))
    inliers = similarities[medoid] >= 0.40
    centroid = unit_vector(np.average(vectors[inliers], axis=0, weights=weights[inliers]))
    consistency = float(np.quantile(vectors[inliers] @ centroid, 0.25))
    count = int(inliers.sum())
    quality = {
        "samples": len(vectors), "inliers": count,
        "inlier_fraction": float(inliers.mean()), "consistency": consistency,
        "match_eligible": bool(count >= 2 and inliers.mean() >= 0.60 and consistency >= 0.65),
        "inlier_indices": np.flatnonzero(inliers).tolist(),
    }
    return centroid, quality


def cluster_tracks(tracks, threshold=0.50, ambiguity_margin=0.05):
    """Complete-link 避免相似度链式误合并；同帧出现的人脸禁止合并。"""
    if not -1 <= threshold <= 1 or not 0 <= ambiguity_margin <= 2:
        raise ValueError("Invalid cosine threshold or ambiguity margin")
    ids = [track["id"] for track in tracks]
    if len(ids) != len(set(ids)):
        raise ValueError("Track IDs must be unique")
    frames = [set(track["frames"]) for track in tracks]
    vectors = [None if track["embedding"] is None else unit_vector(track["embedding"]) for track in tracks]
    eligible = [vector is not None and track["quality"]["match_eligible"]
                for track, vector in zip(tracks, vectors)]
    scores, simultaneous = {}, {}
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            simultaneous[i, j] = bool(frames[i] & frames[j])
            scores[i, j] = None if vectors[i] is None or vectors[j] is None else float(
                np.clip(vectors[i] @ vectors[j], -1, 1)
            )

    def pair(i, j):
        return (min(i, j), max(i, j))

    def plausible(i, j):
        key = pair(i, j)
        return (eligible[i] and eligible[j] and not simultaneous[key]
                and scores[key] is not None and scores[key] >= threshold)

    ambiguous = set()
    for i in range(len(tracks)):
        candidates = [j for j in range(len(tracks)) if i != j and plausible(i, j)]
        for offset, j in enumerate(candidates):
            for k in candidates[offset + 1:]:
                incompatible = simultaneous[pair(j, k)] or scores[pair(j, k)] < threshold
                if incompatible and abs(scores[pair(i, j)] - scores[pair(i, k)]) < ambiguity_margin:
                    ambiguous.update((pair(i, j), pair(i, k)))

    groups = [{i} for i in range(len(tracks))]
    edges = sorted((key for key in scores if plausible(*key) and key not in ambiguous),
                   key=lambda key: (-scores[key], ids[key[0]], ids[key[1]]))
    for i, j in edges:
        left = next(group for group in groups if i in group)
        right = next(group for group in groups if j in group)
        if left is right:
            continue
        if all(plausible(a, b) and pair(a, b) not in ambiguous for a in left for b in right):
            left.update(right)
            groups.remove(right)
    groups.sort(key=lambda group: min((min(frames[i], default=10**12), ids[i]) for i in group))
    assignment = {i: f"person_{number:04d}" for number, group in enumerate(groups, 1) for i in group}
    evidence = []
    for (i, j), score in scores.items():
        if simultaneous[i, j]:
            reason = "co_visible_tracks"
        elif not eligible[i] or not eligible[j]:
            reason = "insufficient_identity_quality"
        elif score < threshold:
            reason = "below_cosine_threshold"
        elif (i, j) in ambiguous:
            reason = "ambiguous_similar_identity"
        elif assignment[i] == assignment[j]:
            reason = "accepted_complete_link"
        else:
            reason = "cluster_complete_link_veto"
        evidence.append({"track_a": ids[i], "track_b": ids[j], "cosine": score,
                         "merged": assignment[i] == assignment[j], "reason": reason})
    return {ids[i]: assignment[i] for i in range(len(tracks))}, evidence


def normalize_tracks(vision):
    if abs(vision["fps"] - 25) > 0.01:
        raise ValueError("Identity input must use the shared 25 FPS timeline")
    result = []
    for track in vision["tracks"]:
        frames = np.asarray(track["frames"])
        if not np.isfinite(frames).all() or (frames != np.floor(frames)).any():
            raise ValueError(f"Frame indices must be finite integers: {track['id']}")
        frames = frames.astype(np.int64)
        boxes = np.asarray(track["bboxes"], dtype=np.float64).reshape(-1, 4)
        if len(set(frames.tolist())) != len(frames):
            raise ValueError(f"Duplicate frame indices: {track['id']}")
        if boxes.shape != (len(frames), 4) or not np.isfinite(boxes).all():
            raise ValueError(f"Invalid xyxy boxes: {track['id']}")
        if ((boxes[:, 2:] - boxes[:, :2]) <= 0).any() or (frames < 0).any() or (frames >= vision["frames"]).any():
            raise ValueError(f"Out-of-range frames or boxes: {track['id']}")
        order = np.argsort(frames)
        item = {"id": track["id"], "shot_id": track["shot_id"],
                "frames": frames[order].tolist(), "bboxes": boxes[order].tolist()}
        for key in ("asd_scores", "confidence"):
            values = track[key]
            if len(values) != len(frames):
                raise ValueError(f"Frame/score alignment differs: {track['id']}")
            item[key] = [values[index] for index in order]
        result.append(item)
    if len({track["id"] for track in result}) != len(result):
        raise ValueError("Track IDs must be unique")
    return result


def candidate_indices(track, max_candidates=48):
    """兼顾轨迹内时间分布与画质，ASD 只作轻微排序因素。"""
    count = len(track["frames"])
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    selected = np.unique(np.linspace(0, count - 1, min(count, max_candidates), dtype=int)) if count else []
    boxes = np.asarray(track["bboxes"])

    def quality(index):
        area = np.prod(boxes[index, 2:] - boxes[index, :2])
        confidence = np.clip(track["confidence"][index], 0, 1)
        asd_score = track["asd_scores"][index]
        speech_tiebreak = 1.0 if asd_score is None else 1 + 0.05 * np.tanh(asd_score)
        return area * confidence * speech_tiebreak

    remaining, ordered = list(map(int, selected)), []
    span = max(1, max(track["frames"], default=0) - min(track["frames"], default=0))
    while remaining:
        def priority(index):
            distance = min((abs(track["frames"][index] - track["frames"][other]) / span
                            for other in ordered), default=1)
            return np.sqrt(quality(index)) * (0.2 + 0.8 * distance), -track["frames"][index]
        best = max(remaining, key=priority)
        ordered.append(best)
        remaining.remove(best)
    return ordered


def _iou(box, boxes):
    boxes = np.asarray(boxes)
    lower = np.maximum(box[:2], boxes[:, :2])
    upper = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(upper - lower, 0), axis=1)
    area = np.prod(np.maximum(box[2:] - box[:2], 0))
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-8)


def _alignment(landmarks):
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


class FaceEncoder:
    """SCRFD-10GF 检测、五点对齐和 ArcFace R50。"""

    def __init__(self, model_dir, threads=4, device="cpu"):
        import cv2
        import onnxruntime as ort

        self.cv2 = cv2
        self.paths = {"detector": Path(model_dir) / "det_10g.onnx",
                      "recognizer": Path(model_dir) / "w600k_r50.onnx"}
        for path in self.paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Missing official buffalo_l model: {path}; source: {MODEL_URL}")
        self.device = "cuda:0" if device == "cuda" else device
        if self.device.startswith("cuda"):
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("Resident face identity requires the ONNX Runtime CUDA execution provider")
            ort.preload_dlls()
            providers = [("CUDAExecutionProvider", {"device_id": int(self.device.partition(":")[2]), "use_tf32": 0})]
        else:
            providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.detector = ort.InferenceSession(str(self.paths["detector"]), options,
                                            providers=providers)
        self.recognizer = ort.InferenceSession(str(self.paths["recognizer"]), options,
                                              providers=providers)
        if self.device.startswith("cuda"):
            for session in (self.detector, self.recognizer):
                if session.get_providers()[0] != "CUDAExecutionProvider":
                    raise RuntimeError("ONNX Runtime fell back to CPU for resident face identity")
                session.disable_fallback()
        if len(self.detector.get_outputs()) != 9:
            raise ValueError("Expected official SCRFD three-scale bbox + five-landmark model")
        self.anchor_cache = {}

    def detect(self, image):
        cv2 = self.cv2
        scale = 640 / max(image.shape[:2])
        height, width = np.maximum(np.rint(np.array(image.shape[:2]) * scale).astype(int), 1)
        canvas = np.zeros((640, 640, 3), dtype=np.uint8)
        canvas[:height, :width] = cv2.resize(image, (width, height))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128.0, (640, 640), (127.5,) * 3, swapRB=True)
        outputs = self.detector.run(None, {self.detector.get_inputs()[0].name: blob})
        boxes, landmarks, confidences = [], [], []
        for index, stride in enumerate((8, 16, 32)):
            scores = outputs[index].reshape(-1)
            distance = outputs[index + 3].reshape(-1, 4) * stride
            points = outputs[index + 6].reshape(-1, 5, 2) * stride
            if stride not in self.anchor_cache:
                centers = np.stack(np.mgrid[:640 // stride, :640 // stride][::-1], axis=-1)
                self.anchor_cache[stride] = np.repeat(centers.reshape(-1, 2), 2, axis=0) * stride
            centers = self.anchor_cache[stride]
            selected = scores >= 0.60
            boxes.append(np.column_stack((centers - distance[:, :2], centers + distance[:, 2:]))[selected] / scale)
            landmarks.append((centers[:, None] + points)[selected] / scale)
            confidences.append(scores[selected])
        boxes, landmarks, confidences = np.concatenate(boxes), np.concatenate(landmarks), np.concatenate(confidences)
        order = np.argsort(-confidences)
        keep = []
        while len(order):
            current = order[0]
            keep.append(current)
            order = order[1:][_iou(boxes[current], boxes[order[1:]]) <= 0.40]
        return boxes[keep], landmarks[keep], confidences[keep]

    def embed(self, image, bbox):
        cv2 = self.cv2
        box = np.asarray(bbox, dtype=np.float64)
        size = box[2:] - box[:2]
        if min(size) < 40:
            return None, {"rejected": "face_too_small"}
        padding = size * 0.35
        lower = np.maximum(np.floor(box[:2] - padding).astype(int), 0)
        upper = np.minimum(np.ceil(box[2:] + padding).astype(int), image.shape[1::-1])
        if (upper <= lower).any():
            return None, {"rejected": "bbox_outside_frame"}
        crop = image[lower[1]:upper[1], lower[0]:upper[0]]
        boxes, landmarks, scores = self.detect(crop)
        if not len(boxes):
            return None, {"rejected": "no_aligned_face"}
        overlaps = _iou(box - np.tile(lower, 2), boxes)
        best = int(np.argmax(overlaps))
        if overlaps[best] < 0.35:
            return None, {"rejected": "detector_track_mismatch", "detector_iou": float(overlaps[best])}
        if len(overlaps) > 1 and np.sort(overlaps)[-1] - np.sort(overlaps)[-2] < 0.15:
            return None, {"rejected": "ambiguous_face_crop"}
        points = landmarks[best]
        eye_distance = float(np.linalg.norm(points[1] - points[0]))
        if eye_distance < 12:
            return None, {"rejected": "eyes_too_small"}
        yaw = abs(points[2, 0] - points[:2, 0].mean()) / eye_distance
        if yaw > 0.65:
            return None, {"rejected": "strong_profile", "yaw_proxy": float(yaw)}
        aligned = cv2.warpAffine(crop, _alignment(points), (112, 112), borderValue=0)
        sharpness = float(cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if sharpness < 15:
            return None, {"rejected": "blurred_face", "sharpness": sharpness}
        blob = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)
        embedding = unit_vector(self.recognizer.run(None, {self.recognizer.get_inputs()[0].name: blob})[0])
        quality = {"detector_score": float(scores[best]), "detector_iou": float(overlaps[best]),
                   "sharpness": sharpness, "yaw_proxy": float(yaw),
                   "weight": float(scores[best] * (1 - yaw) * min(2, np.log1p(sharpness) / 4))}
        return embedding, quality

    def metadata(self):
        return {"family": "InsightFace buffalo_l", "detector": str(self.paths["detector"]),
                "recognizer": str(self.paths["recognizer"]), "device": self.device,
                "license": "non-commercial research only", "license_url": MODEL_LICENSE}


def load_face_encoder(model_dir, threads=4, device="cpu"):
    model_dir = str(Path(model_dir).resolve())
    device = "cuda:0" if is_enabled() else device
    return get(("face_encoder", model_dir, device, threads),
               lambda: FaceEncoder(model_dir, threads=threads, device=device))


def build_identities(video, vision, model_dir, *, threshold=0.50, ambiguity_margin=0.05,
                     max_samples=12, max_candidates=48, threads=4, device="cpu"):
    """Return identity.json data from a 25-FPS video and detected face tracks."""
    import cv2

    if max_samples < 2 or max_candidates < max_samples:
        raise ValueError("Need at least two samples and max_candidates >= max_samples")
    tracks = normalize_tracks(vision)
    encoder = load_face_encoder(model_dir, threads=threads, device=device)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open working video: {video}")
    if abs(capture.get(cv2.CAP_PROP_FPS) - 25) > 0.01:
        capture.release()
        raise ValueError("Working video must be 25 FPS, matching vision frame indices")
    records = []
    try:
        for track in tracks:
            embeddings, weights, samples, rejected = [], [], [], []
            for index in candidate_indices(track, max_candidates):
                frame_index = track["frames"][index]
                # Adjacent crops are correlated; require a little temporal diversity.
                if any(abs(frame_index - sample["frame"]) < 3 for sample in samples):
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode requested frame {frame_index}")
                embedding, quality = encoder.embed(image, track["bboxes"][index])
                evidence = {"frame": frame_index, "bbox": track["bboxes"][index], **quality}
                if embedding is None:
                    rejected.append(evidence)
                    continue
                embeddings.append(embedding)
                weights.append(quality["weight"])
                samples.append(evidence)
                if len(samples) >= max_samples:
                    break
            embedding, quality = aggregate_embeddings(embeddings, weights)
            records.append({**track, "embedding": None if embedding is None else embedding.tolist(),
                            "quality": quality, "samples": samples, "rejected_samples": rejected,
                            "representative_frame": max(samples, key=lambda sample: sample["weight"])["frame"] if samples else None})
            print(f"identity: track={track['id']} accepted={len(samples)} eligible={quality['match_eligible']}", flush=True)
    finally:
        capture.release()
    assignment, similarities = cluster_tracks(records, threshold, ambiguity_margin)
    for record in records:
        record["person_id"] = assignment[record["id"]]
        record["status"] = "appearance_cluster" if record["quality"]["match_eligible"] else "unresolved_low_quality"
    return {"schema_version": 1, "video": str(Path(video).resolve()), "fps": 25,
            "track_to_person": assignment, "tracks": records, "similarities": similarities,
            "models": encoder.metadata(),
            "parameters": {"cosine_threshold": threshold, "ambiguity_margin": ambiguity_margin,
                           "max_samples": max_samples, "max_candidates": max_candidates},
            "identity_kind": "anonymous_appearance_cluster"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--vision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default=str(ROOT / "models/insightface/buffalo_l"))
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--ambiguity-margin", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    with Path(args.vision).open(encoding="utf-8") as handle:
        vision = json.load(handle)
    result = build_identities(args.video, vision, args.model_dir, threshold=args.threshold,
                              ambiguity_margin=args.ambiguity_margin, max_samples=args.max_samples,
                              max_candidates=args.max_candidates, threads=args.threads, device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(output)
    reliable = {track["person_id"] for track in result["tracks"] if track["quality"]["match_eligible"]}
    unresolved = sum(not track["quality"]["match_eligible"] for track in result["tracks"])
    print(f"Saved {output}: {len(reliable)} appearance clusters, {unresolved} unresolved tracks", flush=True)


if __name__ == "__main__":
    main()

