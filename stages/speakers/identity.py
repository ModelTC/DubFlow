"""用多帧 SCRFD/ArcFace 外观特征关联跨镜头人脸，不推断真实姓名。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stages.speakers.model import load_face_encoder
from stages.speakers.utils import unit_vector, write_json

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


def run_stage(*, video, vision, output, model_dir,
              threshold=.5, ambiguity_margin=.05, max_samples=12, max_candidates=48,
              threads=4, device="cpu"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    vision_data = json.loads(Path(vision).read_text())
    result = build_identities(video, vision_data, model_dir, threshold=threshold,
                              ambiguity_margin=ambiguity_margin, max_samples=max_samples,
                              max_candidates=max_candidates, threads=threads, device=device)
    write_json(output, result)
    reliable = {track["person_id"] for track in result["tracks"] if track["quality"]["match_eligible"]}
    unresolved = sum(not track["quality"]["match_eligible"] for track in result["tracks"])
    print(f"Saved {output}: {len(reliable)} appearance clusters, {unresolved} unresolved tracks", flush=True)
    return result
