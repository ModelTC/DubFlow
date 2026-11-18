"""Face detection, identity encoding, and active-speaker model adapters."""

from pathlib import Path

import numpy as np
from runtime.models import get, is_enabled

from stages.speakers.utils import align_landmarks, box_overlaps, unit_vector


def load_s3fd(checkpoint, device):
    from stages.speakers.models.s3fd.model import S3FDFaceDetector

    checkpoint = Path(checkpoint).resolve()
    return get(("s3fd", str(checkpoint), device), lambda: S3FDFaceDetector(checkpoint, device))

def create_grounding_detector(model_path, device):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path, local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    return model, processor


def load_grounding_detector(model_path, device):
    return get(("grounding_dino", str(model_path), device),
               lambda: create_grounding_detector(model_path, device))


MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MODEL_LICENSE = "https://github.com/deepinsight/insightface/tree/master/model_zoo"


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
            order = order[1:][box_overlaps(boxes[current], boxes[order[1:]]) <= 0.40]
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
        overlaps = box_overlaps(box - np.tile(lower, 2), boxes)
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
        aligned = cv2.warpAffine(crop, align_landmarks(points), (112, 112), borderValue=0)
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


def create_loconet(checkpoint, device):
    import torch
    from stages.speakers.models.loconet.model import LoCoNetEncoder

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("model.module."): value for key, value in state.items()}
    model = LoCoNetEncoder()
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()
                           if key.startswith("model.")}, strict=True)
    classifier = torch.nn.Linear(256, 2)
    classifier.load_state_dict({key.removeprefix("lossAV.FC."): value for key, value in state.items()
                                if key.startswith("lossAV.FC.")}, strict=True)
    return model.to(device).eval(), classifier.to(device).eval()


def load_loconet(checkpoint, device):
    checkpoint = Path(checkpoint).resolve()
    return get(("unitalk_loconet", str(checkpoint), device), lambda: create_loconet(checkpoint, device))

