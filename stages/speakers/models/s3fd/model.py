"""Image preprocessing and checkpoint loading for S3FD face detection."""

import cv2
import numpy as np

from stages.speakers.utils import numpy_nms
from .network import S3FDNet


class S3FDFaceDetector:
    """Original S3FD float32 forward with reusable, exact-shape prior boxes.

    The backbone and head operations follow TalkNet-ASD's S3FDNet.forward.
    Priors are immutable and depend only on feature-map and input dimensions.
    """

    def __init__(self, checkpoint, device):
        import torch

        self.device = device
        self.net = S3FDNet().to(device)
        self.net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        self.net.eval()
        self.mean = np.array([104., 117., 123.], dtype=np.float32)[:, None, None]

    def detect_faces(self, image, conf_th=.8, scales=(1.,)):
        import torch

        height, width = image.shape[:2]
        boxes = []
        with torch.no_grad():
            for scale in scales:
                pixels = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                pixels = np.swapaxes(np.swapaxes(pixels, 1, 2), 1, 0)
                pixels = pixels[[2, 1, 0]].astype(np.float32)
                pixels -= self.mean
                pixels = pixels[[2, 1, 0]]
                predictions = self.net(torch.from_numpy(pixels).unsqueeze(0).to(self.device))
                dimensions = torch.tensor([width, height, width, height])
                for category in predictions[0]:
                    selected = category[category[:, 0] > conf_th]
                    points = (selected[:, 1:] * dimensions).cpu().numpy()
                    boxes.append(np.column_stack((points, selected[:, 0].cpu().numpy())).astype(np.float64))
        boxes = np.concatenate(boxes, axis=0)
        return boxes[numpy_nms(boxes, .1)]
