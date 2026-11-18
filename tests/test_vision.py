"""CPU regressions for frame persistence and resumable speaker detection."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from stages.speakers import run
from stages.speakers.utils import AnalysisFrames


class VideoCapture:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        self.read_count += 1
        if self.index == len(self.frames):
            return False, None
        image = self.frames[self.index]
        self.index += 1
        return True, image.copy()

    def set(self, property_id, value):
        if property_id != cv2.CAP_PROP_POS_FRAMES:
            return False
        self.index = int(value)
        return True

    def release(self):
        self.released = True


class FaceDetector:
    def __init__(self, frame_dir, start=0):
        self.frame_dir = frame_dir
        self.index = start

    def detect_faces(self, image, conf_th, scales):
        path = self.frame_dir / f"{self.index:07d}.jpg"
        if not path.is_file() or cv2.imread(str(path)) is None:
            raise AssertionError("Detection started before its JPEG was persisted")
        self.index += 1
        score = .6 + float(image[0, 0, 0]) / 1000
        return np.asarray([[0, 0, 8, 8, score]], dtype=np.float64)


class FramePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in range(6)]

    def tearDown(self):
        self.temporary.cleanup()

    def test_prefetch_is_bounded_and_jpeg_bytes_match_serial_writer(self):
        capture = VideoCapture(self.frames)
        directory = self.root / "frames"
        with patch.object(cv2, "VideoCapture", return_value=capture):
            with AnalysisFrames("video.mp4", directory) as frames:
                index, rgb = next(frames)
                frames.pending.result(timeout=5)
                self.assertEqual(index, 0)
                self.assertTrue(np.array_equal(rgb, cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2RGB)))
                self.assertEqual(capture.read_count, 2)
                self.assertEqual(len(list(directory.glob("*.jpg"))), 2)
        self.assertTrue(capture.released)
        for index in range(2):
            original = self.root / f"serial_{index}.jpg"
            cv2.imwrite(str(original), self.frames[index], [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.assertEqual(original.read_bytes(), (directory / f"{index:07d}.jpg").read_bytes())
        self.assertEqual(list(directory.glob("*.tmp.jpg")), [])

    def detect(self, directory, rows=None, checkpoint=None):
        detector = FaceDetector(directory, len(rows or []))
        with patch.object(cv2, "VideoCapture", side_effect=lambda _: VideoCapture(self.frames)):
            with patch.object(run, "load_s3fd", return_value=detector):
                return run.detect_faces("video.mp4", directory, "cpu", 1., .5, 1,
                                        detections=rows, checkpoint=checkpoint, weights="weights")

    def test_write_failure_checkpoints_only_completed_frames_and_resume_matches(self):
        expected = self.detect(self.root / "reference")
        directory = self.root / "resumed"
        saved = []

        def checkpoint(rows):
            self.assertTrue(all((directory / f"{index:07d}.jpg").is_file() for index in range(len(rows))))
            saved[:] = copy.deepcopy(rows)

        original_write = cv2.imwrite

        def fail_write(path, *args, **kwargs):
            if Path(path).name == "0000003.tmp.jpg":
                return False
            return original_write(path, *args, **kwargs)

        with patch.object(cv2, "imwrite", side_effect=fail_write):
            with self.assertRaisesRegex(OSError, "Cannot save analysis frame 3"):
                self.detect(directory, checkpoint=checkpoint)
        self.assertEqual(saved, expected[:3])
        result = self.detect(directory, rows=saved, checkpoint=checkpoint)
        self.assertEqual(result, expected)
        self.assertEqual(saved, expected)
        for index in range(len(self.frames)):
            name = f"{index:07d}.jpg"
            self.assertEqual((directory / name).read_bytes(), (self.root / "reference" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
