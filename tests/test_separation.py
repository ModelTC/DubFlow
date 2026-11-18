"""Regression coverage for the stage-owned SAM audio processor and imports."""

import subprocess
import sys
import unittest

import torch

from stages.separation.models.sam_audio.processor import SAMAudioProcessor


class SeparationTests(unittest.TestCase):
    def test_model_import_does_not_need_vendor_editable_packages(self):
        code = """
import importlib.abc
import sys
class BlockVendorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sam_audio', 'dacvae', 'core'}:
            raise ImportError('Legacy vendor import: ' + fullname)
sys.meta_path.insert(0, BlockVendorImports())
from stages.separation.models.sam_audio import SAMAudio, SAMAudioProcessor
"""
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_anchor_indices_and_padding_keep_audio_timeline(self):
        processor = SAMAudioProcessor(1920, 48000)
        first = torch.linspace(-.5, .5, 96000).repeat(2, 1)
        second = torch.ones(1, 48001)
        batch = processor(audios=[first, second], descriptions=['speech', 'speech'],
                          anchors=[[('+', .5, 1.5)], []]).to('cpu')
        self.assertEqual(batch.sizes.tolist(), [50., 26.])
        self.assertEqual(batch.wav_sizes.tolist(), [96000, 48001])
        self.assertTrue(torch.equal(batch.audios[0, 0], first[0]))
        self.assertEqual(batch.anchor_ids.tolist(), [[0, 3, 1], [0, 3, 3]])
        self.assertTrue(torch.all(batch.anchor_alignment[0, 13:38] == 2))
        self.assertTrue(torch.all(batch.anchor_alignment[1, :26] == 0))
        self.assertTrue(torch.all(batch.anchor_alignment[1, 26:] == 1))
        self.assertEqual(batch.audio_pad_mask.sum(dim=1).tolist(), [50, 26])


if __name__ == '__main__':
    unittest.main()
