"""Timeline identity remains valid through relative and symlink stage inputs."""

import json
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

from stages.postprocess import retime


class RetimeTests(unittest.TestCase):
    def test_relative_symlink_plan_uses_the_same_timeline(self):
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            Path("source.mp4").write_bytes(b"source")
            Path("plan.json").write_text(json.dumps({"fps": 25, "frames": 10, "shots": []}))
            Path("alias.json").symlink_to("plan.json")
            Path("dialogue.json").write_text(json.dumps({
                "source_plan": str(Path("plan.json").resolve()), "utterances": [],
            }))
            with patch.object(retime, "media_info", return_value=(10, .4, .4)), \
                    patch.object(retime, "render", return_value=10) as render:
                retime.run_stage(source="source.mp4", plan="alias.json", dialogue="dialogue.json",
                                 output="final.mp4", manifest="manifest.json")
            self.assertEqual(render.call_args.args[-1], [])
            self.assertEqual(json.loads(Path("manifest.json").read_text())["frames"], 10)
