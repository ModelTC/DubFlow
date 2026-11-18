"""Regression coverage for stage isolation, resident dispatch and job resume."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from runtime.config import normalize_config
from runtime.execution import FunctionCall, call_function
from runtime.job import Job
from runtime.processes import Engine, StagePool, model_roles
from runtime.utils import read_json, write_json


class ConfigTests(unittest.TestCase):
    def test_external_tts_keeps_provider_name_and_skips_local_model_role(self):
        config = normalize_config({"python": sys.executable, "tts_backend": "openai_compatible",
                                   "tts_url": "https://tts.example/v1", "tts_model": "provider-voice"})
        self.assertEqual(config["tts_model"], "provider-voice")
        self.assertFalse(any("tts" in roles for roles in model_roles(config).values()))
        self.assertEqual(config["tts_api_key_env"], "LIGHTDUBBER_TTS_API_KEY")

    def test_interpreter_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            python = Path(temporary) / "python"
            python.symlink_to(sys.executable)
            config = normalize_config({"python": str(python)})
            self.assertEqual(config["video_python"], str(python))


class DispatchTests(unittest.TestCase):
    def test_subprocess_result_and_failure(self):
        result = call_function(sys.executable, "builtins", "print", {"end": "complete"}, capture=True)
        self.assertEqual(result.stdout, "complete")
        failure = call_function(sys.executable, "json", "loads", {"s": "broken"}, capture=True, check=False)
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("JSONDecodeError", failure.stderr)
        with self.assertRaises(subprocess.CalledProcessError):
            call_function(sys.executable, "json", "loads", {"s": "broken"}, capture=True)

    def test_resident_nested_calls_logs_and_failure_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            write_json(config, {"python": sys.executable})
            other_python = root / "other-python"
            other_python.symlink_to(sys.executable)
            fixture = root / "dispatch_fixture.py"
            fixture.write_text(
                "import logging\nfrom runtime.execution import call_function\n"
                "def nested(python, invalid=False):\n"
                "    if invalid:\n"
                "        result = call_function(python, 'json', 'loads', {'s': 'broken'}, capture=True, check=False)\n"
                "        assert result.returncode == 1\n"
                "    else:\n"
                "        result = call_function(python, 'builtins', 'print', {'end': 'nested-ok'}, capture=True)\n"
                "        print(result.stdout)\n"
                "    logging.info('logged nested call')\n")
            environment = {"PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
            pool = StagePool.__new__(StagePool)
            pool.engines = {}
            try:
                with patch.dict(os.environ, environment):
                    for index, python in enumerate((sys.executable, str(other_python))):
                        engine = Engine(python, ["speech"], config, "", root / f"engine{index}")
                        pool.engines[str(Path(python).absolute())] = engine
                        self.assertEqual(engine.receive()["type"], "ready")
                for index, python in enumerate((sys.executable, str(other_python))):
                    pool.current_log = root / f"stage{index}.log"
                    call = FunctionCall("dispatch_fixture", "nested", {"python": python})
                    result = pool.call(sys.executable, call, capture=True)
                    self.assertEqual(result.stdout, "nested-ok\n")
                    self.assertIn("logged nested call", pool.current_log.read_text())
                pool.current_log = root / "failure.log"
                call = FunctionCall("dispatch_fixture", "nested", {"python": str(other_python), "invalid": True})
                self.assertEqual(pool.call(sys.executable, call).returncode, 0)
                result = pool.call(sys.executable, FunctionCall("builtins", "print", {"end": "still-live"}), capture=True)
                self.assertEqual(result.stdout, "still-live")
            finally:
                pool.close()


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root / "source.mp4"
        source.write_bytes(b"source-video")
        self.args = SimpleNamespace(source=source, output=self.root / "output", resume=False,
                                    analysis_plan=None, source_audio_manifest=None, dialogue=None)
        self.config = normalize_config({"python": sys.executable, "gpu": 0})

    def tearDown(self):
        self.temporary.cleanup()

    def test_old_relative_config_resumes_and_changed_request_is_rejected(self):
        legacy = {"python": sys.executable, "s3fd_repo": "third_party/s3fd", "gpu": 0}
        job = Job(self.args, legacy)
        self.args.resume = True
        current = normalize_config(legacy)
        resumed = Job(self.args, current)
        self.assertEqual(resumed.state["request"]["config"], legacy)
        self.assertEqual(read_json(job.state_path)["request"]["config"], legacy)
        with self.assertRaises(ValueError):
            Job(self.args, {**current, "gpu": 1})

    def test_completed_stage_skips_and_incomplete_artifact_is_archived(self):
        calls = []

        def execute(python, call, log_path, environment):
            calls.append(environment["CUDA_VISIBLE_DEVICES"])
            write_json(call.parameters["output"], {"status": "complete"})

        job = Job(self.args, self.config, execute)
        artifact = job.output / "artifact.json"
        call = FunctionCall("fixture", "stage", {"output": str(artifact)})
        job.stage("first", sys.executable, call, artifact)
        self.assertEqual(calls, [""])
        self.args.resume = True
        job = Job(self.args, self.config, execute)
        job.stage("first", sys.executable, call, artifact)
        self.assertEqual(calls, [""])
        second = job.output / "second.json"
        write_json(second, {"status": "incomplete"})
        job.stage("retry", sys.executable, FunctionCall("fixture", "stage", {"output": str(second)}), second, gpu=True)
        self.assertTrue((job.output / "second.incomplete.1.json").is_file())
        self.assertEqual(calls, ["", "0"])
        second.unlink()
        with self.assertRaises(FileNotFoundError):
            job.stage("retry", sys.executable, call, second)


if __name__ == "__main__":
    unittest.main()
