"""Exercise installation choices without creating real model environments."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.configure import ROOT, configure


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lightdubber install ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        for name in ("install.sh", "install_unified.sh"):
            shutil.copy(ROOT / "scripts" / name, self.root / "scripts" / name)
        self.log = self.root / "commands.jsonl"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        uv = self.bin / "uv"
        uv.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\nfrom pathlib import Path\n"
            "with open(os.environ['INSTALL_LOG'], 'a') as stream:\n"
            "    stream.write(json.dumps(['uv', *sys.argv[1:]]) + '\\n')\n"
            "if sys.argv[1] == 'venv':\n"
            "    python = Path(sys.argv[-1]) / 'bin/python'\n"
            "    python.parent.mkdir(parents=True)\n"
            "    python.write_text('#!/bin/sh\\nexit 0\\n')\n"
            "    python.chmod(0o755)\n")
        uv.chmod(0o755)
        self.environment = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", INSTALL_LOG=str(self.log))
        for name in ("DUB_ENV", "TTS_ENV", "LIGHTLLM_ENV"):
            self.environment.pop(name, None)
        self.config = self.root / "config.local.json"
        self.config.write_text('{"existing": true}')

    def install(self, *arguments):
        result = subprocess.run(["bash", str(self.root / "scripts/install.sh"), *arguments],
                                env=self.environment, capture_output=True, text=True)
        self.assertEqual(self.config.read_text(), '{"existing": true}')
        return result

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_local_installs_pipeline_and_tts_only(self):
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / ".venv/bin/python").is_file())
        self.assertTrue((self.root / ".venv-tts/bin/python").is_file())
        self.assertFalse((self.root / ".venv-lightllm").exists())
        requirements = [command[-1] for command in self.commands() if "-r" in command]
        self.assertEqual(requirements, [str(self.root / "requirements-unified.txt"),
                                        str(self.root / "requirements-tts.txt")])

    def test_api_skips_tts(self):
        result = self.install("--tts", "api")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / ".venv/bin/python").is_file())
        self.assertFalse((self.root / ".venv-tts").exists())
        self.assertFalse((self.root / ".venv-lightllm").exists())

    def test_lightllm_is_optional_and_pinned(self):
        result = self.install("--tts", "api", "--translation", "lightllm")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / ".venv-lightllm/bin/python").is_file())
        self.assertFalse((self.root / ".venv-tts").exists())
        sources = [argument for command in self.commands() for argument in command
                   if "github" in argument]
        self.assertEqual(len(sources), 2)
        self.assertTrue(all("1b38e8d4e133874f1af8d0533c668d4fe66a306e" in source for source in sources))
        requirements = next(command for command in self.commands()
                            if any("raw.githubusercontent.com" in argument for argument in command))
        self.assertIn("--index-strategy", requirements)
        self.assertEqual(requirements[requirements.index("--index-strategy") + 1], "unsafe-best-match")

    def test_invalid_options_fail_before_install(self):
        for arguments in (("--tts", "other"), ("--tts",), ("--translation", "other"), ("--unknown",)):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.install(*arguments).returncode, 2)
                self.assertFalse(self.log.exists())

    def test_help_does_not_install(self):
        result = self.install("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.log.exists())


class ConfigureTests(unittest.TestCase):
    def test_local_defaults_and_interpreter_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            python = Path(temporary) / "python"
            python.symlink_to(sys.executable)
            config = configure(temporary, python)
            self.assertEqual(config["python"], str(python))
            self.assertEqual(config["tts_python"], str(ROOT / ".venv-tts/bin/python"))
            self.assertEqual(config["tts_model"], str(Path(temporary) / "Qwen3-TTS-12Hz-1.7B-Base"))
            self.assertEqual(config["translation_backend"], "lightllm")
            self.assertEqual(config["translation_model"], "qwen35-dubbing")
            self.assertEqual(config["translation_api_key_env"], "LIGHTDUBBER_LLM_API_KEY")
            self.assertEqual(config["translation_options"], {})

    def test_api_keeps_model_id_and_uses_main_interpreter(self):
        config = configure("models", sys.executable, tts_backend="openai_compatible",
                           tts_url="https://tts.example/v1", tts_model="provider/speech-v1", tts_voice="voice")
        self.assertEqual(config["tts_model"], "provider/speech-v1")
        self.assertEqual(config["tts_python"], str(Path(sys.executable).absolute()))
        self.assertEqual(config["tts_voice"], "voice")

    def test_api_requires_endpoint_and_model(self):
        for fields in ({"tts_url": "https://tts.example/v1"}, {"tts_model": "speech-v1"}):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "--tts-url and --tts-model"):
                configure("models", sys.executable, tts_backend="openai_compatible", **fields)

    def test_translation_api_requires_endpoint_and_model(self):
        for fields in ({"translation_url": "https://llm.example/v1"}, {"translation_model": "chat-model"}):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "--translation-url and --translation-model"):
                configure("models", sys.executable, translation_backend="openai_compatible", **fields)

    def test_local_translation_service_can_be_configured(self):
        config = configure("models", sys.executable, translation_backend="vllm",
                           translation_url="http://127.0.0.1:19000/v1", translation_model="custom-model")
        self.assertEqual(config["translation_backend"], "vllm")
        self.assertEqual(config["translation_url"], "http://127.0.0.1:19000/v1")
        self.assertEqual(config["translation_model"], "custom-model")
        self.assertEqual(config["tts_backend"], "qwen3tts")

    def test_translation_options_require_a_json_object(self):
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            configure("models", sys.executable, translation_options=["invalid"])

    def test_api_cli_combines_translation_and_tts_without_writing_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "config.json"
            command = [sys.executable, str(ROOT / "scripts/configure.py"), "--models", temporary,
                       "--output", str(output), "--translation-backend", "openai_compatible",
                       "--translation-url", "https://llm.example/v1", "--translation-model", "models/provider/chat",
                       "--translation-api-key-env", "EXAMPLE_LLM_KEY",
                       "--translation-options", '{"thinking":{"type":"disabled"},"max_tokens":2048}',
                       "--tts-backend", "openai_compatible",
                       "--tts-url", "https://tts.example/v1", "--tts-model", "provider/speech"]
            result = subprocess.run(command, env=dict(os.environ, EXAMPLE_LLM_KEY="test-secret-not-for-config"),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(output.read_text())
            self.assertEqual(config["translation_backend"], "openai_compatible")
            self.assertEqual(config["translation_url"], "https://llm.example/v1")
            self.assertEqual(config["translation_model"], "models/provider/chat")
            self.assertEqual(config["translation_api_key_env"], "EXAMPLE_LLM_KEY")
            self.assertEqual(config["translation_options"], {"thinking": {"type": "disabled"}, "max_tokens": 2048})
            self.assertNotIn("test-secret-not-for-config", output.read_text())
            self.assertEqual(config["tts_model"], "provider/speech")
            self.assertEqual(config["tts_python"], str(Path(sys.executable).absolute()))

    def test_existing_config_requires_force(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "config.json"
            output.write_text('{"existing": true}')
            command = [sys.executable, str(ROOT / "scripts/configure.py"), "--models", temporary,
                       "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(), '{"existing": true}')
            result = subprocess.run([*command, "--force"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())["models_dir"], temporary)


if __name__ == "__main__":
    unittest.main()
