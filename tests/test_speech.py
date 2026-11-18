"""CPU-only regression coverage for synthesis reuse, retry stopping, and API auth."""

import copy
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from runtime.clients import Translator
from stages.speech import run as speech
from stages.speech.model import APISpeechModel
from stages.speech.utils import loudness_references, summarize_timings


class FakeSpeechModel:
    name = "fixture"
    reference_mode = "x_vector_only"

    def __init__(self):
        self.identity = {"backend": "fixture", "revision": 1}
        self.calls = []
        self.durations = {}

    def generate(self, request_path):
        request = json.loads(Path(request_path).read_text())
        self.calls.append(request)
        for row in request["rows"]:
            if Path(row["path"]).exists():
                continue
            samples = round(self.durations.get(row["text"], 2) * 24000)
            timeline = np.arange(samples) / 24000
            audio = (.1 * np.sin(2 * np.pi * 240 * timeline)).astype(np.float32)
            sf.write(row["path"], audio, 24000, subtype="FLOAT")


class SpeechCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        reference = self.output / "reference.wav"
        sf.write(reference, np.sin(np.arange(24000) / 10).astype(np.float32) * .1,
                 24000, subtype="FLOAT")
        self.references = {"speaker": {"path": str(reference), "text": "reference", "rms": .1}}
        self.args = SimpleNamespace(
            seed=42, target_language="en", source_language="zh", device="cpu",
            tts_backend="qwen3tts", tts_model="fixture", tts_python="unused", tts_voice="alloy",
            qwen_x_vector_only_mode=True, duration_mode="fixed", translation_url="unused",
            translation_model="fixture", translation_backend="openai_compatible",
            translation_api_key_env="TEST_LLM_KEY", tts_url="", tts_api_key_env="TEST_TTS_KEY",
        )
        self.record = {
            "id": "u0", "speaker_id": "speaker", "text": "source", "target_text": "same words",
            "start": 0, "end": 1, "window_start": 0, "window_end": 1.1, "status": "pending",
            "tts_seed_index": 0, "translation_attempt": 0, "translation_ready": True,
            "translation_history": [{"round": 0, "text": "same words"}],
        }
        self.backend = FakeSpeechModel()
        factory = patch.object(speech, "create_speech_model", return_value=self.backend)
        factory.start()
        self.addCleanup(factory.stop)

    def initial(self):
        return speech.synthesize([self.record], self.references, self.args, self.output)[0]

    def round_dir(self, number):
        output = self.output / f"shortening_round_{number}"
        output.mkdir(exist_ok=True)
        return output

    def translator(self, texts):
        calls = []

        def translate(records, conversation, args, shorten=False, output=None):
            calls.append([record["id"] for record in records])
            text = texts[len(calls) - 1]
            return [{**record, "target_text": text, "translation_attempt": record["shortening_round"],
                     "translation_history": [*record["translation_history"],
                                             {"round": record["shortening_round"], "text": text}]}
                    for record in records]

        return calls, translate

    def test_same_text_reuses_exact_wav_and_seed(self):
        item = self.initial()
        path = Path(item["tts_raw_path"])
        original = path.read_bytes()
        candidate = {**item, "translation_attempt": 1}
        result = speech.synthesize([candidate], self.references, self.args, self.round_dir(1), attempt=1)[0]
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(result["tts_seed"], 42)
        self.assertEqual(result["tts_raw_path"], str(path))
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(result["tts_reused"])
        self.assertEqual(summarize_timings(self.output)["tts_requested_utterances"], 1)
        self.assertEqual(summarize_timings(self.output)["tts_cache_hits"], 1)

    def test_api_voice_does_not_require_clean_clone_reference(self):
        self.args.tts_backend = "openai_compatible"
        self.backend.reference_mode = "preset_voice"
        records = [{**self.record, "speaker_id": "overlapping", "start": 0., "end": .1},
                   {**self.record, "id": "u1", "speaker_id": "short", "start": .05, "end": .1}]
        references = loudness_references(records, np.ones(4800, dtype=np.float32) * .1, 48000, "alloy")
        self.assertEqual(set(references), {"overlapping", "short"})
        self.assertAlmostEqual(references["overlapping"]["rms"], .1)
        self.assertNotIn("path", references["overlapping"])
        result = speech.synthesize(records, references, self.args, self.output)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(item["reference_speaker_id"] is None for item in result))
        cached = speech.synthesize(result, references, self.args, self.round_dir(1), attempt=1)
        self.assertTrue(all(item["tts_reused"] for item in cached))
        self.assertEqual(summarize_timings(self.output)["tts_cache_hits"], 2)

    def test_rewritten_identical_reference_is_reusable_on_resume(self):
        item = self.initial()
        path = Path(self.references["speaker"]["path"])
        path.write_bytes(path.read_bytes())
        result = speech.synthesize([item], self.references, self.args, self.round_dir(1), attempt=1)[0]
        self.assertTrue(result["tts_reused"])
        self.assertEqual(len(self.backend.calls), 1)

    def test_changed_conditioning_invalidates_cache(self):
        item = self.initial()
        for number, change in enumerate(("text", "seed", "reference", "speaker", "model", "audio"), 1):
            with self.subTest(change=change):
                candidate, args, references = copy.deepcopy(item), copy.copy(self.args), copy.deepcopy(self.references)
                if change == "text":
                    candidate["target_text"] = "different words"
                elif change == "seed":
                    args.seed += 1
                elif change == "reference":
                    references["speaker"]["text"] = "changed reference"
                elif change == "speaker":
                    candidate["speaker_id"] = "other"
                    references["other"] = references["speaker"]
                elif change == "model":
                    self.backend.identity = {"backend": "fixture", "revision": 2}
                elif change == "audio":
                    Path(item["tts_raw_path"]).write_bytes(b"corrupt")
                before = len(self.backend.calls)
                result = speech.synthesize([candidate], references, args, self.round_dir(number), attempt=number)[0]
                self.assertEqual(len(self.backend.calls), before + 1)
                self.assertFalse(result["tts_reused"])
                self.backend.identity = {"backend": "fixture", "revision": 1}

    def test_unchanged_translation_stops_and_resumes(self):
        item = self.initial()
        calls, translate = self.translator(["same words"])
        with patch.object(speech, "translate", translate):
            result = speech.shorten_to_fit([item], self.references, self.args, self.output)[0]
            self.assertEqual(result["shortening_stopped_reason"], "unchanged_translation")
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(self.backend.calls), 1)
            speech.shorten_to_fit([result], self.references, self.args, self.output)
            self.assertEqual(len(calls), 1)

    def test_changed_text_without_duration_gain_keeps_previous(self):
        item = self.initial()
        self.backend.durations["longer words"] = 2.5
        calls, translate = self.translator(["longer words"])
        with patch.object(speech, "translate", translate):
            result = speech.shorten_to_fit([item], self.references, self.args, self.output)[0]
        self.assertEqual(result["shortening_stopped_reason"], "no_duration_improvement")
        self.assertEqual(result["tts_raw_path"], item["tts_raw_path"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(self.backend.calls), 2)

    def test_run_revalidates_completed_audio_on_resume(self):
        self.args.output_dir = str(self.output / "speech")
        self.args.plan = str(self.output / "plan.json")
        self.args.speech_source = str(self.output / "source.wav")
        Path(self.args.plan).write_text(json.dumps({"frames": 48, "fps": 24}))
        sf.write(self.args.speech_source, np.ones(96000, dtype=np.float32) * .1, 48000, subtype="FLOAT")

        def assemble(records, references, plan, args, output):
            result = {"utterances": records, "voice_references": references}
            speech.write_json(output / "dialogue.json", result)
            return result

        with patch.object(speech, "speaking_windows", return_value=[self.record]), \
                patch.object(speech, "voice_references", return_value=self.references), \
                patch.object(speech, "shorten_to_fit", side_effect=lambda records, *args: records), \
                patch.object(speech, "assemble", assemble):
            original = speech.run_stage(**vars(self.args))
            path = Path(original["utterances"][0]["tts_raw_path"])
            audio = path.read_bytes()
            speech.run_stage(**vars(self.args))
            self.assertEqual(len(self.backend.calls), 1)
            path.write_bytes(b"corrupt")
            recovered = speech.run_stage(**vars(self.args))
            self.assertEqual(path.read_bytes(), audio)
            self.assertEqual(len(self.backend.calls), 2)
            self.assertFalse(recovered["utterances"][0]["tts_reused"])

    def test_resume_after_improvement_continues_next_round(self):
        item = self.initial()
        self.backend.durations["short words"] = 1.5
        calls, translate = self.translator(["short words"])
        with patch.object(speech, "translate", translate), patch.object(speech, "SHORTENING_ROUNDS", 1):
            item = speech.shorten_to_fit([item], self.references, self.args, self.output)[0]
        self.assertEqual(item["shortening_rounds_completed"], 1)
        self.assertNotIn("shortening_stopped_reason", item)
        calls, translate = self.translator(["short words"])
        with patch.object(speech, "translate", translate):
            result = speech.shorten_to_fit([item], self.references, self.args, self.output)[0]
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["shortening_rounds_completed"], 2)
        self.assertEqual(result["shortening_stopped_reason"], "unchanged_translation")
        self.assertEqual(len(self.backend.calls), 2)

    def test_improvement_continues_until_unchanged(self):
        item = self.initial()
        self.backend.durations["short words"] = 1.5
        calls, translate = self.translator(["short words", "short words"])
        with patch.object(speech, "translate", translate):
            result = speech.shorten_to_fit([item], self.references, self.args, self.output)[0]
        self.assertEqual(result["target_text"], "short words")
        self.assertEqual(result["shortening_stopped_reason"], "unchanged_translation")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.backend.calls), 2)


class SpeechAPITests(unittest.TestCase):
    def setUp(self):
        stream = io.BytesIO()
        sf.write(stream, np.ones(2400, dtype=np.float32) * .1, 24000, format="WAV", subtype="FLOAT")
        audio, self.received = stream.getvalue(), []
        received = self.received

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, self.headers.get("Authorization"), payload))
                self.send_response(200)
                self.end_headers()
                if self.path.endswith("/chat/completions"):
                    self.wfile.write(json.dumps({"choices": [{
                        "finish_reason": "stop", "message": {"content": "translation"},
                    }]}).encode())
                else:
                    self.wfile.write(audio)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        environment = patch.dict(os.environ, {"TEST_LLM_KEY": "test-llm", "TEST_TTS_KEY": "test-tts",
                                              "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"})
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request_translation(self, model, backend="openai_compatible", options=None):
        client = Translator(self.url, model, "TEST_LLM_KEY", backend, options)
        try:
            self.assertEqual(client("prompt"), "translation")
        finally:
            client.close()
        path, authorization, payload = self.received[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(authorization, "Bearer test-llm")
        self.assertEqual(payload["model"], model)
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": "prompt"})
        self.assertIs(payload["stream"], False)
        return payload

    def test_separate_api_auth_and_preset_voice_wav(self):
        payload = self.request_translation("remote-llm")
        self.assertEqual(set(payload), {"model", "messages", "stream"})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "tts.wav"
            request = Path(temporary) / "request.json"
            args = SimpleNamespace(tts_model="remote-tts", tts_url=self.url,
                                   tts_voice="voice-a", tts_api_key_env="TEST_TTS_KEY")
            request.write_text(json.dumps({"rows": [{"text": "hello", "path": str(output)}]}))
            backend = APISpeechModel(args)
            backend.generate(request)
            waveform, rate = sf.read(output)
            self.assertEqual(rate, 24000)
            self.assertEqual(len(waveform), 2400)
            self.assertNotIn("test-tts", json.dumps(backend.identity))
        self.assertEqual(self.received[1][1], "Bearer test-tts")
        self.assertEqual(self.received[1][2]["voice"], "voice-a")
        self.assertEqual(self.received[1][2]["response_format"], "wav")

    def test_deepseek_thinking_options_are_sent_in_request_body(self):
        options = {"thinking": {"type": "disabled"}, "max_tokens": 2048}
        payload = self.request_translation("deepseek-chat", options=options)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(set(payload), {"model", "messages", "stream", "thinking", "max_tokens"})
        self.assertEqual(options, {"thinking": {"type": "disabled"}, "max_tokens": 2048})

    def test_gpt_reasoning_options_do_not_include_legacy_parameters(self):
        payload = self.request_translation("gpt-5", options={
            "reasoning_effort": "low", "max_completion_tokens": 2048,
        })
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["max_completion_tokens"], 2048)
        self.assertEqual(set(payload), {"model", "messages", "stream", "reasoning_effort",
                                       "max_completion_tokens"})

    def test_local_translation_defaults_are_preserved(self):
        for backend in ("lightllm", "vllm"):
            with self.subTest(backend=backend):
                payload = self.request_translation("local-model", backend=backend)
                self.assertEqual(payload["temperature"], 0)
                self.assertEqual(payload["max_tokens"], 1024)
                self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
                self.assertNotIn("reasoning_effort", payload)
                self.assertNotIn("max_completion_tokens", payload)

    def test_speech_stage_passes_options_to_api_and_records_them_for_resume(self):
        options = {"thinking": {"type": "disabled"}, "max_tokens": 2048}
        record = {"id": "u0", "speaker_id": "speaker", "text": "source", "start": 0., "end": .5,
                  "window_start": 0., "window_end": 1., "status": "pending", "human_speech": True}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, plan, output = directory / "source.wav", directory / "plan.json", directory / "speech"
            sf.write(source, np.zeros(48000, dtype=np.float32), 48000, subtype="FLOAT")
            plan.write_text(json.dumps({"frames": 24, "fps": 24}))
            settings = {"plan": str(plan), "speech_source": str(source), "output_dir": str(output),
                        "source_language": "zh", "target_language": "en", "tts_model": "fixture",
                        "translation_url": self.url, "translation_model": "deepseek-chat",
                        "translation_backend": "openai_compatible", "translation_api_key_env": "TEST_LLM_KEY",
                        "translation_options": options}
            with patch.object(speech, "create_speech_model", return_value=FakeSpeechModel()), \
                    patch.object(speech, "speaking_windows", return_value=[record]), \
                    patch.object(speech, "voice_references", return_value={"speaker": {"rms": .1}}), \
                    patch.object(speech, "synthesize", side_effect=lambda rows, *args: rows), \
                    patch.object(speech, "shorten_to_fit", side_effect=lambda rows, *args: rows), \
                    patch.object(speech, "assemble", side_effect=lambda rows, *args: {"utterances": rows}):
                result = speech.run_stage(**settings)
                self.assertEqual(result["utterances"][0]["target_text"], "translation")
                self.assertEqual(len(self.received), 1)
                payload = self.received[0][2]
                self.assertEqual(payload["thinking"], options["thinking"])
                self.assertEqual(payload["max_tokens"], 2048)
                self.assertNotIn("temperature", payload)
                saved = json.loads((output / "request.json").read_text())
                self.assertEqual(saved["backends"]["translation_options"], options)
                with self.assertRaisesRegex(ValueError, "another request"):
                    speech.run_stage(**{**settings, "translation_options": {"max_tokens": 4096}})
                self.assertEqual(len(self.received), 1)


if __name__ == "__main__":
    unittest.main()
