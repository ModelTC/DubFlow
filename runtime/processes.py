"""Resident model interpreters communicate through structured Python calls."""

import argparse
import io
import json
import logging
import os
import subprocess
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from runtime.config import ROOT
from runtime.environment import native_environment
from runtime.execution import FunctionCall


class Channel:
    def __init__(self):
        self.reader = sys.stdin
        self.writer = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def send(self, value):
        self.writer.write(json.dumps(value, ensure_ascii=False) + "\n")
        self.writer.flush()

    def receive(self):
        line = self.reader.readline()
        if not line:
            raise EOFError("Resident process channel closed")
        return json.loads(line)


def execute_call(call, capture=False):
    output, errors = io.StringIO(), io.StringIO()
    try:
        if capture:
            with redirect_stdout(output), redirect_stderr(errors):
                call.invoke()
        else:
            call.invoke()
        return {"type": "done", "returncode": 0, "stdout": output.getvalue(), "stderr": errors.getvalue()}
    except Exception as error:
        traceback.print_exc()
        return {"type": "done", "returncode": getattr(error, "returncode", 1),
                "stdout": output.getvalue(), "stderr": errors.getvalue() + str(error)}


def completed_call(python, call, result, check):
    completed = subprocess.CompletedProcess([python, call.module, call.function], result["returncode"],
                                            stdout=result.get("stdout", ""), stderr=result.get("stderr", ""))
    if check:
        completed.check_returncode()
    return completed


@contextmanager
def stage_output(path):
    logger = logging.getLogger()
    previous, level = list(logger.handlers), logger.level
    with Path(path).open("a") as log:
        handler = logging.StreamHandler(log)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        try:
            with redirect_stdout(log), redirect_stderr(log):
                yield
        finally:
            logger.handlers = previous
            logger.setLevel(level)
            handler.close()


def model_roles(config):
    roles = {}
    entries = [(config["audio_python"], "recognition"), (config["speech_python"], "speech"),
               (config["separation_python"], "separation"), (config["video_python"], "video"),
               (config["vision_python"], "vision"), (config["identity_python"], "identity")]
    if config["tts_backend"] == "qwen3tts":
        entries.append((config["tts_python"], "tts"))
    for interpreter, role in entries:
        selected = roles.setdefault(str(Path(interpreter).absolute()), [])
        if role not in selected:
            selected.append(role)
    return roles


def check_services(config):
    from runtime.clients import LipSync, Translator

    clients = [LipSync(config["tbdub_url"])]
    if config["translation_backend"] in ("lightllm", "vllm"):
        clients.append(Translator(config["translation_url"], config["translation_model"],
                                  api_key_env=config["translation_api_key_env"], backend=config["translation_backend"]))
    for client in clients:
        try:
            client.health()
        finally:
            client.close()


def warm_models(roles, config):
    from runtime.models import enable, info

    enable()
    device = "cuda:0"
    for role in roles:
        print(f"Loading resident models: {role}", flush=True)
        if role == "recognition":
            from stages.recognition.model import QwenForcedAligner, nemotron3_model, qwen_asr
            qwen_asr(config["asr_model"], device)
            QwenForcedAligner(config["alignment_model"], device).close()
            nemotron3_model(config["diar_model"], device)
        elif role == "separation":
            from stages.separation.model import load_sam
            load_sam(Path(config["sam_audio_model"]), Path(config["sam_audio_text_encoder"]), device)
        elif role == "tts":
            from stages.speech.model import load_qwen
            load_qwen(config["tts_model"], device)
        elif role == "video":
            from stages.lipsync.model import load_landmarker
            load_landmarker(config["landmarker_model"])
        elif role == "vision":
            from stages.speakers.model import load_loconet, load_s3fd
            load_s3fd(config["s3fd_model"], device)
            load_loconet(config["loconet_model"], device)
        elif role == "identity":
            from stages.speakers.model import load_face_encoder
            load_face_encoder(config["identity_model"])
        elif role != "speech":
            raise ValueError(f"Unknown resident model role: {role}")
    return info()


class Engine:
    def __init__(self, python, roles, config, gpu, log_directory, inherited_fds=()):
        self.python = python
        log_directory.mkdir(parents=True, exist_ok=True)
        self.log = (log_directory / f"{roles[0]}.log").open("a")
        command = [python, "-m", "runtime.processes", "--config", str(config), "--roles", ",".join(roles)]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MODELSCOPE_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
        self.process = subprocess.Popen(command, cwd=ROOT, env=native_environment(python, environment),
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, bufsize=1, pass_fds=inherited_fds)
        self.reader, self.writer = self.process.stdout, self.process.stdin

    def receive(self):
        line = self.reader.readline()
        if not line:
            raise RuntimeError(f"Resident {self.python} stopped (exit={self.process.poll()}); inspect engine logs")
        return json.loads(line)

    def send(self, value):
        self.writer.write(json.dumps(value, ensure_ascii=False) + "\n")
        self.writer.flush()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.close()
        self.writer.close()
        self.log.close()


class StagePool:
    def __init__(self, config, config_path, gpu, log_directory, inherited_fds=()):
        self.engines, self.loaded = {}, []
        self.current_log = None
        check_services(config)
        try:
            for python, roles in model_roles(config).items():
                engine = Engine(python, roles, config_path, gpu, log_directory, inherited_fds)
                self.engines[python] = engine
                ready = engine.receive()
                if ready["type"] != "ready":
                    raise RuntimeError(ready.get("error", "Resident model loading failed"))
                self.loaded.append({"roles": roles, **ready["models"]})
        except BaseException:
            self.close()
            raise

    def call(self, python, call, capture=False, check=True):
        engine = self.engines[str(Path(python).absolute())]
        engine.send({"type": "run", "call": call.payload(), "capture": capture, "log": str(self.current_log)})
        while True:
            result = engine.receive()
            if result["type"] == "call":
                try:
                    nested = self.call(result["python"], FunctionCall(**result["call"]), result["capture"], False)
                    response = {"type": "return", "returncode": nested.returncode,
                                "stdout": nested.stdout, "stderr": nested.stderr}
                except Exception as error:
                    response = {"type": "return", "returncode": 1, "stdout": "", "stderr": str(error)}
                engine.send(response)
                continue
            if result["type"] != "done":
                raise RuntimeError("Unexpected resident stage response")
            return completed_call(python, call, result, check)

    def execute(self, python, call, log_path, environment):
        self.current_log = Path(log_path)
        self.current_log.parent.mkdir(parents=True, exist_ok=True)
        self.call(str(python), call)

    def snapshot(self):
        snapshots = []
        for engine in self.engines.values():
            engine.send({"type": "inspect"})
            snapshots.append(engine.receive()["models"])
        return snapshots

    def close(self):
        for engine in self.engines.values():
            engine.close()


class EngineDispatcher:
    def __init__(self, channel):
        self.channel = channel
        self.python = str(Path(sys.executable).absolute())

    def __call__(self, interpreter, call, capture, check):
        if str(Path(interpreter).absolute()) == self.python:
            result = execute_call(call, capture)
        else:
            self.channel.send({"type": "call", "python": interpreter, "call": call.payload(), "capture": capture})
            result = self.channel.receive()
        return completed_call(interpreter, call, result, check)


def engine_loop(config_path, roles):
    from runtime.config import load_config
    from runtime.execution import bind

    channel = Channel()
    bind(EngineDispatcher(channel))
    with redirect_stdout(sys.stderr):
        try:
            models = warm_models(roles, load_config(config_path))
        except Exception as error:
            traceback.print_exc()
            channel.send({"type": "failed", "error": str(error)})
            raise
    channel.send({"type": "ready", "models": models})
    while True:
        try:
            request = channel.receive()
        except EOFError:
            break
        if request["type"] == "inspect":
            from runtime.models import info
            channel.send({"type": "info", "models": info()})
            continue
        with stage_output(request["log"]):
            result = execute_call(FunctionCall(**request["call"]), request["capture"])
        channel.send(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--roles", required=True)
    arguments = parser.parse_args()
    engine_loop(arguments.config, arguments.roles.split(","))
