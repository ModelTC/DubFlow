"""Private, serial model engines keep incompatible environments resident on one GPU."""

import argparse
import io
import json
import os
import runpy
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from runtime.config import ROOT


class Channel:
    def __init__(self):
        self.reader = sys.stdin
        self.writer = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
        # Native libraries may write directly to fd 1; only this duplicate carries protocol messages.
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def send(self, value):
        self.writer.write(json.dumps(value, ensure_ascii=False) + "\n")
        self.writer.flush()

    def receive(self):
        line = self.reader.readline()
        if not line:
            raise EOFError("Resident process channel closed")
        return json.loads(line)


def run_module(module, arguments, input_text=None):
    original_argv, original_stdin = sys.argv, sys.stdin
    sys.argv = [module, *arguments]
    if input_text is not None:
        sys.stdin = io.StringIO(input_text)
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as error:
        if error.code:
            raise subprocess.CalledProcessError(error.code if isinstance(error.code, int) else 1, sys.argv) from None
    finally:
        sys.argv, sys.stdin = original_argv, original_stdin


def model_roles(config):
    roles = {}
    entries = [(config["audio_python"], "recognition"), (config["speech_python"], "recognition"),
               (config["separation_python"], "separation"), (config["video_python"], "video"),
               (config["vision_python"], "vision"),
               (config["identity_python"], "identity"),
               (config["tts_python"], "tts")]
    for interpreter, role in entries:
        python = str((ROOT / interpreter).absolute())
        selected = roles.setdefault(python, [])
        if role not in selected:
            selected.append(role)
    return roles


def warm_models(roles, config):
    from runtime.models import enable, info

    enable()
    device = "cuda:0"
    from runtime.clients import LipSync, Translator
    for client in (Translator(config["translation_url"], config["translation_model"]), LipSync(config["tbdub_url"])):
        try:
            client.health()
        finally:
            client.close()
    for role in roles:
        print(f"Loading resident models: {role}", flush=True)
        if role == "recognition":
            from stages.recognition.models import (
                QwenForcedAligner,
                nemotron3_model,
                qwen_asr,
            )
            qwen_asr(config["asr_model"], device)
            QwenForcedAligner(config["alignment_model"], device).close()
            nemotron3_model(config["diar_model"], device)
        elif role == "separation":
            from stages.separate import load_sam
            load_sam(Path(config["sam_audio_model"]), Path(config["sam_audio_text_encoder"]), device)
        elif role == "tts":
            from stages.speech.models import load_qwen
            load_qwen(config["tts_model"], device)
        elif role == "video":
            from stages.lipsync.crop import load_landmarker
            load_landmarker(config["landmarker_model"])
        elif role == "vision":
            from stages.speakers.active_speaker import load_loconet
            from stages.speakers.detect import load_s3fd
            repo = ROOT / config["s3fd_repo"]
            load_s3fd(repo, config["s3fd_model"], device)
            load_loconet(ROOT / config["loconet_model"], device)
        elif role == "identity":
            from stages.speakers.identity import load_face_encoder
            load_face_encoder(config["identity_model"])
        else:
            raise ValueError(f"Unknown resident model role: {role}")
    return info()


def native_library_paths(python):
    # 保留虚拟环境的解释器路径；resolve() 会跟随软链跳到系统 Python。
    environment = Path(python).absolute().parent.parent
    paths = [*sorted(environment.glob("lib/python*/site-packages/nvidia/*/lib")), environment / "lib/ffmpeg"]
    return [str(path) for path in paths if path.is_dir()]


def native_environment(python, environment):
    libraries = native_library_paths(python)
    if not libraries:
        return environment
    existing = environment.get("LD_LIBRARY_PATH")
    return {**environment, "LD_LIBRARY_PATH": ":".join([*libraries, *([existing] if existing else [])])}


class Engine:
    def __init__(self, python, roles, config, gpu, log_directory, inherited_fds=()):
        self.python = python
        log_directory.mkdir(parents=True, exist_ok=True)
        self.log = (log_directory / f"{roles[0]}.log").open("a")
        command = [python, "-m", "runtime.processes", "--config", str(config), "--roles", ",".join(roles)]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MODELSCOPE_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
        environment = native_environment(python, environment)
        self.process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
                                        pass_fds=inherited_fds)
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
        self.engines = {}
        self.loaded = []
        self.current_log = None
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

    def call(self, python, module, arguments, input_text=None, check=True):
        engine = self.engines[str(Path(python).absolute())]
        engine.send({"type": "run", "module": module, "arguments": list(map(str, arguments)),
                     "input": input_text, "log": str(self.current_log)})
        while True:
            result = engine.receive()
            if result["type"] == "call":
                try:
                    nested = self.call(result["python"], result["module"], result["arguments"], result["input"], False)
                    engine.send({"type": "return", "returncode": nested.returncode, "stderr": nested.stderr})
                except Exception as error:
                    engine.send({"type": "return", "returncode": 1, "stderr": str(error)})
                continue
            if result["type"] != "done":
                raise RuntimeError("Unexpected resident stage response")
            completed = subprocess.CompletedProcess([python, module, *arguments], result["returncode"], stderr=result.get("error", ""))
            if check:
                completed.check_returncode()
            return completed

    def execute(self, python, module, arguments, log_path, environment):
        self.current_log = Path(log_path)
        self.current_log.parent.mkdir(parents=True, exist_ok=True)
        self.call(str(python), module, arguments)

    def snapshot(self):
        snapshots = []
        for engine in self.engines.values():
            engine.send({"type": "inspect"})
            snapshots.append(engine.receive()["models"])
        return snapshots

    def close(self):
        for engine in self.engines.values():
            engine.close()


def engine_loop(config_path, roles):
    from runtime.config import load_config
    from runtime.execution import bind

    channel = Channel()
    python = str(Path(sys.executable).absolute())

    def nested(interpreter, module, arguments, input_text, check):
        if str(Path(interpreter).absolute()) == python:
            try:
                run_module(module, arguments, input_text)
                result = subprocess.CompletedProcess([interpreter, module, *arguments], 0, stderr="")
            except Exception as error:
                result = subprocess.CompletedProcess([interpreter, module, *arguments], 1, stderr=str(error))
        else:
            channel.send({"type": "call", "python": interpreter, "module": module,
                          "arguments": arguments, "input": input_text})
            result = channel.receive()
            result = subprocess.CompletedProcess([interpreter, module, *arguments], result["returncode"], stderr=result["stderr"])
        if check:
            result.check_returncode()
        return result

    bind(nested)
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
        with Path(request["log"]).open("a") as log, redirect_stdout(log), redirect_stderr(log):
            try:
                run_module(request["module"], request["arguments"], request["input"])
                result = {"type": "done", "returncode": 0}
            except Exception as error:
                traceback.print_exc()
                result = {"type": "done", "returncode": getattr(error, "returncode", 1), "error": str(error)}
        channel.send(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--roles", required=True)
    arguments = parser.parse_args()
    engine_loop(arguments.config, arguments.roles.split(","))
