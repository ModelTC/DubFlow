"""阶段执行、日志和断点记录，与具体翻译模型解耦。"""

import json
import os
import subprocess
from pathlib import Path

from runtime.config import ROOT
from runtime.processes import native_environment


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def file_signature(path):
    path = Path(path).resolve()
    info = path.stat()
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def execute_stage(python, module, arguments, log_path, env):
    command = [str(python), "-m", module, *map(str, arguments)]
    with log_path.open("a") as log:
        subprocess.run(command, cwd=ROOT, env=native_environment(python, env),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


class Job:
    def __init__(self, args, config, execute=None):
        source, output = Path(args.source).resolve(), Path(args.output).resolve()
        request = {"schema_version": 4, "source": file_signature(source), "config": config,
                   "analysis_plan": file_signature(args.analysis_plan) if args.analysis_plan else None,
                   "source_audio_manifest": file_signature(args.source_audio_manifest) if args.source_audio_manifest else None,
                   "dialogue": file_signature(args.dialogue) if args.dialogue else None}
        state_path = output / "run.json"
        if state_path.exists():
            state = read_json(state_path)
            if not args.resume or state["request"] != request:
                raise ValueError("Existing output belongs to another request; choose a new directory")
        else:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(output)
            output.mkdir(parents=True, exist_ok=True)
            state = {"request": request, "status": "running", "completed": []}
            write_json(state_path, state)
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        self.source, self.output = source, output
        self.state, self.state_path = state, state_path
        self.logs, self.config = logs, config
        self.resume = args.resume
        self.execute = execute or execute_stage

    def stage(self, name, python, module, arguments, expected, gpu=False):
        state, state_path = self.state, self.state_path
        logs, config = self.logs, self.config
        expected = Path(expected)
        if self.resume and name not in state["completed"] and expected.is_file():
            try:
                artifact = read_json(expected)
            except json.JSONDecodeError:
                artifact = {"status": "incomplete"}
            if artifact.get("status", "complete") == "complete":
                state["completed"].append(name)
                write_json(state_path, state)
            elif artifact.get("status") == "incomplete":
                # A failed check is evidence, not a reusable completed stage.
                # Keep it before rerunning tools which refuse to overwrite files.
                index = 1
                archived = expected.with_name(f"{expected.stem}.incomplete.{index}{expected.suffix}")
                while archived.exists():
                    index += 1
                    archived = expected.with_name(f"{expected.stem}.incomplete.{index}{expected.suffix}")
                expected.rename(archived)
        if name in state["completed"]:
            if not expected.is_file():
                raise FileNotFoundError(f"Completed stage lost its output: {expected}")
            print(f"Retaining {name}", flush=True)
            return
        state.update(status="running", stage=name)
        write_json(state_path, state)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(config["gpu"]) if gpu else "",
               "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false"}
        print(f"Running {name}: {logs / (name + '.log')}", flush=True)
        self.execute(python, module, arguments, logs / f"{name}.log", env)
        if not expected.is_file():
            raise RuntimeError(f"Stage did not create its output: {expected}")
        state["completed"].append(name)
        write_json(state_path, state)
