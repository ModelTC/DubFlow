"""Stage execution, artifacts and resume validation."""

import json
import os
import subprocess
from pathlib import Path

from runtime.config import ROOT, normalize_config
from runtime.environment import native_environment
from runtime.paths import JobInputs, JobPaths
from runtime.utils import file_signature, read_json, write_json


def execute_stage(python, call, log_path, env):
    with log_path.open("a") as log:
        subprocess.run([str(python), "-m", "runtime.execution"], input=json.dumps(call.payload()), text=True,
                       cwd=ROOT, env=native_environment(python, env),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


class Job:
    def __init__(self, args, config, execute=None):
        self.inputs = JobInputs.from_args(args)
        self.paths = JobPaths(Path(args.output).resolve())
        source, output = self.inputs.source, self.paths.root
        request = {"schema_version": 4, "source": file_signature(source), "config": config,
                   **{name: file_signature(value) if value else None
                      for name in ("analysis_plan", "source_audio_manifest", "dialogue")
                      for value in (getattr(self.inputs, name),)}}
        state_path = self.paths.state
        if state_path.exists():
            state = read_json(state_path)
            previous = {**state["request"], "config": normalize_config(state["request"]["config"])}
            if not args.resume or previous != request:
                raise ValueError("Existing output belongs to another request; choose a new directory")
        else:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(output)
            output.mkdir(parents=True, exist_ok=True)
            state = {"request": request, "status": "running", "completed": []}
            write_json(state_path, state)
        self.paths.logs.mkdir(exist_ok=True)
        self.source, self.output = source, output
        self.state, self.state_path = state, state_path
        self.logs, self.config = self.paths.logs, config
        self.resume = args.resume
        self.execute = execute or execute_stage

    def stage(self, name, python, call, expected, gpu=False):
        state, state_path = self.state, self.state_path
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
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.config["gpu"]) if gpu else "",
               "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false"}
        log = self.logs / f"{name}.log"
        print(f"Running {name}: {log}", flush=True)
        self.execute(python, call, log, env)
        if not expected.is_file():
            raise RuntimeError(f"Stage did not create its output: {expected}")
        state["completed"].append(name)
        write_json(state_path, state)
