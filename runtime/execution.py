"""Structured Python calls, either in a child interpreter or a resident engine."""

import importlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from runtime.config import ROOT
from runtime.environment import native_environment

_dispatch = None


def serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


@dataclass(frozen=True)
class FunctionCall:
    module: str
    function: str
    parameters: dict

    def payload(self):
        return serializable(asdict(self))

    def invoke(self):
        function = getattr(importlib.import_module(self.module), self.function)
        return function(**self.parameters)


def bind(dispatch):
    global _dispatch
    _dispatch = dispatch


def call_function(python, module, function, parameters, *, capture=False, check=True, env=None):
    call = FunctionCall(module, function, serializable(parameters))
    if _dispatch is not None:
        return _dispatch(str(python), call, capture, check)
    return subprocess.run([str(python), "-m", "runtime.execution"], input=json.dumps(call.payload()),
                          text=True, capture_output=capture, check=check,
                          env=native_environment(python, dict(os.environ) if env is None else env), cwd=ROOT)


if __name__ == "__main__":
    FunctionCall(**json.load(sys.stdin)).invoke()
