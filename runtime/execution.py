"""Run a model stage locally, or through the resident worker's private channel."""

import subprocess

from runtime.config import ROOT

_dispatch = None


def bind(dispatch):
    global _dispatch
    _dispatch = dispatch


def run_python(python, module, arguments, *, input_text=None, capture=False, check=True, env=None):
    if _dispatch is not None:
        return _dispatch(str(python), module, list(map(str, arguments)), input_text, check)
    command = [str(python), "-m", module, *map(str, arguments)]
    return subprocess.run(command, input=input_text, text=True, capture_output=capture, check=check, env=env, cwd=ROOT)
