"""Library lookup for the isolated model interpreters."""

from pathlib import Path


def native_library_paths(python):
    # Keep the virtualenv entrypoint rather than resolving its Python symlink.
    environment = Path(python).absolute().parent.parent
    paths = [*sorted(environment.glob("lib/python*/site-packages/nvidia/*/lib")), environment / "lib/ffmpeg"]
    return [str(path) for path in paths if path.is_dir()]


def native_environment(python, environment):
    libraries = native_library_paths(python)
    if not libraries:
        return environment
    existing = environment.get("LD_LIBRARY_PATH")
    return {**environment, "LD_LIBRARY_PATH": ":".join([*libraries, *([existing] if existing else [])])}
