"""One request at a time on one GPU; model engines survive between requests."""

import argparse
import hashlib
import json
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import main

from runtime.config import load_config
from runtime.processes import Channel, StagePool


def run(config_path, gpu):
    channel = Channel()
    config_path = Path(config_path).absolute()
    signature = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config = load_config(config_path)
    data_dir = Path(os.environ.get("DUB_DATA_DIR", main.ROOT / "outputs/web"))
    lock_fd = os.environ.get("DUB_WORKER_LOCK_FD")
    descriptors = (int(lock_fd),) if lock_fd is not None else ()
    with redirect_stdout(os.sys.stderr):
        try:
            pool = StagePool(config, config_path, gpu, data_dir / "resident" / f"gpu{gpu}", descriptors)
        except Exception as error:
            traceback.print_exc()
            channel.send({"type": "failed", "error": str(error)})
            return
    channel.send({"type": "ready", "models": pool.loaded})
    try:
        while True:
            try:
                request = channel.receive()
            except EOFError:
                break
            output = Path(request["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            with (output.parent / "pipeline.log").open("a") as log, redirect_stdout(log), redirect_stderr(log):
                try:
                    if hashlib.sha256(config_path.read_bytes()).hexdigest() != signature:
                        raise RuntimeError("Resident model configuration changed; restart the service")
                    arguments = SimpleNamespace(source=request["source"], output=str(output), config=str(config_path),
                        source_filename=request.get("source_filename"),
                        gpu=gpu, source_language=request["source_language"], target_language=request["target_language"],
                        duration_mode=request.get("duration_mode", "fixed"), resume=request["resume"], analysis_plan=None,
                        source_audio_manifest=None, dialogue=None, merge_speaker_turns=None)
                    main.run(arguments, execute=pool.execute)
                    (output / "resident_models.json").write_text(json.dumps(
                        {"gpu": gpu, "engines": pool.snapshot()}, ensure_ascii=False, indent=2) + "\n")
                    result = {"type": "done", "id": request["id"], "ok": True}
                except Exception as error:
                    traceback.print_exc()
                    if any(engine.process.poll() is not None for engine in pool.engines.values()):
                        raise
                    result = {"type": "done", "id": request["id"], "ok": False, "error": str(error)}
            channel.send(result)
    finally:
        pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    run(args.config, args.gpu)
