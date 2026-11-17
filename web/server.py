"""A private-per-browser upload queue around the existing translation pipeline."""

import asyncio
import fcntl
import hashlib
import json
import logging
import math
import os
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from runtime.config import ROOT, load_config
from runtime.languages import LANGUAGE_LABELS
from runtime.languages import LANGUAGES as SUPPORTED_LANGUAGES
from stages.speech.models import validate_qwen_assets
from starlette.requests import ClientDisconnect

STATIC = Path(__file__).parent / "static"
COOKIE = "dub_session"
LANGUAGES = [{"code": code, "label": LANGUAGE_LABELS[code]} for code in SUPPORTED_LANGUAGES]
LANGUAGE_PATTERN = "^(" + "|".join(SUPPORTED_LANGUAGES) + ")$"


def available_languages(settings):
    return LANGUAGES


ARTIFACTS = {"final.mp4", "comparison.mp4", "source.txt", "translated.txt", "report.json"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
STAGES = {
    "prepare": "准备视频", "vision": "定位人物和说话画面", "identity": "关联人物身份",
    "fusion": "绑定人物与台词", "source_separation": "分离对白和背景音",
    "speech": "翻译与配音", "audio_mix": "保留背景并混音",
    "target": "准备口型片段", "rois": "准备人物编辑区域", "render": "生成同步口型",
    "compose": "拼接视频", "adaptive_duration": "按配音调整画面时长", "report": "整理翻译报告",
    "audio_nemotron3": "识别对白与说话人",
}
PROGRESS_STAGES = ["prepare", "vision", "identity", "audio_nemotron3", "fusion", "source_separation",
                   "speech", "audio_mix", "target", "rois", "render", "compose", "report"]
STATS = ("translated", "preserved_original", "lip_edited", "partial_lip_edited", "audio_only")
PROFILES = {
    "modern": {"label": "视频翻译", "models": [
        {"stage": "原文识别", "name": "Qwen3-ASR-1.7B"},
        {"stage": "词时间", "name": "Qwen3-ForcedAligner-0.6B"},
        {"stage": "说话人", "name": "Nemotron 3"},
        {"stage": "发言定位", "name": "UniTalk-LoCoNet"},
        {"stage": "对白分离", "name": "SAM-Audio Large-TV"},
        {"stage": "翻译", "name": "Qwen3.5-9B · vLLM"},
        {"stage": "配音", "name": "Qwen3-TTS 1.7B Base"},
        {"stage": "口型", "name": "TBDub"},
    ]},
}


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DUB_DATA_DIR", ROOT / "outputs/web")))
    config: Path = field(default_factory=lambda: Path(os.environ.get("DUB_CONFIG", ROOT / "config.local.json")))
    gpus: tuple[int, ...] = field(default_factory=lambda: tuple(int(value) for value in os.environ.get("DUB_GPUS", "6,7").split(",")))
    python: str = field(default_factory=lambda: os.environ.get("DUB_PYTHON", sys.executable))
    max_upload_bytes: int = field(default_factory=lambda: int(os.environ.get("DUB_MAX_UPLOAD_MB", "250")) * 1024**2)
    max_duration_seconds: float = field(default_factory=lambda: float(os.environ.get("DUB_MAX_DURATION_SECONDS", "0")))
    min_free_bytes: int = field(default_factory=lambda: int(os.environ.get("DUB_MIN_FREE_GB", "20")) * 1024**3)
    max_pending: int = field(default_factory=lambda: int(os.environ.get("DUB_MAX_PENDING", "8")))
    session_pending: int = field(default_factory=lambda: int(os.environ.get("DUB_SESSION_PENDING", "4")))
    chunk_bytes: int = field(default_factory=lambda: int(os.environ.get("DUB_UPLOAD_CHUNK_MB", "32")) * 1024**2)
    upload_ttl_seconds: int = 3600

    def __post_init__(self):
        self.data_dir = self.data_dir.resolve()
        self.config = self.config.resolve()
        if (not self.gpus or len(set(self.gpus)) != len(self.gpus) or min(self.gpus) < 0 or
                min(self.max_upload_bytes, self.chunk_bytes, self.max_pending, self.session_pending) <= 0 or
                not math.isfinite(self.max_duration_seconds) or self.max_duration_seconds < 0):
            raise ValueError("GPU and queue/upload limits must be valid; duration limit must be nonnegative")


class UploadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0)


class JobInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    upload_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    source_language: str = Field(default="auto", pattern="^(auto|" + "|".join(SUPPORTED_LANGUAGES) + ")$")
    target_language: str = Field(pattern=LANGUAGE_PATTERN)
    profile: str = Field(default="modern", pattern=r"^modern$")
    duration_mode: str = Field(default="adaptive_video", pattern=r"^(fixed|adaptive_video)$")


class Store:
    def __init__(self, settings):
        self.settings = settings
        for directory in (settings.data_dir, settings.data_dir / "uploads", settings.data_dir / "jobs"):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)
        self.path = settings.data_dir / "queue.sqlite3"
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, csrf TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, filename TEXT NOT NULL,
                    size INTEGER NOT NULL, offset INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'uploading', created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, upload_id TEXT UNIQUE NOT NULL,
                    source_language TEXT NOT NULL, target_language TEXT NOT NULL,
                    duration_mode TEXT NOT NULL DEFAULT 'adaptive_video',
                    profile TEXT NOT NULL DEFAULT 'modern', assigned_gpu INTEGER,
                    duration REAL NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    recoveries INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '');
            """)
        self.path.chmod(0o600)

    @contextmanager
    def connection(self, write=False):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def job(self, job_id, owner=None):
        with self.connection() as db:
            row = db.execute("SELECT j.*, u.filename FROM jobs j JOIN uploads u ON u.id=j.upload_id WHERE j.id=?", (job_id,)).fetchone()
        if row is None or owner is not None and row["owner"] != owner:
            raise HTTPException(404, "任务不存在")
        return row

    def upload(self, db, upload_id, owner):
        row = db.execute("SELECT * FROM uploads WHERE id=? AND owner=?", (upload_id, owner)).fetchone()
        if row is None:
            raise HTTPException(404, "上传不存在")
        return row

    def upload_path(self, upload_id):
        return self.settings.data_dir / "uploads" / upload_id / "source"

    def output(self, job_id):
        return self.settings.data_dir / "jobs" / job_id / "output"

    def capacity(self, db, owner, include_uploads=False):
        pending = "SELECT owner FROM jobs WHERE status IN ('queued','running')"
        if include_uploads:
            pending += " UNION ALL SELECT owner FROM uploads WHERE status='uploading'"
        rows = db.execute(pending).fetchall()
        if len(rows) >= self.settings.max_pending:
            raise HTTPException(429, "当前队列已满，请稍后再试")
        if sum(row["owner"] == owner for row in rows) >= self.settings.session_pending:
            raise HTTPException(429, f"每个浏览器最多同时处理 {self.settings.session_pending} 个视频，请等待已有任务完成")

    def expire_uploads(self, db):
        expired = db.execute("SELECT id FROM uploads WHERE status='uploading' AND created_at<?",
                             (time.time() - self.settings.upload_ttl_seconds,)).fetchall()
        for row in expired:
            # These paths were created by this service, never supplied by users.
            self.upload_path(row["id"]).unlink(missing_ok=True)
            db.execute("UPDATE uploads SET status='expired' WHERE id=?", (row["id"],))


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def public_completeness(report):
    if "audio_completeness" not in report:
        return None
    return {key: report["audio_completeness"][key] for key in
            ("recognized_dialogue", "replaced_dialogue", "unreplaced_ids", "scope") if key in report["audio_completeness"]}


def public_video_result(report, state):
    result = {}
    if report.get("translation_skipped") == "auto_detected_same_as_target":
        result["translation_skipped"] = report["translation_skipped"]
        result["other_detected_languages"] = report.get("other_detected_languages", [])
    if report.get("content_verification") == "disabled":
        result["content_verification"] = "disabled"
    if report.get("video_mode") in {"lip_edited", "partially_lip_edited", "audio_only", "preserve_original"}:
        result["video_mode"] = report["video_mode"]
    return result


def result_video_name(row, state):
    if state.get("result_mode") != "passthrough_original":
        return "final.mp4"
    suffix = Path(row["filename"]).suffix.lower()
    return "final" + (suffix if suffix in VIDEO_EXTENSIONS else ".bin")


def translation_model_name(config):
    return config["translation_model"] + " · vLLM"


def profile_catalog(settings):
    try:
        config = load_config(settings.config)
    except (OSError, ValueError, KeyError):
        return [{"id": "modern", **PROFILES["modern"], "available": False, "reason": "模型配置尚未就绪"}]
    missing = []
    for key, label in (("asr_model", "Qwen3-ASR"), ("alignment_model", "Qwen3 对齐器"),
                       ("diar_model", "Nemotron 3")):
        path = ROOT / config.get(key, "")
        metadata = "config.json"
        has_weights = any(any(path.glob(pattern)) for pattern in ("*.safetensors", "*.bin", "*.pth"))
        if not (path / metadata).is_file() or not has_weights:
            missing.append(label)
    sam = ROOT / config.get("sam_audio_model", "")
    if not all((sam / name).is_file() for name in ("config.json", "checkpoint.pt")):
        missing.append("SAM-Audio")
    text_encoder = ROOT / config.get("sam_audio_text_encoder", "")
    if (not (text_encoder / "config.json").is_file() or
            not any((text_encoder / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")) or
            not any((text_encoder / name).is_file() for name in ("tokenizer.json", "spiece.model"))):
        missing.append("SAM 文本编码器")
    try:
        validate_qwen_assets(ROOT / config["tts_model"])
    except FileNotFoundError:
        missing.append("Qwen3-TTS 1.7B Base")
    return [{"id": "modern", **PROFILES["modern"],
             "models": job_models("modern", {"request": {"config": config | {"content_verification": "disabled"}}}),
             "available": not missing, "reason": "尚未就绪：" + "、".join(missing) if missing else ""}]


def job_models(profile, state):
    models = PROFILES[profile]["models"]
    config = state.get("request", {}).get("config", {})
    if not config:
        return models
    names = {"对白分离": "SAM-Audio Large-TV", "翻译": translation_model_name(config),
             "配音": "Qwen3-TTS 1.7B Base", "口型": "TBDub Student · LightX2V",
             "词时间": "Qwen3-ForcedAligner-0.6B"}
    return [model | {"name": names[model["stage"]]} if model["stage"] in names else model for model in models]


def probe_video(path, settings):
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
                                 "-format_whitelist", "mov,matroska,webm,avi",
                                 "-show_streams", "-show_format", "-of", "json", str(path)],
                                capture_output=True, timeout=20, check=True)
        media = json.loads(result.stdout)
        streams = media["streams"]
        video = next(stream for stream in streams if stream["codec_type"] == "video")
        if not any(stream["codec_type"] == "audio" for stream in streams):
            raise ValueError("missing audio")
        duration = float(media["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0 or video["width"] <= 0 or video["height"] <= 0:
            raise ValueError("invalid media")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, StopIteration, TypeError):
        raise HTTPException(422, "无法读取视频，请上传包含画面和音轨的有效视频文件") from None
    if settings.max_duration_seconds > 0 and duration > settings.max_duration_seconds:
        raise HTTPException(422, f"视频最长支持 {settings.max_duration_seconds:g} 秒")
    return duration


class Worker:
    def __init__(self, store):
        self.store = store
        self.settings = store.settings
        self.stop = threading.Event()
        self.wakeup = threading.Event()
        self.process_lock = threading.Lock()
        self.processes = {}
        self.active_jobs = {}
        self.ready = {gpu: False for gpu in self.settings.gpus}
        self.threads = []
        self.lock_file = None

    def start(self):
        # 一个 HTTP 进程管理整个 GPU 池，不能再由 uvicorn 复制模型。
        self.lock_file = (self.settings.data_dir / "worker.lock").open("a")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_file.close()
            raise RuntimeError("Another worker owns this queue; start uvicorn with --workers 1") from None
        with self.store.connection(write=True) as db:
            db.execute("UPDATE jobs SET status='failed', error=?, updated_at=? WHERE status IN ('running','queued') AND profile!='modern'",
                       ("旧流程已停用，请重新提交视频", time.time()))
            db.execute("UPDATE jobs SET status='failed', error=?, updated_at=? WHERE status='running' AND recoveries>=1",
                       ("任务多次被服务重启中断，请重新上传", time.time()))
            db.execute("UPDATE jobs SET status='queued', recoveries=recoveries+1, updated_at=? WHERE status='running'", (time.time(),))
            placeholders = ",".join("?" for _ in self.settings.gpus)
            db.execute(f"UPDATE jobs SET assigned_gpu=NULL WHERE status='queued' AND assigned_gpu IS NOT NULL "
                       f"AND assigned_gpu NOT IN ({placeholders})", self.settings.gpus)
        for gpu in self.settings.gpus:
            thread = threading.Thread(target=self.run, args=(gpu,), name=f"dubbing-gpu-{gpu}", daemon=True)
            self.threads.append(thread)
            thread.start()

    def close(self):
        self.stop.set()
        self.wakeup.set()
        with self.process_lock:
            processes = list(self.processes.values())
        for process in processes:
            self.terminate(process)
        for thread in self.threads:
            thread.join(timeout=10)
        if self.lock_file is not None:
            self.lock_file.close()

    @staticmethod
    def terminate(process):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                pass
        # The pipeline can exit before a model subprocess handles SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)

    def cancel(self, job_id, owner):
        self.store.job(job_id, owner)
        with self.process_lock:
            with self.store.connection(write=True) as db:
                db.execute("UPDATE jobs SET status='canceled', updated_at=? WHERE id=? AND status IN ('queued','running')", (time.time(), job_id))
            process = next((self.processes[gpu] for gpu, active in self.active_jobs.items() if active == job_id), None)
        if process is not None:
            self.terminate(process)
        self.wakeup.set()

    def command(self, gpu):
        return [self.settings.python, "-m", "runtime.worker", "--config", str(self.settings.config), "--gpu", str(gpu)]

    def status(self):
        with self.process_lock:
            return {"total_workers": len(self.settings.gpus), "ready_workers": sum(self.ready.values()),
                    "busy_workers": len(self.active_jobs),
                    "workers": [{"gpu": gpu, "ready": self.ready[gpu], "busy": gpu in self.active_jobs}
                                for gpu in self.settings.gpus]}

    def claim(self, gpu):
        with self.store.connection(write=True) as db:
            row = db.execute("SELECT j.*, u.filename FROM jobs j JOIN uploads u ON u.id=j.upload_id "
                             "WHERE j.status='queued' AND (j.assigned_gpu IS NULL OR j.assigned_gpu=?) "
                             "ORDER BY j.created_at LIMIT 1", (gpu,)).fetchone()
            if row is not None:
                db.execute("UPDATE jobs SET status='running', assigned_gpu=?, updated_at=? WHERE id=?", (gpu, time.time(), row["id"]))
            return row

    @staticmethod
    def receive(process):
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("Resident GPU worker exited without a reply")
        return json.loads(line)

    def run_one(self, row, gpu, process):
        job_id = row["id"]
        if row["profile"] != "modern":
            raise RuntimeError("The previous model route is disabled")
        selected = profile_catalog(self.settings)[0]
        if not selected["available"]:
            raise RuntimeError(selected["reason"])
        if shutil.disk_usage(self.settings.data_dir).free < self.settings.min_free_bytes:
            raise RuntimeError("Insufficient disk space to start a queued job")
        self.store.output(job_id).parent.mkdir(exist_ok=True, mode=0o700)
        request = {"type": "run", "id": job_id, "source": str(self.store.upload_path(row["upload_id"])),
                   "source_filename": row["filename"],
                   "output": str(self.store.output(job_id)), "source_language": row["source_language"],
                   "target_language": row["target_language"], "duration_mode": row["duration_mode"], "resume": True}
        with self.process_lock:
            if self.stop.is_set() or self.store.job(job_id)["status"] == "canceled":
                return
            self.active_jobs[gpu] = job_id
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
        result = self.receive(process)
        if result["type"] != "done" or result["id"] != job_id:
            raise RuntimeError("Resident GPU worker replied to another job")
        with self.process_lock:
            self.active_jobs.pop(gpu, None)
        if self.stop.is_set():
            return  # Leave interrupted work resumable for the next startup.
        output = self.store.output(job_id)
        video = output / result_video_name(row, read_json(output / "run.json"))
        success = (result["ok"] and video.is_file() and video.stat().st_size > 0
                   and read_json(output / "report.json").get("status") == "complete")
        if not result["ok"]:
            logging.error("Dubbing job %s failed: %s", job_id, result.get("error", ""))
        with self.store.connection(write=True) as db:
            db.execute("UPDATE jobs SET status=?, error=?, updated_at=? WHERE id=? AND status='running'",
                       ("succeeded" if success else "failed", "" if success else "处理失败，请稍后重试或联系管理员查看任务日志", time.time(), job_id))

    def run(self, gpu):
        log_path = self.settings.data_dir / f"worker_gpu{gpu}.log"
        while not self.stop.is_set():
            warmed = False
            process = None
            row = None
            try:
                with log_path.open("ab") as log:
                    descriptors = (self.lock_file.fileno(),)
                    with self.process_lock:
                        if self.stop.is_set():
                            break
                        process = subprocess.Popen(self.command(gpu), cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=log, text=True, encoding="utf-8", bufsize=1, start_new_session=True, pass_fds=descriptors,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                                 "DUB_WORKER_LOCK_FD": str(self.lock_file.fileno())})
                        self.processes[gpu] = process
                    if self.receive(process)["type"] != "ready":
                        raise RuntimeError("Resident GPU worker did not finish model warmup")
                    with self.process_lock:
                        self.ready[gpu] = True
                    warmed = True
                    while not self.stop.is_set():
                        self.wakeup.clear()
                        row = self.claim(gpu)
                        if row is None:
                            self.wakeup.wait(1)
                            continue
                        self.run_one(row, gpu, process)
                        row = None
            except Exception:
                canceled = row is not None and self.store.job(row["id"])["status"] == "canceled"
                if not self.stop.is_set() and not canceled:
                    logging.exception("GPU %s worker failed", gpu)
                    if row is not None:
                        with self.store.connection(write=True) as db:
                            db.execute("UPDATE jobs SET status='failed', error=?, updated_at=? WHERE id=? AND status='running'",
                                       ("任务执行异常，请联系管理员查看任务日志", time.time(), row["id"]))
            finally:
                with self.process_lock:
                    self.ready[gpu] = False
                    self.active_jobs.pop(gpu, None)
                    self.processes.pop(gpu, None)
                if process is not None:
                    self.terminate(process)
                    process.stdin.close()
                    process.stdout.close()
            if not warmed:
                break
            self.stop.wait(1)


def create_app(settings=None, start_worker=True):
    settings = settings or Settings()
    store = Store(settings)
    worker = Worker(store)
    upload_slots = asyncio.Semaphore(len(settings.gpus))
    probe_slots = threading.BoundedSemaphore(len(settings.gpus))

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            if not settings.config.is_file():
                raise RuntimeError("DUB_CONFIG does not exist")
            if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
                raise RuntimeError("ffmpeg and ffprobe must be installed")
            profile_catalog(settings)
            worker.start()
        try:
            yield
        finally:
            if start_worker:
                await asyncio.to_thread(worker.close)

    app = FastAPI(title="Video Dubbing", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.store, app.state.worker = store, worker

    def session(request, mutate=False):
        token = request.cookies.get(COOKIE, "")
        owner = hashlib.sha256(token.encode()).hexdigest()
        with store.connection() as db:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (owner,)).fetchone()
        if row is None:
            raise HTTPException(401, "请刷新页面后重试")
        if mutate:
            origin = request.headers.get("origin")
            if request.headers.get("sec-fetch-site") == "cross-site" or origin and urlsplit(origin).netloc != request.headers.get("host"):
                raise HTTPException(403, "不允许跨站请求")
            if not secrets.compare_digest(request.headers.get("x-csrf-token", ""), row["csrf"]):
                raise HTTPException(403, "请求校验失败，请刷新页面")
        return owner

    def free_space():
        if shutil.disk_usage(settings.data_dir).free < settings.min_free_bytes + settings.max_upload_bytes:
            raise HTTPException(507, "服务器可用磁盘不足，请稍后再试")

    def public_job(row):
        output = store.output(row["id"])
        state = read_json(output / "run.json")
        stage = state.get("stage", "") if row["status"] != "queued" else ""
        completed = set(state.get("completed", []))
        stage_order = state.get("stage_order", PROGRESS_STAGES)
        progress = round(100 * len(completed.intersection(stage_order)) / len(stage_order))
        files, stats, report = {"source": f"/api/jobs/{row['id']}/source"}, {}, {}
        if row["status"] == "succeeded":
            progress = 100
            files.update({name: f"/api/jobs/{row['id']}/files/{name}" for name in sorted(ARTIFACTS) if (output / name).is_file()})
            report = read_json(output / "report.json")
            stats = {key: report.get(key, 0) for key in STATS}
            result_name = result_video_name(row, state)
            if (output / result_name).is_file():
                files[result_name] = f"/api/jobs/{row['id']}/files/{result_name}"
        with store.connection() as db:
            queue_position = db.execute("SELECT count(*) FROM jobs WHERE status='queued' AND created_at<=?", (row["created_at"],)).fetchone()[0] if row["status"] == "queued" else 0
        selected = PROFILES[row["profile"]]
        models = (job_models(row["profile"], state) if state
                  else profile_catalog(settings)[0]["models"])
        public = {key: row[key] for key in ("id", "status", "filename", "source_language", "target_language", "profile", "duration_mode", "duration", "created_at", "updated_at", "error")} | {
            "stage": stage, "stage_label": STAGES.get(stage, "等待处理"), "progress": progress,
            "queue_position": queue_position, "stats": stats, "files": files,
            "profile_label": selected["label"], "models": models,
        }
        if row["status"] == "succeeded":
            public["result_video_file"] = result_video_name(row, state)
        if row["source_language"] == "auto" and report.get("source_language") in SUPPORTED_LANGUAGES:
            public["detected_source_language"] = report["source_language"]
        if row["status"] == "succeeded":
            if state.get("result_mode") == "passthrough_original":
                public["output_duration"] = row["duration"]
            else:
                timing = read_json(output / ("adaptive_duration.json" if row["duration_mode"] == "adaptive_video" else "compose.json"))
                if isinstance(timing.get("duration"), (int, float)):
                    public["output_duration"] = timing["duration"]
        public.update(public_video_result(report, state))
        completeness = public_completeness(report)
        if completeness is not None:
            public["audio_completeness"] = completeness
        return public

    @app.middleware("http")
    async def security_headers(request, call_next):
        if request.method in {"POST", "DELETE"} and request.url.path.startswith("/api/"):
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > 16384:
                    return JSONResponse({"detail": "请求内容过大"}, status_code=413)
                body.extend(chunk)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers.update({"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
                                 "X-Frame-Options": "DENY", "Cross-Origin-Resource-Policy": "same-origin",
                                 "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"})
        if request.url.path == "/" or request.url.path.startswith(("/api/", "/static/")):
            response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.get("/healthz")
    def health():
        if start_worker and any(not thread.is_alive() for thread in worker.threads):
            raise HTTPException(503, "部分模型进程启动失败，请查看服务日志")
        pool = worker.status()
        return {"status": "ok" if not start_worker or pool["ready_workers"] == pool["total_workers"] else "warming", **pool}

    @app.get("/api/bootstrap")
    def bootstrap(request: Request):
        token = request.cookies.get(COOKIE, "")
        owner = hashlib.sha256(token.encode()).hexdigest()
        with store.connection(write=True) as db:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (owner,)).fetchone()
            if row is None:
                token = secrets.token_urlsafe(32)
                owner, csrf = hashlib.sha256(token.encode()).hexdigest(), secrets.token_urlsafe(32)
                db.execute("INSERT INTO sessions VALUES (?,?)", (owner, csrf))
            else:
                csrf = row["csrf"]
        profiles = profile_catalog(settings)
        response = JSONResponse({"csrf_token": csrf, "languages": available_languages(settings), "duration_modes": ["fixed", "adaptive_video"],
                                 "profiles": profiles, "default_profile": "modern",
                                 "service": worker.status(),
                                 "limits": {"max_upload_bytes": settings.max_upload_bytes, "chunk_bytes": settings.chunk_bytes,
                                            "max_duration_seconds": settings.max_duration_seconds}})
        response.set_cookie(COOKIE, token, httponly=True, secure=request.url.scheme == "https", samesite="lax", max_age=30 * 86400)
        return response

    @app.post("/api/uploads", status_code=201)
    def init_upload(body: UploadInput, request: Request):
        owner = session(request, mutate=True)
        if body.size > settings.max_upload_bytes:
            raise HTTPException(413, "视频文件超过上传大小限制")
        filename = body.filename.replace("\\", "/").rsplit("/", 1)[-1]
        filename = "".join(character for character in filename if character.isprintable())
        if Path(filename).suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            raise HTTPException(422, "请选择 MP4、MOV、MKV、WebM、AVI 或 M4V 视频")
        free_space()
        upload_id = secrets.token_hex(16)
        with store.connection(write=True) as db:
            store.expire_uploads(db)
            store.capacity(db, owner, include_uploads=True)
            source = store.upload_path(upload_id)
            source.parent.mkdir(mode=0o700)
            source.touch(mode=0o600)
            db.execute("INSERT INTO uploads (id,owner,filename,size,created_at) VALUES (?,?,?,?,?)", (upload_id, owner, filename, body.size, time.time()))
        return {"upload_id": upload_id, "offset": 0, "chunk_bytes": settings.chunk_bytes}

    @app.put("/api/uploads/{upload_id}")
    async def upload_chunk(upload_id: str, request: Request):
        owner = session(request, mutate=True)
        try:
            offset = int(request.headers.get("upload-offset", "-1"))
        except ValueError:
            raise HTTPException(422, "Upload-Offset 必须是字节偏移") from None
        with store.connection() as db:
            row = store.upload(db, upload_id, owner)
            if row["status"] != "uploading" or offset != row["offset"]:
                raise HTTPException(409, "上传偏移或状态不匹配，请重新上传")
        async with upload_slots:
            data = bytearray()
            try:
                async for chunk in request.stream():
                    if len(data) + len(chunk) > settings.chunk_bytes:
                        raise HTTPException(413, f"单个分块不能超过 {settings.chunk_bytes / 1024**2:g} MB")
                    data.extend(chunk)
            except ClientDisconnect:
                logging.warning("Upload disconnected: id=%s offset=%d received=%d bytes; offset unchanged",
                                upload_id, offset, len(data))
                return JSONResponse({"detail": "上传连接中断，已提交分块保留，可继续上传"}, status_code=499)
        if not data:
            raise HTTPException(422, "上传分块不能为空")
        with store.connection(write=True) as db:
            row = store.upload(db, upload_id, owner)
            if row["status"] != "uploading" or offset != row["offset"]:
                raise HTTPException(409, "上传偏移或状态不匹配，请重新上传")
            if offset + len(data) > row["size"]:
                raise HTTPException(413, "上传内容超过申报文件大小")
            free_space()
            with store.upload_path(upload_id).open("r+b") as handle:
                handle.truncate(offset)
                handle.seek(offset)
                handle.write(data)
            offset += len(data)
            db.execute("UPDATE uploads SET offset=?,created_at=? WHERE id=?", (offset, time.time(), upload_id))
        return {"offset": offset}

    @app.get("/api/uploads/{upload_id}")
    def upload_status(upload_id: str, request: Request):
        with store.connection() as db:
            row = store.upload(db, upload_id, session(request))
        return {"upload_id": upload_id, "offset": row["offset"], "status": row["status"]}

    @app.delete("/api/uploads/{upload_id}")
    def discard_upload(upload_id: str, request: Request):
        owner = session(request, mutate=True)
        with store.connection(write=True) as db:
            row = store.upload(db, upload_id, owner)
            if row["status"] != "uploading":
                raise HTTPException(409, "已提交的上传不能删除")
            store.upload_path(upload_id).unlink(missing_ok=True)
            db.execute("UPDATE uploads SET status='discarded' WHERE id=?", (upload_id,))
        return {"status": "discarded"}

    @app.post("/api/jobs", status_code=201)
    def submit_job(body: JobInput, request: Request):
        owner = session(request, mutate=True)
        if body.source_language == body.target_language:
            raise HTTPException(422, "源语言和目标语言应不同")
        selected = next(profile for profile in profile_catalog(settings) if profile["id"] == body.profile)
        if not selected["available"]:
            raise HTTPException(503, selected["reason"])
        with store.connection() as db:
            row = store.upload(db, body.upload_id, owner)
        if row["status"] != "uploading" or row["offset"] != row["size"]:
            raise HTTPException(409, "请先完成视频上传")
        with probe_slots:
            duration = probe_video(store.upload_path(body.upload_id), settings)
        free_space()
        job_id, now = secrets.token_hex(16), time.time()
        with store.connection(write=True) as db:
            row = store.upload(db, body.upload_id, owner)
            if row["status"] != "uploading":
                raise HTTPException(409, "该上传已提交")
            store.capacity(db, owner)
            db.execute("INSERT INTO jobs (id,owner,upload_id,source_language,target_language,profile,duration_mode,duration,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (job_id, owner, body.upload_id, body.source_language, body.target_language, body.profile, body.duration_mode, duration, now, now))
            db.execute("UPDATE uploads SET status='submitted' WHERE id=?", (body.upload_id,))
        worker.wakeup.set()
        return public_job(store.job(job_id, owner))

    @app.get("/api/jobs")
    def list_jobs(request: Request):
        owner = session(request)
        with store.connection() as db:
            rows = db.execute("SELECT j.*,u.filename FROM jobs j JOIN uploads u ON u.id=j.upload_id WHERE j.owner=? ORDER BY created_at DESC LIMIT 50", (owner,)).fetchall()
        return {"jobs": [public_job(row) for row in rows], "service": worker.status()}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        return public_job(store.job(job_id, session(request)))

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, request: Request):
        owner = session(request, mutate=True)
        await asyncio.to_thread(worker.cancel, job_id, owner)
        return public_job(store.job(job_id, owner))

    @app.get("/api/jobs/{job_id}/files/{filename}")
    def get_file(job_id: str, filename: str, request: Request):
        row = store.job(job_id, session(request))
        output = store.output(job_id)
        state = read_json(output / "run.json")
        result_name = result_video_name(row, state)
        if row["status"] != "succeeded" or filename not in ARTIFACTS | {result_name}:
            raise HTTPException(404, "文件不存在")
        path = output / filename
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "文件不存在")
        if filename == "report.json":
            report = read_json(path)
            models = (job_models(row["profile"], state) if state
                      else profile_catalog(settings)[0]["models"])
            public = {key: report[key] for key in ("status", "automatic", "human_review_required", "source_language", "target_language",
                                                   "translation_review", "time_compression_policy", "max_tempo", "delivery_policy", *STATS) if key in report}
            public.update(public_video_result(report, state))
            completeness = public_completeness(report)
            if completeness is not None:
                public["audio_completeness"] = completeness
            public.update(profile=row["profile"], duration_mode=row["duration_mode"], profile_label=PROFILES[row["profile"]]["label"],
                          models=models)
            public["video"] = f"/api/jobs/{job_id}/files/{result_name}"
            public["utterances"] = [{key: row[key] for key in ("id", "source_text", "target_text", "audio", "video",
                                                               "translatable_speech", "translation_review", "max_tempo", "time_compression") if key in row}
                                     for row in report.get("utterances", [])]
            for row, published in zip(report.get("utterances", []), public["utterances"]):
                reason = row.get("video_reason", "")
                if reason:
                    published["video_reason"] = reason if reason in {"no_target_match", "unbound_speaker", "missing_core_face_track"} else "mouth_edit_unavailable"
            return JSONResponse(public, headers={"Content-Disposition": 'attachment; filename="report.json"'})
        media_type = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
                      ".mkv": "video/x-matroska", ".avi": "video/x-msvideo", ".m4v": "video/mp4"}.get(path.suffix, "text/plain; charset=utf-8")
        video = filename == result_name or filename == "comparison.mp4"
        download_name = row["filename"] if filename == result_name and state.get("result_mode") == "passthrough_original" else filename
        return FileResponse(path, media_type=media_type, filename=download_name,
                            content_disposition_type="inline" if video else "attachment")

    @app.get("/api/jobs/{job_id}/source")
    def get_source(job_id: str, request: Request):
        row = store.job(job_id, session(request))
        media_type = {".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo"}.get(Path(row["filename"]).suffix.lower(), "video/mp4")
        return FileResponse(store.upload_path(row["upload_id"]), media_type=media_type,
                            filename=row["filename"], content_disposition_type="inline")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
