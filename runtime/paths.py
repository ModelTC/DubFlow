"""Locations of job inputs and artifacts, shared by CLI and web workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobInputs:
    source: Path
    analysis_plan: Path | None = None
    source_audio_manifest: Path | None = None
    dialogue: Path | None = None

    @classmethod
    def from_args(cls, args):
        return cls(Path(args.source).resolve(),
                   Path(args.analysis_plan).resolve() if args.analysis_plan else None,
                   Path(args.source_audio_manifest).resolve() if args.source_audio_manifest else None,
                   Path(args.dialogue).resolve() if args.dialogue else None)


@dataclass(frozen=True)
class JobPaths:
    root: Path

    @property
    def state(self):
        return self.root / "run.json"

    @property
    def logs(self):
        return self.root / "logs"

    @property
    def work(self):
        return self.root / "work"

    @property
    def media(self):
        return self.work / "media.json"

    @property
    def working_video(self):
        return self.work / "video_25fps.mp4"

    @property
    def source_audio(self):
        return self.work / "audio_16k.wav"

    @property
    def recognition(self):
        return self.root / "audio_nemotron3.json"

    @property
    def vision_work(self):
        return self.work / "vision"

    @property
    def vision(self):
        return self.root / "vision.json"

    @property
    def identity(self):
        return self.root / "identity.json"

    @property
    def plan(self):
        return self.root / "plan.json"

    @property
    def audio(self):
        return self.root / "audio"

    @property
    def source_audio_manifest(self):
        return self.audio / "source_audio.json"

    @property
    def audio_mix(self):
        return self.audio / "audio_mix.json"

    @property
    def speech(self):
        return self.root / "speech"

    @property
    def dialogue(self):
        return self.speech / "dialogue.json"

    @property
    def render_plan(self):
        return self.root / "render_plan.json"

    @property
    def native_plan(self):
        return self.root / "native_plan.json"

    @property
    def render(self):
        return self.root / "render"

    @property
    def render_manifest(self):
        return self.render / "render_manifest.json"

    @property
    def final(self):
        return self.root / "final.mp4"

    @property
    def fixed_video(self):
        return self.root / "final_fixed.mp4"

    @property
    def compose(self):
        return self.root / "compose.json"

    @property
    def comparison(self):
        return self.root / "comparison.mp4"

    @property
    def adaptation(self):
        return self.root / "adaptive_duration.json"

    @property
    def report(self):
        return self.root / "report.json"

    def original_copy(self, filename):
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            suffix = ".bin"
        return self.root / f"final{suffix}"

    @staticmethod
    def cached_media(plan):
        return Path(plan["source_video"]).parent / "media.json"

    @staticmethod
    def asr_windows(audio_result):
        return Path(audio_result).with_name("audio_asr.json")
