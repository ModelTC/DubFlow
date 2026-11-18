"""Typed stage orchestration shared by command-line and resident workers."""

import math
import shutil
from pathlib import Path

from runtime.config import load_config
from runtime.execution import FunctionCall, serializable
from runtime.job import Job
from runtime.languages import LANGUAGES
from runtime.paths import JobPaths
from runtime.utils import read_json, write_json
from stages.recognition.model import ASR_POLICY
from stages.speakers.bind import VISUAL_BINDING_POLICY, speaker_turn_merge_policy


def configure(args, resident):
    config = load_config(args.config)
    for key in ("gpu", "source_language", "target_language", "duration_mode", "merge_speaker_turns"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if not config.get("target_language") or config["target_language"] not in LANGUAGES:
        raise ValueError("Specify a supported target language")
    source = config["source_language"]
    if source not in ("auto", *LANGUAGES):
        raise ValueError(f"Unsupported source language {source!r}")
    if source != "auto" and source == config["target_language"]:
        raise ValueError("Source and target languages must differ")
    config.setdefault("duration_mode", "fixed")
    if config["duration_mode"] not in ("fixed", "adaptive_video"):
        raise ValueError("duration_mode must be fixed or adaptive_video")
    for key in ("context_seconds", "transition_seconds"):
        value = getattr(args, key, None)
        if value is not None:
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
            config[key] = value
    if getattr(args, "context_seconds", None) is not None:
        config["edit_context_seconds"] = min(config["edit_context_seconds"], config["context_seconds"])
    if not 0 <= config["edit_context_seconds"] <= config["context_seconds"] or config["transition_seconds"] < 0:
        raise ValueError("Require 0 <= edit context <= model context and nonnegative transitions")
    if config.get("merge_speaker_turns", False):
        config["speaker_turn_merge_policy"] = speaker_turn_merge_policy(
            config.get("speaker_turn_max_gap_seconds", .8), config.get("speaker_turn_max_seconds", 15.))
    if resident:
        config["execution_mode"] = "resident_gpu_v1"
    config.update(composition_policy="continuous_track_roi_blend_v5",
                  mouth_planning_policy="supported_track_runs_landmark_fallback_v1",
                  visual_binding_policy=VISUAL_BINDING_POLICY, asr_recognition_policy=ASR_POLICY,
                  pipeline_recipe=f"{config['translation_backend']}_lightx2v_large_tv_loconet_v3",
                  content_verification="disabled", separation_recipe="sam_large_tv_asr_anchors",
                  separation_validation="disabled")
    if config["duration_mode"] == "adaptive_video":
        config["retiming_policy"] = "paired_av_segments_v2"
    return config


def other_source_languages(audio_path, audio_result, target_language):
    windows_path = JobPaths.asr_windows(audio_path)
    if not windows_path.is_file():
        return set()
    windows = read_json(windows_path)
    fingerprint = audio_result.get("audio_sha256")
    if not fingerprint or windows.get("audio_sha256") != fingerprint:
        return set()
    return {row["language"] for row in windows.get("asr_segments", [])
            if row.get("language") in LANGUAGES and row["language"] != target_language}


class Pipeline:
    def __init__(self, args, execute=None):
        self.args = args
        self.config = configure(args, execute is not None)
        self.job = Job(args, self.config, execute)
        self.inputs, self.paths = self.job.inputs, self.job.paths
        self.state = self.job.state
        self.duration_mode = self.config["duration_mode"]
        self.merge_policy = self.config.get("speaker_turn_merge_policy") if self.config.get("merge_speaker_turns") else None
        self.state["stage_order"] = [] if self.inputs.analysis_plan else [
            "prepare", "audio_nemotron3", "vision", "identity", "fusion"]
        if not self.inputs.source_audio_manifest:
            self.state["stage_order"].append("source_separation")
        if not self.inputs.dialogue:
            self.state["stage_order"].append("speech")
        self.state["stage_order"] += ["audio_mix", "target", "rois", "render", "compose"]
        if self.duration_mode == "adaptive_video":
            self.state["stage_order"].append("adaptive_duration")
        self.state["stage_order"].append("report")
        self.save_state()

    def save_state(self):
        write_json(self.paths.state, self.state)

    def stage(self, name, role, module, expected, *, gpu=False, **parameters):
        call = FunctionCall(module, "run_stage", serializable(parameters))
        self.job.stage(name, self.config[f"{role}_python"], call, expected, gpu=gpu)

    def preserve_same_language(self, audio_path):
        audio = read_json(audio_path)
        language = audio.get("language")
        if self.config["source_language"] != "auto" or language != self.config["target_language"]:
            return False
        final = self.paths.original_copy(getattr(self.args, "source_filename", None) or self.inputs.source.name)
        temporary = final.with_suffix(final.suffix + ".tmp")
        shutil.copyfile(self.inputs.source, temporary)
        temporary.replace(final)
        write_json(self.paths.report, {
            "status": "complete", "automatic": True, "source_language": language,
            "target_language": language, "video_mode": "preserve_original", "delivery_policy": "original_unchanged",
            "translation_skipped": "auto_detected_same_as_target",
            "other_detected_languages": sorted(other_source_languages(audio_path, audio, language)),
            "translated": 0, "preserved_original": 0, "lip_edited": 0, "partial_lip_edited": 0, "audio_only": 0,
        })
        if "report" not in self.state["completed"]:
            self.state["completed"].append("report")
        self.state["stage_order"] = list(self.state["completed"])
        self.state.update(status="complete", stage="report", result_mode="passthrough_original",
                          detected_source_language=language, final_video=str(final))
        self.state.pop("error", None)
        self.save_state()
        print(final)
        return True

    def analyze(self):
        c, p = self.config, self.paths
        if self.inputs.analysis_plan:
            plan = read_json(self.inputs.analysis_plan)
            media = read_json(p.cached_media(plan))
            if Path(media["source_video"]).resolve() != self.inputs.source:
                raise ValueError("Cached analysis belongs to another original video")
            if plan["models"]["audio"]["asr"].get("recognition_policy") != ASR_POLICY:
                raise ValueError("Cached analysis predates full-audio ASR; rebuild the analysis plan")
            if plan.get("diagnostics", {}).get("visual_binding_policy") != VISUAL_BINDING_POLICY:
                raise ValueError("Cached analysis uses an older visual binding policy; rebuild the analysis plan")
            return None if self.preserve_same_language(plan["selected_audio_result"]) else plan
        self.stage("prepare", "video", "stages.prepare.run", p.media,
                   video=self.inputs.source,
                   work_dir=p.work, max_side=c["max_side"])
        self.stage("audio_nemotron3", "audio", "stages.recognition.run", p.recognition, gpu=True,
                   audio=p.source_audio, output=p.recognition, asr_model=c["asr_model"],
                   align_model=c["alignment_model"], diar_model=c["diar_model"], language=c["source_language"])
        if self.preserve_same_language(p.recognition):
            return None
        self.stage("vision", "vision", "stages.speakers.run", p.vision, gpu=True,
                   video=p.working_video, audio=p.source_audio, output=p.vision, work_dir=p.vision_work,
                   s3fd_model=c["s3fd_model"], loconet_model=c["loconet_model"],
                   detection_scale=c.get("detection_scale", 1.), confidence=c.get("detection_confidence", .5),
                   grounding_model=c.get("grounding_model"))
        self.stage("identity", "identity", "stages.speakers.identity", p.identity,
                   video=p.working_video, vision=p.vision, model_dir=c["identity_model"], output=p.identity)
        self.stage("fusion", "video", "stages.speakers.bind", p.plan,
                   vision=p.vision, identity=p.identity, audio=[p.recognition], output=p.plan, mode="speaker_visual",
                   merge_speaker_turns=bool(self.merge_policy),
                   speaker_turn_max_gap_seconds=self.merge_policy["max_gap_seconds"] if self.merge_policy else .8,
                   speaker_turn_max_seconds=self.merge_policy["max_duration_seconds"] if self.merge_policy else 15.)
        return read_json(p.plan)

    def validate_analysis(self, plan):
        selected_audio = read_json(plan["selected_audio_result"])
        language = selected_audio["language"]
        if language not in LANGUAGES:
            raise ValueError(f"Detected source language {language!r} is unsupported")
        requested = self.config["source_language"]
        if requested != "auto" and requested != language:
            raise ValueError("Cached analysis source language differs from this request")
        if language == self.config["target_language"]:
            raise ValueError("Detected source and target languages are the same")
        self.state["detected_source_language"] = language
        self.save_state()
        actual = plan.get("diagnostics", {}).get("speaker_turn_merging")
        policy = {key: actual.get(key) for key in speaker_turn_merge_policy()} if actual else None
        if policy != self.merge_policy:
            raise ValueError("Analysis uses a different speaker-turn merge policy; build a new plan and dialogue")
        return language

    def separate(self, source_plan, duration):
        c, p = self.config, self.paths
        manifest = self.inputs.source_audio_manifest or p.source_audio_manifest
        if not self.inputs.source_audio_manifest:
            self.stage("source_separation", "separation", "stages.separation.run", manifest, gpu=True,
                       source_video=self.inputs.source, duration=duration, analysis_plan=source_plan,
                       output_dir=p.audio, model_dir=c["sam_audio_model"], text_encoder_dir=c["sam_audio_text_encoder"],
                       description=c["sam_audio_description"], chunk_seconds=c["sam_audio_chunk_seconds"],
                       overlap_seconds=c["sam_audio_overlap_seconds"])
        separated = read_json(manifest)
        if Path(separated["source_video"]).resolve() != self.inputs.source or abs(separated["duration"] - duration) > 1 / 48000:
            raise ValueError("Source speech/background cache belongs to another video timeline")
        self.state.update(selected_separation_backend=separated["metadata"].get("separation_backend", "external"),
                          selected_source_audio_manifest=str(manifest),
                          selected_separation_model=separated["metadata"]["separation_model"])
        self.save_state()
        return manifest, separated

    def speak(self, source_plan, separated, language):
        c, p = self.config, self.paths
        dialogue = self.inputs.dialogue or p.dialogue
        if not self.inputs.dialogue:
            backend_keys = ("translation_url", "translation_model", "translation_backend", "translation_api_key_env",
                            "translation_options",
                            "tts_backend", "tts_model", "tts_python", "tts_url", "tts_api_key_env", "tts_voice",
                            "qwen_x_vector_only_mode")
            self.stage("speech", "speech", "stages.speech.run", dialogue, gpu=True,
                       plan=source_plan, speech_source=separated["source_speech_path"], output_dir=p.speech,
                       source_language=language, target_language=c["target_language"], duration_mode=self.duration_mode,
                       **{key: c[key] for key in backend_keys if key in c})
        self.state["selected_dialogue_path"] = str(dialogue)
        self.save_state()
        rows = read_json(dialogue)["utterances"]
        incomplete = [row["id"] for row in rows if row.get("human_speech") and row["status"] != "translated"]
        if incomplete:
            raise RuntimeError(f"Dialogue translation incomplete; refusing an untranslated soundtrack: {incomplete}")
        return dialogue, rows

    def render(self, source_plan, manifest, dialogue, rows, duration):
        c, p = self.config, self.paths
        self.stage("audio_mix", "audio", "stages.postprocess.audio", p.audio_mix,
                   source_manifest=manifest, dialogue=dialogue, duration=duration, output_dir=p.audio)
        required = {row["id"] for row in rows if row.get("human_speech")}
        accepted = set(read_json(p.audio_mix)["metadata"]["accepted_replacement_ids"])
        if required - accepted:
            raise RuntimeError(f"Mix did not replace every dialogue: {sorted(required - accepted)}")
        self.stage("target", "audio", "stages.lipsync.plan", p.render_plan,
                   plan=source_plan, dialogue=dialogue, audio_manifest=p.audio_mix, output_dir=p.root,
                   context_seconds=c["context_seconds"], edit_context_seconds=c["edit_context_seconds"])
        self.stage("rois", "video", "stages.lipsync.crop", p.native_plan,
                   plan=p.render_plan, landmarker=c["landmarker_model"], output=p.native_plan)
        self.stage("render", "video", "stages.lipsync.run", p.render_manifest,
                   plan=p.native_plan, output_dir=p.render, server_url=c["tbdub_url"])

    def compose(self, source_plan, dialogue):
        c, p = self.config, self.paths
        adaptive = self.duration_mode == "adaptive_video"
        video = p.fixed_video if adaptive else p.final
        self.stage("compose", "video", "stages.postprocess.run", p.compose,
                   plan=p.native_plan, manifest=p.render_manifest, output=video,
                   comparison=p.comparison if c["comparison"] and not adaptive else None,
                   transition_seconds=c["transition_seconds"])
        if adaptive:
            self.stage("adaptive_duration", "video", "stages.postprocess.retime", p.adaptation,
                       source=video, plan=source_plan, dialogue=dialogue, output=p.final, manifest=p.adaptation)
            if not p.final.is_file():
                raise FileNotFoundError(p.final)
        self.stage("report", "video", "stages.postprocess.report", p.report, run_dir=p.root)

    def run(self):
        try:
            plan = self.analyze()
            if plan is None:
                return
            language = self.validate_analysis(plan)
            source_plan = self.inputs.analysis_plan or self.paths.plan
            duration = plan["frames"] / plan["fps"]
            manifest, separated = self.separate(source_plan, duration)
            dialogue, rows = self.speak(source_plan, separated, language)
            self.render(source_plan, manifest, dialogue, rows, duration)
            self.compose(source_plan, dialogue)
        except BaseException as error:
            self.state.update(status="failed", error=str(error))
            self.save_state()
            raise
        self.state.pop("error", None)
        self.state.update(status="complete", final_video=str(self.paths.final))
        self.save_state()
        print(self.state["final_video"])


def run(args, execute=None):
    return Pipeline(args, execute).run()
