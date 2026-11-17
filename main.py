"""视频翻译入口：准备 → 识别/人物绑定 → 分离 → 翻译配音 → 对口型 → 合成。"""

import argparse
import math
import shutil
from pathlib import Path

from runtime.config import ROOT, load_config
from runtime.job import Job, read_json, write_json
from runtime.languages import LANGUAGES
from stages.recognition.models import ASR_POLICY
from stages.speakers.bind import VISUAL_BINDING_POLICY, speaker_turn_merge_policy


def other_source_languages(audio_path, audio_result, target_language):
    """Read per-window ASR language evidence only from the same audio analysis."""
    windows_path = Path(audio_path).with_name("audio_asr.json")
    if not windows_path.is_file():
        return set()
    windows = read_json(windows_path)
    fingerprint = audio_result.get("audio_sha256")
    if not fingerprint or windows.get("audio_sha256") != fingerprint:
        return set()
    return {row["language"] for row in windows.get("asr_segments", [])
            if row.get("language") in LANGUAGES and row["language"] != target_language}


def run(args, execute=None):
    resident = execute is not None
    if args.target_language is None:
        raise ValueError("Specify --target-language")
    if args.target_language not in LANGUAGES:
        raise ValueError(f"Unsupported target language {args.target_language!r}; choose from {', '.join(LANGUAGES)}")
    config = load_config(args.config)
    duration_mode = getattr(args, "duration_mode", None) or config.get("duration_mode", "fixed")
    if duration_mode not in ("fixed", "adaptive_video"):
        raise ValueError("duration_mode must be fixed or adaptive_video")
    config["duration_mode"] = duration_mode
    if resident:
        config["execution_mode"] = "resident_gpu_v1"
    config["composition_policy"] = "continuous_track_roi_blend_v5"
    config["mouth_planning_policy"] = "supported_track_runs_landmark_fallback_v1"
    if duration_mode == "adaptive_video":
        config["retiming_policy"] = "paired_av_segments_v2"
    config["visual_binding_policy"] = VISUAL_BINDING_POLICY
    for key in ("gpu", "source_language", "target_language"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    requested_source_language = config["source_language"]
    if requested_source_language not in ("auto", *LANGUAGES):
        raise ValueError(f"Unsupported source language {requested_source_language!r}")
    if requested_source_language != "auto" and requested_source_language == config["target_language"]:
        raise ValueError("Source and target languages must differ")
    context_seconds = getattr(args, "context_seconds", None)
    if context_seconds is not None:
        if not math.isfinite(context_seconds) or context_seconds < 0:
            raise ValueError("--context-seconds must be a finite nonnegative number")
        config["context_seconds"] = context_seconds
        config["edit_context_seconds"] = min(config["edit_context_seconds"], context_seconds)
    transition_seconds = getattr(args, "transition_seconds", None)
    if transition_seconds is not None:
        if not math.isfinite(transition_seconds) or transition_seconds < 0:
            raise ValueError("--transition-seconds must be a finite nonnegative number")
        config["transition_seconds"] = transition_seconds
    if getattr(args, "merge_speaker_turns", None) is not None:
        config["merge_speaker_turns"] = args.merge_speaker_turns
    turn_merging = None
    if config.get("merge_speaker_turns", False):
        turn_merging = speaker_turn_merge_policy(config.get("speaker_turn_max_gap_seconds", .8),
                                                 config.get("speaker_turn_max_seconds", 15.))
        config["speaker_turn_merge_policy"] = turn_merging
    config["asr_recognition_policy"] = ASR_POLICY
    config["pipeline_recipe"] = "vllm_lightx2v_large_tv_loconet_v3"
    config["content_verification"] = "disabled"
    config["separation_recipe"] = "sam_large_tv_asr_anchors"
    config["separation_validation"] = "disabled"
    if not 0 <= config["edit_context_seconds"] <= config["context_seconds"] or config["transition_seconds"] < 0:
        raise ValueError("Require 0 <= edit context <= model context and nonnegative transitions")
    job = Job(args, config, execute)
    source, output = job.source, job.output
    state, state_path = job.state, job.state_path
    stage = job.stage
    # 保留虚拟环境的解释器入口；resolve 会跳到系统 Python，丢失该环境的包。
    video_python = (ROOT / config["video_python"]).absolute()
    audio_python = (ROOT / config["audio_python"]).absolute()
    vision_python = (ROOT / config["vision_python"]).absolute()
    identity_python = (ROOT / config["identity_python"]).absolute()
    separation_python = (ROOT / config["separation_python"]).absolute()
    s3fd_repo = (ROOT / config["s3fd_repo"]).resolve()
    asr_model = (ROOT / config["asr_model"]).resolve()
    diar_model = (ROOT / config["diar_model"]).resolve()
    identity_model = (ROOT / config["identity_model"]).resolve()
    state["stage_order"] = ([] if args.analysis_plan else [
        "prepare", "audio_nemotron3", "vision", "identity", "fusion"])
    if not args.source_audio_manifest:
        state["stage_order"].append("source_separation")
    if not args.dialogue:
        state["stage_order"].append("speech")
    state["stage_order"] += ["audio_mix", "target", "rois", "render", "compose"]
    if duration_mode == "adaptive_video":
        state["stage_order"].append("adaptive_duration")
    state["stage_order"].append("report")
    write_json(state_path, state)

    def separate(video, destination, duration, analysis_plan):
        manifest = destination / "source_audio.json"
        arguments = ["--source-video", video, "--duration", duration, "--output-dir", destination,
            "--analysis-plan", analysis_plan,
            "--model-dir", (ROOT / config["sam_audio_model"]).resolve(),
            "--text-encoder-dir", (ROOT / config["sam_audio_text_encoder"]).resolve(),
            "--description", config["sam_audio_description"],
            "--chunk-seconds", config["sam_audio_chunk_seconds"],
            "--overlap-seconds", config["sam_audio_overlap_seconds"]]
        stage("source_separation", separation_python, "stages.separate", arguments, manifest, gpu=True)
        return manifest

    def passthrough_if_same_language(audio_path):
        audio = read_json(audio_path)
        language = audio.get("language")
        if requested_source_language != "auto" or language != config["target_language"]:
            return False
        suffix = Path(getattr(args, "source_filename", None) or source.name).suffix.lower()
        if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            suffix = ".bin"
        final = output / f"final{suffix}"
        temporary = output / f"final{suffix}.tmp"
        shutil.copyfile(source, temporary)
        temporary.replace(final)
        other_languages = sorted(other_source_languages(audio_path, audio, language))
        write_json(output / "report.json", {
            "status": "complete", "automatic": True, "source_language": language,
            "target_language": language, "video_mode": "preserve_original",
            "delivery_policy": "original_unchanged",
            "translation_skipped": "auto_detected_same_as_target",
            "other_detected_languages": other_languages,
            "translated": 0, "preserved_original": 0, "lip_edited": 0,
            "partial_lip_edited": 0, "audio_only": 0,
        })
        if "report" not in state["completed"]:
            state["completed"].append("report")
        state["stage_order"] = list(state["completed"])
        state.update(status="complete", stage="report", result_mode="passthrough_original",
                     detected_source_language=language, final_video=str(final))
        state.pop("error", None)
        write_json(state_path, state)
        print(final)
        return True

    try:
        if args.analysis_plan:
            source_plan = Path(args.analysis_plan).resolve()
            plan = read_json(source_plan)
            media = read_json(Path(plan["source_video"]).parent / "media.json")
            if Path(media["source_video"]).resolve() != source:
                raise ValueError("Cached analysis belongs to another original video")
            if plan["models"]["audio"]["asr"].get("recognition_policy") != ASR_POLICY:
                raise ValueError("Cached analysis predates full-audio ASR; rerun without --analysis-plan")
            if plan.get("diagnostics", {}).get("visual_binding_policy") != VISUAL_BINDING_POLICY:
                raise ValueError("Cached analysis uses an older visual binding policy; rebuild the analysis plan")
            if passthrough_if_same_language(plan["selected_audio_result"]):
                return
        else:
            # 1. 准备素材；2. 识别对白、词时间与声学说话人。
            work = output / "work"
            stage("prepare", video_python, "stages.prepare", ["--video", source, "--work-dir", work,
                  "--max-side", config["max_side"]], work / "media.json")
            working_video, source_audio = work / "video_25fps.mp4", work / "audio_16k.wav"
            path = output / "audio_nemotron3.json"
            arguments = ["--audio", source_audio, "--output", path,
                         "--asr-model", asr_model, "--align-model", (ROOT / config["alignment_model"]).resolve(),
                         "--diar-model", diar_model, "--language", config["source_language"]]
            stage("audio_nemotron3", audio_python, "stages.recognition.run", arguments, path, gpu=True)
            if passthrough_if_same_language(path):
                return
            # 3. 检测画面人物，将台词绑定到可见轨迹。
            vision_arguments = ["--video", working_video, "--audio", source_audio,
                  "--output", output / "vision.json", "--work-dir", work / "vision", "--s3fd-repo", s3fd_repo,
                  "--s3fd-model", (ROOT / config["s3fd_model"]).resolve(),
                  "--loconet-model", (ROOT / config["loconet_model"]).resolve(),
                  "--detection-scale", config.get("detection_scale", 1.0),
                  "--confidence", config.get("detection_confidence", .5)]
            if config.get("grounding_model"):
                vision_arguments.extend(["--grounding-model", (ROOT / config["grounding_model"]).resolve()])
            stage("vision", vision_python, "stages.speakers.detect", vision_arguments, output / "vision.json", gpu=True)
            stage("identity", identity_python, "stages.speakers.identity", ["--video", working_video, "--vision", output / "vision.json",
                  "--model-dir", identity_model, "--output", output / "identity.json"], output / "identity.json")
            source_plan = output / "plan.json"
            fusion_arguments = ["--vision", output / "vision.json", "--identity", output / "identity.json",
                                "--audio", path, "--output", source_plan, "--mode", "speaker_visual"]
            if turn_merging:
                fusion_arguments.extend(["--merge-speaker-turns", "--speaker-turn-max-gap-seconds", turn_merging["max_gap_seconds"],
                                         "--speaker-turn-max-seconds", turn_merging["max_duration_seconds"]])
            stage("fusion", video_python, "stages.speakers.bind", fusion_arguments, source_plan)
            plan = read_json(source_plan)
        selected_audio = read_json(plan["selected_audio_result"])
        detected_source_language = selected_audio["language"]
        if detected_source_language not in LANGUAGES:
            raise ValueError(f"Detected source language {detected_source_language!r} is unsupported")
        if requested_source_language != "auto" and requested_source_language != detected_source_language:
            raise ValueError("Cached analysis source language differs from this request")
        if detected_source_language == config["target_language"]:
            raise ValueError("Detected source and target languages are the same")
        state["detected_source_language"] = detected_source_language
        write_json(state_path, state)
        actual_merging = plan.get("diagnostics", {}).get("speaker_turn_merging")
        actual_policy = {key: actual_merging.get(key) for key in speaker_turn_merge_policy()} if actual_merging else None
        if actual_policy != turn_merging:
            raise ValueError("Analysis uses a different speaker-turn merge policy; build a new plan and dialogue")
        duration = plan["frames"] / plan["fps"]
        speech_python = (ROOT / config["speech_python"]).absolute()
        # 4. 分离原片对白和背景。
        source_audio_manifest = Path(args.source_audio_manifest).resolve() if args.source_audio_manifest else output / "audio/source_audio.json"
        if not args.source_audio_manifest:
            separate(source, output / "audio", duration, source_plan)
        separated = read_json(source_audio_manifest)
        if Path(separated["source_video"]).resolve() != source or abs(separated["duration"] - duration) > 1 / 48000:
            raise ValueError("Source speech/background cache belongs to another video timeline")
        state["selected_separation_backend"] = separated["metadata"].get(
            "separation_backend", "external" if args.source_audio_manifest else config["separation_backend"])
        state["selected_source_audio_manifest"] = str(source_audio_manifest)
        state["selected_separation_model"] = separated["metadata"]["separation_model"]
        write_json(state_path, state)
        # 5. 翻译与配音，保留同一条对白 ID。
        dialogue_path = Path(args.dialogue).resolve() if args.dialogue else output / "speech/dialogue.json"
        if not args.dialogue:
            speech_arguments = ["--plan", source_plan, "--speech-source", separated["source_speech_path"],
                  "--output-dir", output / "speech", "--source-language", detected_source_language,
                  "--target-language", config["target_language"], "--duration-mode", duration_mode]
            for key in ("translation_url", "translation_model", "tts_model", "tts_python", "qwen_x_vector_only_mode"):
                if key in config:
                    if key == "qwen_x_vector_only_mode":
                        speech_arguments.append("--qwen-x-vector-only-mode" if config[key] else "--no-qwen-x-vector-only-mode")
                    else:
                        speech_arguments.extend(["--" + key.replace("_", "-"), str(config[key])])
            stage("speech", speech_python, "stages.speech.run", speech_arguments, dialogue_path, gpu=True)
        state["selected_dialogue_path"] = str(dialogue_path)
        write_json(state_path, state)
        incomplete = [row["id"] for row in read_json(dialogue_path)["utterances"]
                      if row.get("human_speech") and row["status"] != "translated"]
        if incomplete:
            raise RuntimeError(f"Dialogue translation incomplete; refusing an untranslated soundtrack: {incomplete}")
        # 6. 混音并规划、裁剪、生成口型。
        stage("audio_mix", audio_python, "stages.output.audio", ["--source-manifest", source_audio_manifest,
              "--dialogue", dialogue_path, "--duration", duration,
              "--output-dir", output / "audio"], output / "audio/audio_mix.json")
        required = {row["id"] for row in read_json(dialogue_path)["utterances"] if row.get("human_speech")}
        accepted = set(read_json(output / "audio/audio_mix.json")["metadata"]["accepted_replacement_ids"])
        if required - accepted:
            raise RuntimeError(f"Mix did not replace every dialogue: {sorted(required - accepted)}")
        stage("target", audio_python, "stages.lipsync.plan", ["--plan", source_plan, "--dialogue", dialogue_path,
              "--audio-manifest", output / "audio/audio_mix.json",
              "--context-seconds", config["context_seconds"], "--edit-context-seconds", config["edit_context_seconds"],
              "--output-dir", output], output / "render_plan.json")
        composed_video = output / ("final_fixed.mp4" if duration_mode == "adaptive_video" else "final.mp4")
        arguments = ["--plan", output / "native_plan.json", "--manifest", output / "render/render_manifest.json",
                     "--output", composed_video, "--transition-seconds", config["transition_seconds"]]
        if config["comparison"] and duration_mode == "fixed":
            arguments.extend(["--comparison", output / "comparison.mp4"])
        stage("rois", video_python, "stages.lipsync.crop", ["--plan", output / "render_plan.json",
              "--landmarker", (ROOT / config["landmarker_model"]).resolve(), "--output", output / "native_plan.json"], output / "native_plan.json")
        render_arguments = ["--plan", output / "native_plan.json", "--output-dir", output / "render",
                            "--server-url", config["tbdub_url"]]
        stage("render", video_python, "stages.lipsync.render", render_arguments, output / "render/render_manifest.json")
        # 7. 回贴原画面、调整时长并导出。
        stage("compose", video_python, "stages.output.compose", arguments, output / "compose.json")
        if duration_mode == "adaptive_video":
            stage("adaptive_duration", video_python, "stages.output.retime",
                  ["--source", composed_video, "--plan", source_plan, "--dialogue", dialogue_path,
                   "--output", output / "final.mp4", "--manifest", output / "adaptive_duration.json"], output / "adaptive_duration.json",
                  gpu=False)
            if not (output / "final.mp4").is_file():
                raise FileNotFoundError(output / "final.mp4")
        stage("report", video_python, "stages.output.report", ["--run-dir", output], output / "report.json")
    except BaseException as error:
        state.update(status="failed", error=str(error))
        write_json(state_path, state)
        raise
    state.pop("error", None)
    state.update(status="complete", final_video=str(output / "final.mp4"))
    write_json(state_path, state)
    print(state["final_video"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", required=True, help="原始视频，包含原语言对白")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(ROOT / "config.example.json"))
    parser.add_argument("--merge-speaker-turns", action=argparse.BooleanOptionalAction, default=None,
                        help="合并同一人连续发言；不跨换人、重叠或切镜，默认沿用配置")
    parser.add_argument("--context-seconds", type=float, help="覆盖 TBDub 输入视频前后上下文秒数；贴回上下文不会超过此值")
    parser.add_argument("--transition-seconds", type=float, help="覆盖原片与 TBDub 画面的贴回淡变宽度（秒）")
    parser.add_argument("--analysis-plan", help="复用相同原片的说话人/人脸分析计划")
    parser.add_argument("--source-audio-manifest", help="复用相同原片分离出的对白和背景")
    parser.add_argument("--dialogue", help="复用相同分析计划已完成的翻译/配音结果")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", type=int, help="覆盖配置中的单卡编号")
    parser.add_argument("--source-language", choices=("auto", *LANGUAGES))
    parser.add_argument("--target-language", choices=tuple(LANGUAGES), required=True,
                        help="必填：输出对白语言；新增语种需要 Qwen3-TTS")
    parser.add_argument("--duration-mode", choices=("fixed", "adaptive_video"), default=None,
                        help="fixed 保持原片时长；adaptive_video 按译音长短调整成片时长")
    run(parser.parse_args())
