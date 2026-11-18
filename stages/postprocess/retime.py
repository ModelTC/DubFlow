"""Retime isolated dubbed shots to restore the generated speech pace.

TTS is first fitted to the source timeline for mouth editing. At export, the
edited video and its mixed soundtrack are retimed together, in either direction.
The fixed-duration mode keeps the original timeline.
"""

import json
import math
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

MIN_RATIO = 1.001


def plan_adjustments(dialogue, fps, shots):
    """Select isolated, single-shot lines whose audio was fitted to a source slot."""
    if fps <= 0:
        raise ValueError("Video FPS must be positive")
    all_rows = dialogue["utterances"]
    rows = [row for row in all_rows if row["status"] == "translated"]
    spans = {}
    for row in all_rows:
        start, end = float(row["start"]), float(row["end"])
        if not 0 <= start < end or not all(map(math.isfinite, (start, end))):
            raise ValueError(f"Invalid translated interval: {row['id']}")
        spans[row["id"]] = (math.floor(start * fps), math.ceil(end * fps))

    adjustments, skipped = [], []
    for row in rows:
        ratio = float(row.get("tempo_ratio", 1.0))
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"Invalid speech tempo: {row['id']}")
        if 1 / MIN_RATIO <= ratio <= MIN_RATIO:
            continue
        # A short line is contracted only if speech.py deliberately slowed it
        # before mouth editing; never remove frames from a normal fixed dub.
        if ratio < 1 and not row.get("adaptive_shortening_prepared", False):
            continue
        first, last = spans[row["id"]]
        reason = None
        if last - first < 3:
            reason = "segment_too_short"
        elif not any(shot["start_frame"] <= first < last <= shot["end_frame"] for shot in shots):
            reason = "crosses_shot_boundary"
        elif any(other_id != row["id"] and first < other_last and other_first < last
                 for other_id, (other_first, other_last) in spans.items()):
            reason = "overlaps_other_speech"
        target = round((last - first) * ratio)
        if target < 3:
            reason = reason or "output_segment_too_short"
        if reason is None and target == last - first:
            continue
        if reason is not None:
            skipped.append({"utterance_id": row["id"], "reason": reason})
            continue
        adjustments.append({"utterance_id": row["id"], "start_frame": first, "end_frame": last,
                            "target_frames": target, "extra_frames": target - (last - first), "ratio": ratio})
    adjustments.sort(key=lambda item: item["start_frame"])
    return {"adjustments": adjustments,
            "extensions": [item for item in adjustments if item["extra_frames"] > 0],
            "contractions": [item for item in adjustments if item["extra_frames"] < 0],
            "skipped": skipped}


def media_info(path):
    command = ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,nb_frames,duration,r_frame_rate",
               "-of", "json", str(path)]
    streams = json.loads(subprocess.check_output(command, text=True))["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next((stream for stream in streams if stream["codec_type"] == "audio"), None)
    if audio is None or video.get("nb_frames", "N/A") == "N/A":
        raise ValueError(f"Expected an indexed video with audio: {path}")
    return int(video["nb_frames"]), float(video["duration"]), float(audio["duration"])


def segments_for(frames, adjustments):
    segments, cursor = [], 0
    for item in adjustments:
        if not cursor <= item["start_frame"] < item["end_frame"] <= frames:
            raise ValueError("Adaptive segments overlap or exceed the source video")
        if cursor < item["start_frame"]:
            segments.append((cursor, item["start_frame"], item["start_frame"] - cursor))
        segments.append((item["start_frame"], item["end_frame"], item["target_frames"]))
        cursor = item["end_frame"]
    if cursor < frames:
        segments.append((cursor, frames, frames - cursor))
    return segments


def atempo_chain(tempo):
    factors = []
    while tempo < .5:
        factors.append(.5)
        tempo *= 2
    while tempo > 2:
        factors.append(2.)
        tempo /= 2
    factors.append(tempo)
    return ",".join(f"atempo={factor:.12g}" for factor in factors)


def filter_graph(segments, fps):
    count = len(segments)
    graph = [f"[0:a]asplit={count}" + "".join(f"[ain{index}]" for index in range(count)),
             f"[0:v]split={count}" + "".join(f"[vin{index}]" for index in range(count))]
    for index, (first, last, target) in enumerate(segments):
        source_frames = last - first
        video = f"[vin{index}]trim=start_frame={first}:end_frame={last}"
        audio = (f"[ain{index}]atrim=start={first / fps:.9f}:end={last / fps:.9f},"
                 "asetpts=PTS-STARTPTS")
        if target != source_frames:
            ratio = target / source_frames
            if target > source_frames:
                # MCI needs future anchors; pad within this segment, never across a cut.
                video += ",tpad=stop_mode=clone:stop=2"
            video += f",setpts={ratio:.12g}*(PTS-STARTPTS),"
            video += (f"fps={fps}:round=near" if target < source_frames else
                      f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
            video += f",tpad=stop_mode=clone:stop_duration={2 / fps:.9f},trim=end_frame={target}"
            audio += f",{atempo_chain(1 / ratio)}"
        audio += f",apad,atrim=end={target / fps:.9f}"
        graph.append(video + f",setpts=N/({fps}*TB)[vout{index}]")
        graph.append(audio + f",asetpts=N/SR/TB[aout{index}]")
    # Exact audio duration also anchors single-frame gaps in concat.
    graph.append("".join(f"[vout{index}][aout{index}]" for index in range(count)) +
                 f"concat=n={count}:v=1:a=1[v][a]")
    return ";".join(graph)


def render(source, output, fps, frames, adjustments):
    target_frames = frames + sum(item["extra_frames"] for item in adjustments)
    if target_frames <= 0:
        raise ValueError("Adaptive video would contain no frames")
    temporary = output.with_name(f".{output.stem}-{uuid4().hex}.mp4")
    try:
        if adjustments:
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(source)]
            command.extend(["-filter_complex", filter_graph(segments_for(frames, adjustments), fps),
                       "-map", "[v]", "-map", "[a]", "-frames:v", str(target_frames),
                       "-r", str(fps), "-fps_mode", "cfr",
                       "-t", f"{target_frames / fps:.9f}", "-c:v", "libx264", "-preset", "fast",
                       "-crf", "18", "-threads", "8", "-pix_fmt", "yuv420p", "-c:a", "aac",
                       "-b:a", "192k", "-movflags", "+faststart", str(temporary)])
            subprocess.run(command, check=True)
        else:
            shutil.copyfile(source, temporary)
        actual_frames, video_seconds, audio_seconds = media_info(temporary)
        if (actual_frames != target_frames or abs(video_seconds - target_frames / fps) > 1 / fps or
                abs(audio_seconds - target_frames / fps) > 1 / fps + .03):
            raise ValueError(
                f"Adaptive output has inconsistent media duration: {temporary}; "
                f"actual frames={actual_frames}, video_seconds={video_seconds:.9f}, "
                f"audio_seconds={audio_seconds:.9f}; expected frames={target_frames}, "
                f"video_seconds={target_frames / fps:.9f}, audio_seconds={target_frames / fps:.9f}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return target_frames


def run_stage(*, source, plan, dialogue, output, manifest):
    source, output, manifest, plan_path = map(Path, (source, output, manifest, plan))
    if source == output or output.exists() or manifest.exists():
        raise FileExistsError(f"Adaptive output already exists: {output}")
    plan = json.loads(Path(plan_path).read_text())
    dialogue = json.loads(Path(dialogue).read_text())
    if Path(dialogue["source_plan"]).resolve() != plan_path.resolve():
        raise ValueError("Dialogue and video plan belong to different source timelines")
    fps, frames = plan["fps"], plan["frames"]
    actual_frames, _, _ = media_info(source)
    if actual_frames != frames:
        raise ValueError("Composed source video no longer matches the plan")
    selection = plan_adjustments(dialogue, fps, plan["shots"])
    output.parent.mkdir(parents=True, exist_ok=True)
    backend = "ffmpeg" if selection["extensions"] else "none"
    target_frames = render(source, output, fps, frames, selection["adjustments"])
    result = {"status": "complete", "mode": "adaptive_video", "source_video": str(source),
              "video": str(output), "source_frames": frames, "frames": target_frames,
              "fps": fps, "duration": target_frames / fps, **selection,
              "vfi_backend": backend,
              "video_interpolation_policy": "extensions_only_original_frame_contractions",
              "audio_treatment": "synchronous_segment_retime_including_background"}
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(manifest)
    print(json.dumps({"video": str(output), "frames": target_frames,
                      "extended_utterances": len(selection["extensions"]),
                      "contracted_utterances": len(selection["contractions"])}))
