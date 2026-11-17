"""统一为 25 FPS 时间轴，保留播放速度和原始宽高比。"""

import argparse
import json
import subprocess
from pathlib import Path


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ]))


def prepare_media(video, work_dir, max_side=0):
    source = Path(video).resolve()
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    prepared, audio, metadata = work / "video_25fps.mp4", work / "audio_16k.wav", work / "media.json"
    signature = {"source": str(source), "size": source.stat().st_size,
                 "mtime_ns": source.stat().st_mtime_ns, "max_side": max_side}
    request_path = work / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != signature:
            raise ValueError("Working directory belongs to another video/configuration")
        if metadata.exists():
            return json.loads(metadata.read_text())
    else:
        if any(path.exists() for path in (prepared, audio, metadata)):
            raise FileExistsError("Unknown existing working media")
        temporary = work / "request.tmp"
        temporary.write_text(json.dumps(signature))
        temporary.replace(request_path)
    source_info = probe(source)
    filters = ["fps=25"]
    if max_side:
        stream = next(item for item in source_info["streams"] if item["codec_type"] == "video")
        scale = min(1, max_side / max(stream["width"], stream["height"]))
        width = max(2, int(stream["width"] * scale) // 2 * 2)
        height = max(2, int(stream["height"] * scale) // 2 * 2)
        filters.append(f"scale={width}:{height}")
    if not prepared.exists():
        temporary = work / "video.partial.mp4"
        subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0", "-vf", ",".join(filters),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(temporary),
        ], check=True)
        temporary.replace(prepared)
    if not audio.exists():
        temporary = work / "audio.partial.wav"
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source), "-map", "0:a:0",
            "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(temporary),
        ], check=True)
        temporary.replace(audio)
    result = {"source_video": str(source), "working_video": str(prepared), "analysis_audio": str(audio),
              "source": source_info, "working": probe(prepared), "time_origin_seconds": 0}
    temporary = work / "media.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(metadata)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-side", type=int, default=0, help="可选缩小最长边；0 保持原始分辨率")
    args = parser.parse_args()
    if args.max_side < 0:
        parser.error("--max-side 不能为负数")
    result = prepare_media(args.video, args.work_dir, args.max_side)
    print(json.dumps({"video": result["working_video"], "audio": result["analysis_audio"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
