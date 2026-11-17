"""Expose PyAV's FFmpeg ABI to TorchCodec without shadowing system libraries."""

import sys
from pathlib import Path

import av


def main():
    libraries = Path(av.__file__).resolve().parent.parent / "av.libs"
    destination = Path(sys.prefix) / "lib/ffmpeg"
    destination.mkdir(exist_ok=True)
    ffmpeg = {"libavcodec", "libavdevice", "libavfilter", "libavformat", "libavutil",
              "libpostproc", "libswresample", "libswscale"}
    for library in libraries.glob("*.so.*"):
        names = [library.name]
        name, version = library.name.split(".so.", 1)
        base = name.rsplit("-", 1)[0]
        if base in ffmpeg:
            names.append(f"{base}.so.{version.split('.')[0]}")
        for name in names:
            link = destination / name
            if not link.exists():
                link.symlink_to(library)


if __name__ == "__main__":
    main()
