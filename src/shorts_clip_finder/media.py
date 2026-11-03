"""ffmpeg/ffprobe helpers. Requires ``ffmpeg`` and ``ffprobe`` on PATH."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    pass


def require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise MediaError(f"{tool} not found on PATH; install ffmpeg (https://ffmpeg.org/)")
    return path


@dataclass(frozen=True)
class MediaInfo:
    duration: float | None
    has_video: bool
    has_audio: bool
    width: int | None = None
    height: int | None = None


def probe(path: str | Path) -> MediaInfo:
    cmd = [
        require("ffprobe"),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise MediaError(f"ffprobe failed on {path}: {res.stderr.strip()}")
    data = json.loads(res.stdout or "{}")
    streams = data.get("streams", [])
    video = [
        s
        for s in streams
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
    ]
    dur = data.get("format", {}).get("duration")
    return MediaInfo(
        duration=float(dur) if dur else None,
        has_video=bool(video),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        width=video[0].get("width") if video else None,
        height=video[0].get("height") if video else None,
    )


def decode_audio(path: str | Path, sample_rate: int = 16000):
    """Decode any ffmpeg-readable file to mono float32 PCM (a NumPy array).

    Decoding with the ffmpeg binary avoids PyAV version mismatches inside faster-whisper.
    """
    import numpy as np

    cmd = [
        require("ffmpeg"),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise MediaError(f"ffmpeg could not decode {path}: {res.stderr.decode(errors='replace')}")
    return np.frombuffer(res.stdout, dtype=np.float32)
