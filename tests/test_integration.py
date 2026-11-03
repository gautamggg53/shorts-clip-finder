"""End-to-end runs with real ffmpeg on synthetic media (no ML needed)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from conftest import requires_ffmpeg
from shorts_clip_finder.cli import main
from shorts_clip_finder.media import MediaError, probe

pytestmark = requires_ffmpeg
ARGS = ["--min", "10", "--max", "25", "-n", "2", "-q"]


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    audio, video = d / "talk.mp3", d / "talk.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=330:duration=95",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(audio),
    )
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x180:rate=25:duration=95",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=95",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-g",
        "50",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(video),
    )
    return {"audio": audio, "video": video}


def _duration(path) -> float:
    return probe(path).duration or 0.0


def test_probe(media):
    v, a = probe(media["video"]), probe(media["audio"])
    assert v.has_video and v.has_audio and (v.width, v.height) == (320, 180)
    assert a.has_audio and not a.has_video and a.duration == pytest.approx(95, abs=0.5)
    with pytest.raises(MediaError):
        probe(media["video"].parent / "missing.mp4")


def test_audio_exact_cut_end_to_end(tmp_path, fixtures, media):
    out = tmp_path / "out"
    rc = main(
        [
            str(media["audio"]),
            "-t",
            str(fixtures / "talk_whisper.json"),
            "-o",
            str(out),
            "--export-clips",
            "--cut-mode",
            "exact",
            *ARGS,
        ]
    )
    assert rc == 0
    data = json.loads((out / "clips.json").read_text())
    files = sorted((out / "clips").glob("*.m4a"))
    assert len(files) == len(data["clips"]) == 2
    for clip, f in zip(sorted(data["clips"], key=lambda c: c["rank"]), files):
        assert f.name.startswith(f"{clip['rank']:02d}-")
        assert _duration(f) == pytest.approx(clip["duration"], abs=0.1)
    assert len(list((out / "clips").glob("*.srt"))) == 2
    assert (out / "cut_clips.sh").stat().st_mode & 0o111


def test_video_copy_cut_and_script_reproduces(tmp_path, fixtures, media):
    out = tmp_path / "out"
    assert (
        main(
            [
                str(media["video"]),
                "-t",
                str(fixtures / "talk.srt"),
                "-o",
                str(out),
                "--export-clips",
                *ARGS,
            ]
        )
        == 0
    )
    cut = sorted((out / "clips").glob("*.mp4"))
    assert len(cut) == 2
    for f in cut:
        f.unlink()
    subprocess.run(["sh", str(out / "cut_clips.sh")], check=True)
    assert len(sorted((out / "clips").glob("*.mp4"))) == 2


def test_vertical_with_burned_captions(tmp_path, fixtures, media):
    out = tmp_path / "out"
    assert (
        main(
            [
                str(media["video"]),
                "-t",
                str(fixtures / "talk_whisper.json"),
                "-o",
                str(out),
                "--export-clips",
                "--vertical",
                "--burn-captions",
                *ARGS,
            ]
        )
        == 0
    )
    files = sorted((out / "clips").glob("*.mp4"))
    assert len(files) == 2
    info = probe(files[0])
    assert (info.width, info.height) == (1080, 1920)


def test_vertical_on_audio_is_an_error(tmp_path, fixtures, media):
    rc = main(
        [
            str(media["audio"]),
            "-t",
            str(fixtures / "talk.srt"),
            "-o",
            str(tmp_path),
            "--export-clips",
            "--vertical",
            *ARGS,
        ]
    )
    assert rc == 2


def test_media_without_whisper_extra_gives_clear_error(tmp_path, media, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    rc = main([str(media["audio"]), "-o", str(tmp_path)])
    assert rc == 2
    assert "pip install 'shorts-clip-finder[whisper]'" in capsys.readouterr().err


def test_module_entry_point(tmp_path, fixtures):
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "shorts_clip_finder",
            str(fixtures / "talk.srt"),
            "-o",
            str(tmp_path),
            *ARGS,
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
