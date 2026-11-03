import pytest

from shorts_clip_finder import FinderConfig, find_clips, load_transcript
from shorts_clip_finder.cutter import CutOptions, build_command, cut_script, plan_cuts
from shorts_clip_finder.media import MediaError, MediaInfo

VIDEO = MediaInfo(duration=100, has_video=True, has_audio=True, width=1920, height=1080)
AUDIO = MediaInfo(duration=100, has_video=False, has_audio=True)


@pytest.fixture
def clip(fixtures):
    t = load_transcript(fixtures / "talk_whisper.json")
    return find_clips(t, FinderConfig(min_seconds=10, max_seconds=25, top=1))[0]


def test_copy_mode(tmp_path, clip):
    argv = build_command(tmp_path / "in.mp4", clip, "out.mp4", VIDEO, CutOptions())
    assert argv[:2] == ["ffmpeg", "-nostdin"]
    assert argv[argv.index("-ss") + 1] == f"{clip.start:.3f}"
    assert argv[argv.index("-t") + 1] == f"{clip.duration:.3f}"
    assert argv.index("-ss") < argv.index("-i")
    assert "copy" in argv and "libx264" not in argv and argv[-1] == "out.mp4"


def test_exact_vertical_captions(tmp_path, clip):
    opts = CutOptions(mode="copy", vertical=True, burn_captions=True)
    assert opts.reencode
    argv = build_command(tmp_path / "in.mp4", clip, "o.mp4", VIDEO, opts, srt_name="o.srt")
    vf = argv[argv.index("-vf") + 1]
    assert vf.startswith("scale=1080:1920") and "crop=1080:1920" in vf
    assert "subtitles=o.srt:force_style=" in vf
    assert "libx264" in argv and "aac" in argv


def test_audio_only(tmp_path, clip):
    argv = build_command(tmp_path / "in.mp3", clip, "o.m4a", AUDIO, CutOptions(mode="exact"))
    assert "-vn" in argv and "libx264" not in argv
    with pytest.raises(MediaError):
        build_command(tmp_path / "in.mp3", clip, "o.mp4", AUDIO, CutOptions(vertical=True))


def test_captions_need_srt(tmp_path, clip):
    with pytest.raises(ValueError):
        build_command(tmp_path / "i.mp4", clip, "o.mp4", VIDEO, CutOptions(burn_captions=True))


def test_plan_names_and_script(tmp_path, clip):
    src = tmp_path / "my talk.mov"
    plans = plan_cuts(src, [clip], VIDEO, CutOptions())
    assert plans[0].output.startswith("01-") and plans[0].output.endswith(".mov")
    assert plans[0].srt.endswith(".srt")
    exact = plan_cuts(src, [clip], VIDEO, CutOptions(mode="exact"))
    assert exact[0].output.endswith(".mp4")
    assert plan_cuts(tmp_path / "a.mp3", [clip], AUDIO, CutOptions("exact"))[0].output.endswith(
        ".m4a"
    )
    script = cut_script(plans)
    assert script.startswith("#!/bin/sh") and "'" + str(src.resolve()) + "'" in script
