import csv
import io
import itertools
import json
from fractions import Fraction

import pytest

from shorts_clip_finder import FinderConfig, find_clips, load_transcript
from shorts_clip_finder.export import (
    caption_cues,
    clip_basename,
    clip_srt,
    frame_rate,
    frames_timecode,
    slugify,
    timecode,
    to_csv,
    to_edl,
    to_json,
    to_markdown,
    write_reports,
)
from shorts_clip_finder.models import Word
from shorts_clip_finder.transcripts import parse_srt


@pytest.fixture
def clips(fixtures):
    t = load_transcript(fixtures / "talk_whisper.json")
    return find_clips(t, FinderConfig(min_seconds=10, max_seconds=25, top=3))


META = {"source": "talk.mp4", "min_seconds": 10, "max_seconds": 25}


def test_timecodes():
    assert timecode(3723.4567) == "01:02:03.457"
    assert timecode(1.5, ",") == "00:00:01,500"
    assert timecode(-1) == "00:00:00.000"
    assert frames_timecode(61.5, 30) == "00:01:01:15"
    assert frames_timecode(3600, 25) == "01:00:00:00"


def test_frame_rate_maps_ntsc_decimals_to_exact_rates():
    assert frame_rate(29.97) == Fraction(30000, 1001)
    assert frame_rate("29.97") == Fraction(30000, 1001)
    assert frame_rate("23.976") == Fraction(24000, 1001)
    assert frame_rate("59.94") == Fraction(60000, 1001)
    assert frame_rate("30000/1001") == Fraction(30000, 1001)
    assert frame_rate(25) == 25 and frame_rate("30") == 30 and frame_rate("12.5") == Fraction(25, 2)
    for bad in ("0", "-24", "abc", "1/0", ""):
        with pytest.raises(ValueError):
            frame_rate(bad)


def test_ntsc_timecode_does_not_drift():
    # 30 minutes of 29.97 footage is 53946 frames, labelled 00:29:58:06 in non-drop
    # timecode. Treating it as 30 fps would put the cut 1.8 s late.
    assert frames_timecode(1800, "29.97") == "00:29:58:06"
    assert frames_timecode(1800, 30) == "00:30:00:00"
    assert frames_timecode(3600, "23.976") == "00:59:56:10"


def test_edl_uses_real_frame_rate(clips):
    rate = Fraction(30000, 1001)
    edl = to_edl(clips, "talk.mp4", fps="29.97")
    events = [ln.split() for ln in edl.splitlines() if ln[:3].isdigit()]
    total = round(Fraction(clips[0].start) * rate)
    assert events[0][4] == f"00:{total // 30 // 60:02d}:{total // 30 % 60:02d}:{total % 30:02d}"
    for prev, nxt in itertools.pairwise(events):
        assert prev[7] == nxt[6]


def test_slugify():
    assert slugify("Why Do Budgets Fail? (Part 2)") == "why-do-budgets-fail-part-2"
    assert slugify("???") == "clip"
    assert slugify("x" * 100) == "x" * 40
    assert slugify("on the occasion of my first appearance before the queen") == (
        "on-the-occasion-of-my-first-appearance"
    )


def test_json_and_csv(clips):
    data = json.loads(to_json(clips, META))
    assert data["source"] == "talk.mp4" and len(data["clips"]) == 3
    c0 = data["clips"][0]
    assert {"start", "end", "score", "breakdown", "title", "hook_line", "text"} <= set(c0)
    assert c0["start"] <= c0["speech_start"] < c0["speech_end"] <= c0["end"]
    rows = list(csv.DictReader(io.StringIO(to_csv(clips))))
    assert len(rows) == 3 and rows[0]["rank"] == "1" and float(rows[0]["hook"]) > 0


def test_markdown(clips):
    md = to_markdown(clips, META)
    assert md.startswith("# Clip candidates: talk.mp4")
    assert "| 1 |" in md and f"## 1. {clips[0].title}" in md and "**Hook line:**" in md


def test_edl_is_cmx3600_and_contiguous(clips):
    edl = to_edl(clips, "talk.mp4", fps=25)
    lines = edl.splitlines()
    assert lines[:2] == ["TITLE: shorts-clip-finder", "FCM: NON-DROP FRAME"]
    events = [ln.split() for ln in lines if ln[:3].isdigit()]
    assert [e[0] for e in events] == ["001", "002", "003"]
    assert events[0][6] == "00:00:00:00"
    for prev, nxt in itertools.pairwise(events):
        assert prev[7] == nxt[6]  # record out of one event == record in of the next
    assert "* FROM CLIP NAME: talk.mp4" in edl


def test_caption_cues_are_relative_and_short():
    words = [
        Word(w, 10 + i * 0.4, 10 + i * 0.4 + 0.3)
        for i, w in enumerate("one two three four five six. seven".split())
    ]
    cues = caption_cues(words, offset=9.9, max_words=4, clip_end=13.0)
    assert [c[2] for c in cues] == ["one two three four", "five six.", "seven"]
    assert cues[0][0] == pytest.approx(0.1)
    assert all(e > s for s, e, _ in cues)
    assert cues[-1][1] <= 13.0 - 9.9 + 1e-9


def test_clip_srt_parses_back(clips):
    srt = clip_srt(clips[0])
    words = parse_srt(srt)
    assert " ".join(w.text for w in words) == clips[0].candidate.text
    assert words[-1].end <= clips[0].duration + 1e-6


def test_write_reports(tmp_path, clips):
    written = write_reports(clips, META, tmp_path / "out", ["json", "md", "edl", "csv", "sh"])
    assert sorted(p.name for p in written) == ["clips.csv", "clips.edl", "clips.json", "clips.md"]
    assert clip_basename(clips[0]).startswith("01-")
