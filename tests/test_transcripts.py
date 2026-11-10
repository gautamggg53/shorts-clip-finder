import json

import pytest

from shorts_clip_finder.transcripts import (
    TranscriptError,
    interpolate_words,
    load_transcript,
    parse_json_data,
    parse_srt,
    parse_timestamp,
    parse_vtt,
    save_transcript,
)


def test_parse_timestamp_variants():
    assert parse_timestamp("00:01:02,500") == pytest.approx(62.5)
    assert parse_timestamp("01:02.250") == pytest.approx(62.25)
    assert parse_timestamp("1:00:00.5") == pytest.approx(3600.5)
    with pytest.raises(TranscriptError):
        parse_timestamp("soon")


def test_interpolate_words_spans_cue_proportionally():
    words = interpolate_words("a bbbb", 10.0, 12.0)
    assert [w.text for w in words] == ["a", "bbbb"]
    assert words[0].start == 10.0 and words[-1].end == pytest.approx(12.0)
    assert words[1].end - words[1].start > words[0].end - words[0].start


def test_all_fixture_formats_agree(fixtures):
    j = load_transcript(fixtures / "talk_whisper.json")
    s = load_transcript(fixtures / "talk.srt")
    v = load_transcript(fixtures / "talk.vtt")
    assert [w.text for w in j.words] == [w.text for w in s.words] == [w.text for w in v.words]
    assert j.word_timing == "exact" and j.language == "en"
    assert s.word_timing == v.word_timing == "interpolated"
    # cue boundaries survive interpolation
    assert s.words[0].start == j.words[0].start
    assert s.words[-1].end == pytest.approx(j.words[-1].end)


def test_srt_strips_markup_and_speaker_marks():
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\n<i>Hello</i> {\\an8}there &amp; you\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n>> Next line.\n"
    )
    assert [w.text for w in parse_srt(srt)] == ["Hello", "there", "&", "you", "Next", "line."]


def test_vtt_skips_header_notes_and_rolling_duplicates():
    vtt = (
        "WEBVTT\nKind: captions\n\nNOTE a comment\n\n"
        "00:00:00.000 --> 00:00:02.000 align:start\nhello world\n\n"
        "00:00:02.000 --> 00:00:04.000\nhello world\nhow are you\n\n"
        "00:00:04.000 --> 00:00:06.000\nhow are you\n<00:00:04.500><c>fine</c> thanks\n"
    )
    assert [w.text for w in parse_vtt(vtt)] == [
        "hello",
        "world",
        "how",
        "are",
        "you",
        "fine",
        "thanks",
    ]


def test_json_shapes():
    words, lang, exact = parse_json_data(
        {"words": [{"text": "hi", "start": 0, "end": 0.5}, {"word": " yo", "start": 1, "end": 2}]}
    )
    assert [w.text for w in words] == ["hi", "yo"] and exact and lang is None
    # bare list of words
    words, _, exact = parse_json_data([{"word": "a", "start": 0, "end": 1}])
    assert len(words) == 1 and exact
    # segments without words -> interpolated
    words, _, exact = parse_json_data([{"start": 0, "end": 2, "text": "one two"}])
    assert [w.text for w in words] == ["one", "two"] and not exact


def test_whisperx_untimed_words_are_filled():
    data = {
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "it cost 50 dollars",
                "words": [
                    {"word": "it", "start": 0.0, "end": 0.2, "score": 0.9},
                    {"word": "cost", "start": 0.3, "end": 0.6},
                    {"word": "50"},
                    {"word": "dollars", "start": 1.5, "end": 2.0},
                ],
            }
        ]
    }
    words, _, exact = parse_json_data(data)
    assert not exact
    assert words[2].text == "50" and 0.6 <= words[2].start < words[2].end <= 1.5
    assert words[0].probability == 0.9


def test_roundtrip_own_format(tmp_path, fixtures):
    t = load_transcript(fixtures / "talk_whisper.json")
    out = tmp_path / "t.json"
    save_transcript(t, out, {"model": "test"})
    data = json.loads(out.read_text())
    assert data["format"] == "shorts-clip-finder.transcript" and data["model"] == "test"
    t2 = load_transcript(out)
    assert [(w.text, w.start) for w in t2.words] == [(w.text, w.start) for w in t.words]


@pytest.mark.parametrize(
    ("name", "content"),
    [("x.json", "{not json"), ("x.json", '{"foo": 1}'), ("x.srt", ""), ("x.txt", "hi")],
)
def test_bad_transcripts_raise(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    with pytest.raises(TranscriptError):
        load_transcript(p)


SRT_TEXT = "1\n00:00:01,000 --> 00:00:02,000\nCafé prices rose 5%\n"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-be", "utf-32"])
def test_srt_with_bom_or_utf8_loads(tmp_path, encoding):
    p = tmp_path / "talk.srt"
    raw = SRT_TEXT.encode(encoding)
    if encoding == "utf-16-be":  # this codec writes no BOM itself
        raw = b"\xfe\xff" + raw
    p.write_bytes(raw)
    t = load_transcript(p)
    assert [w.text for w in t.words] == ["Café", "prices", "rose", "5%"]


def test_non_utf8_transcript_raises_clear_error(tmp_path):
    p = tmp_path / "talk.srt"
    p.write_bytes(SRT_TEXT.encode("cp1252"))
    with pytest.raises(TranscriptError, match="re-save the file as UTF-8"):
        load_transcript(p)


def test_utf16_json_transcript_loads(tmp_path):
    p = tmp_path / "t.json"
    data = {"segments": [{"start": 0, "end": 1, "text": "Hello there"}]}
    p.write_text(json.dumps(data), encoding="utf-16")
    assert [w.text for w in load_transcript(p).words] == ["Hello", "there"]
