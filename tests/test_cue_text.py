from shorts_clip_finder.transcripts import parse_srt


def _text(srt: str) -> str:
    return " ".join(w.text for w in parse_srt(srt))


def test_leading_minus_on_number_is_kept():
    srt = "1\n00:00:01,000 --> 00:00:03,000\n-5% on the day, then -$200 more\n"
    assert _text(srt) == "-5% on the day, then -$200 more"


def test_speaker_dash_and_chevrons_are_stripped():
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\n- Hello there\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n>>Next speaker\n"
    )
    assert _text(srt) == "Hello there Next speaker"
