"""Regenerate the synthetic transcript fixtures from talk.txt (deterministic).

Run: python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
WPS = 2.8  # words per second
GAP = 0.45  # pause between sentences


def timed_sentences(lines: list[str]) -> list[list[tuple[str, float, float]]]:
    t = 0.5
    out = []
    for line in lines:
        words = []
        for w in line.split():
            d = round(0.12 + 0.04 * len(w), 3) if len(w) > 2 else 0.2
            d = min(d, 1 / WPS * 1.6)
            words.append((w, round(t, 3), round(t + d, 3)))
            t += 1 / WPS
        out.append(words)
        t += GAP
    return out


def ts(sec: float, sep: str) -> str:
    ms = round(sec * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def main() -> None:
    lines = [ln.strip() for ln in (HERE / "talk.txt").read_text().splitlines() if ln.strip()]
    sents = timed_sentences(lines)
    segments = [
        {
            "start": s[0][1],
            "end": s[-1][2],
            "text": " " + " ".join(w for w, _, _ in s),
            "words": [
                {"word": " " + w, "start": a, "end": b, "probability": 0.95} for w, a, b in s
            ],
        }
        for s in sents
    ]
    (HERE / "talk_whisper.json").write_text(
        json.dumps({"language": "en", "segments": segments}, indent=1) + "\n"
    )
    srt, vtt = [], ["WEBVTT", ""]
    for i, s in enumerate(sents, 1):
        text = " ".join(w for w, _, _ in s)
        srt += [str(i), f"{ts(s[0][1], ',')} --> {ts(s[-1][2], ',')}", text, ""]
        vtt += [f"{ts(s[0][1], '.')} --> {ts(s[-1][2], '.')} align:start", text, ""]
    (HERE / "talk.srt").write_text("\n".join(srt))
    (HERE / "talk.vtt").write_text("\n".join(vtt))


if __name__ == "__main__":
    main()
