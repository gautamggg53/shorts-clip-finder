from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from shorts_clip_finder.models import Candidate, Sentence, Word

FIXTURES = Path(__file__).parent / "fixtures"
requires_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed"
)


def make_words(text: str, start: float = 0.0, wps: float = 2.8, gap: float = 0.4) -> list[Word]:
    """Deterministic word timings: one word every 1/wps s, *gap* s pause after each sentence."""
    words, t = [], start
    for tok in text.split():
        words.append(Word(tok, round(t, 3), round(t + 0.8 / wps, 3)))
        t += 1 / wps
        if tok.endswith((".", "!", "?")):
            t += gap
    return words


def make_candidate(*sentences: str, wps: float = 2.8) -> Candidate:
    words = make_words(" ".join(sentences), wps=wps)
    out, i = [], 0
    for idx, s in enumerate(sentences):
        n = len(s.split())
        out.append(Sentence(idx, tuple(words[i : i + n])))
        i += n
    return Candidate(sentences=tuple(out))


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES
