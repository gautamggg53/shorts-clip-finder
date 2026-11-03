"""Real faster-whisper transcription. Opt-in: ``pytest -m slow``.

Needs the ``[whisper]`` extra and either ``espeak-ng`` on PATH (to synthesise speech) or
``SCF_SPEECH_SAMPLE=/path/to/speech.(wav|mp3)`` pointing at English speech you may use.
The first run downloads the model (~250 MB for small.en).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import requires_ffmpeg

pytestmark = [pytest.mark.slow, requires_ffmpeg]

TEXT = (
    "Why do most budgets fail in the first month? Because people plan for an average month, "
    "and no month is average. Add a buffer of ten percent and review it every Sunday. "
    "That is why simple budgets last longer than clever ones."
)


@pytest.fixture(scope="module")
def speech(tmp_path_factory) -> Path:
    pytest.importorskip("faster_whisper")
    env = os.environ.get("SCF_SPEECH_SAMPLE")
    if env:
        return Path(env)
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if not espeak:
        pytest.skip("set SCF_SPEECH_SAMPLE or install espeak-ng")
    wav = tmp_path_factory.mktemp("speech") / "speech.wav"
    subprocess.run([espeak, "-s", "150", "-w", str(wav), TEXT], check=True)
    return wav


def test_transcribe_returns_word_timestamps(speech):
    from shorts_clip_finder.transcribe import transcribe

    model = os.environ.get("SCF_WHISPER_MODEL", "small.en")
    t = transcribe(speech, model=model, compute_type="int8")
    assert len(t.words) > 10
    assert all(w.end >= w.start for w in t.words)
    assert all(a.start <= b.start for a, b in zip(t.words, t.words[1:]))
    assert t.word_timing == "exact" and t.language == "en"
