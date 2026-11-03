"""Local transcription with faster-whisper.

Optional extra: ``pip install "shorts-clip-finder[whisper]"``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .media import decode_audio
from .models import Transcript, Word


class WhisperUnavailable(RuntimeError):
    pass


def transcribe(
    path: str | Path,
    *,
    model: str = "small.en",
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    beam_size: int = 5,
    vad: bool = True,
    progress: Callable[[float], None] | None = None,
) -> Transcript:
    """Transcribe *path* locally and return word-timed :class:`Transcript`.

    No ``initial_prompt`` is passed (it can make Whisper hallucinate the prompt text), and
    ``condition_on_previous_text`` is off to reduce repetition loops. Voice activity
    detection (VAD) trims silence, which improves word timings around pauses.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover - depends on the optional extra
        raise WhisperUnavailable(
            "faster-whisper is not installed. Install it with "
            "`pip install 'shorts-clip-finder[whisper]'`, or pass an existing "
            "transcript with --transcript."
        ) from e

    if language is None and model.endswith(".en"):
        language = "en"
    audio = decode_audio(path)
    total = len(audio) / 16000 or 1.0
    wm = WhisperModel(model, device=device, compute_type=compute_type)
    segments, info = wm.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=vad,
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(text, float(w.start), float(w.end), float(w.probability)))
        if progress:
            progress(min(1.0, seg.end / total))
    return Transcript(words=words, source=str(path), language=info.language, word_timing="exact")
