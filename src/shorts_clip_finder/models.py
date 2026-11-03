"""Plain data classes shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Word:
    """One spoken word with its timing in seconds."""

    text: str
    start: float
    end: float
    probability: float | None = None


_NO_SPACE_BEFORE = (",", ".", ";", ":", "!", "?", "%", "'", "’", ")")


def join_words(words: tuple[Word, ...] | list[Word]) -> str:
    """Join word texts with spaces, gluing pieces Whisper splits off ("$20" ",000")."""
    out = ""
    for w in words:
        if not w.text:
            continue
        if out and not w.text.startswith(_NO_SPACE_BEFORE):
            out += " "
        out += w.text
    return out


@dataclass(frozen=True)
class Sentence:
    """A run of consecutive words ending on a sentence boundary (or a long pause)."""

    index: int
    words: tuple[Word, ...]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return join_words(self.words)


@dataclass
class Transcript:
    """A flat list of timed words plus a little metadata."""

    words: list[Word]
    source: str | None = None
    language: str | None = None
    # "exact" when the source had per-word timestamps, "interpolated" when word
    # times were estimated from cue/segment times (SRT, VTT, segment-only JSON).
    word_timing: str = "exact"

    @property
    def duration(self) -> float:
        return self.words[-1].end if self.words else 0.0


@dataclass
class Candidate:
    """A window of one or more consecutive sentences that could become a clip."""

    sentences: tuple[Sentence, ...]
    features: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    score: float = 0.0  # 0-100

    @property
    def start(self) -> float:
        return self.sentences[0].start

    @property
    def end(self) -> float:
        return self.sentences[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def words(self) -> tuple[Word, ...]:
        return tuple(w for s in self.sentences for w in s.words)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)

    @property
    def first_index(self) -> int:
        return self.sentences[0].index

    @property
    def last_index(self) -> int:
        return self.sentences[-1].index


@dataclass
class Clip:
    """A selected candidate with padded cut points and drafted text."""

    rank: int
    candidate: Candidate
    start: float
    end: float
    hook_line: str
    title: str
    alt_hook: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def score(self) -> float:
        return self.candidate.score

    def to_dict(self) -> dict:
        c = self.candidate
        return {
            "rank": self.rank,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "speech_start": round(c.start, 3),
            "speech_end": round(c.end, 3),
            "score": round(c.score, 1),
            "breakdown": {k: round(v, 3) for k, v in c.features.items()},
            "notes": list(c.notes),
            "title": self.title,
            "hook_line": self.hook_line,
            "alt_hook": self.alt_hook,
            "sentences": [c.first_index, c.last_index],
            "word_count": len(c.words),
            "text": c.text,
        }
