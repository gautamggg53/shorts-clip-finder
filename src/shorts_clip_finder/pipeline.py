"""Candidate generation, ranking, non-overlapping selection and cut-point padding."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Candidate, Clip, Sentence, Transcript, Word
from .plugins import ScoreFn
from .scoring import DEFAULT_WEIGHTS, HeuristicScorer
from .segment import segment_sentences
from .titles import draft_title, hook_lines


@dataclass
class FinderConfig:
    min_seconds: float = 30.0
    max_seconds: float = 60.0
    top: int = 5
    pad_start: float = 0.15
    pad_end: float = 0.35
    pause_split: float = 1.5
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # Optional extra scorer (e.g. an LLM) blended into the heuristic score.
    scorer_weight: float = 0.5
    scorer_top_k: int = 40

    def validate(self) -> None:
        if self.min_seconds <= 0 or self.max_seconds < self.min_seconds:
            raise ValueError("need 0 < min_seconds <= max_seconds")
        if self.top < 1:
            raise ValueError("top must be >= 1")
        if self.pad_start < 0 or self.pad_end < 0:
            raise ValueError("padding must be >= 0")
        if not 0.0 <= self.scorer_weight <= 1.0:
            raise ValueError("scorer_weight must be between 0 and 1")


def generate_candidates(
    sentences: list[Sentence], min_seconds: float, max_seconds: float
) -> list[Candidate]:
    """Every run of consecutive sentences whose speech span fits [min, max] seconds."""
    out = []
    n = len(sentences)
    for i in range(n):
        for j in range(i, n):
            dur = sentences[j].end - sentences[i].start
            if dur > max_seconds:
                break
            if dur >= min_seconds:
                out.append(Candidate(sentences=tuple(sentences[i : j + 1])))
    return out


def _overlaps(a: Candidate, b: Candidate) -> bool:
    return a.first_index <= b.last_index and b.first_index <= a.last_index


def rank_candidates(
    candidates: list[Candidate], config: FinderConfig, extra: ScoreFn | None = None
) -> list[Candidate]:
    scorer = HeuristicScorer(config.weights)
    target = (config.min_seconds + config.max_seconds) / 2

    def order(c: Candidate) -> tuple[float, float, float]:
        return (-round(c.score, 6), abs(c.duration - target), c.start)

    for c in candidates:
        scorer.score(c)
    ranked = sorted(candidates, key=order)
    if extra is None or config.scorer_weight == 0:
        return ranked
    pool, rest = ranked[: config.scorer_top_k], ranked[config.scorer_top_k :]
    w = config.scorer_weight
    for c in pool:
        value = float(extra(c))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"custom scorer returned {value!r}; expected a value in [0, 1]")
        c.features["custom"] = value
        c.score = (1 - w) * c.score + w * 100.0 * value
    return sorted(pool, key=order) + rest


def select_non_overlapping(ranked: list[Candidate], top: int) -> list[Candidate]:
    picked: list[Candidate] = []
    for c in ranked:
        if all(not _overlaps(c, p) for p in picked):
            picked.append(c)
            if len(picked) == top:
                break
    return picked


def padded_bounds(
    c: Candidate, all_words: list[Word], pad_start: float, pad_end: float
) -> tuple[float, float]:
    """Pad around the first/last word.

    Padding never reaches past the midpoint of the silence to the neighbouring word, so a
    clip never contains part of another sentence and adjacent clips never overlap.
    """
    first, last = c.words[0], c.words[-1]
    prev_end = max((w.end for w in all_words if w.end <= first.start), default=None)
    next_start = min((w.start for w in all_words if w.start >= last.end), default=None)
    start = first.start - pad_start
    if prev_end is not None:  # stay in the first half of the silence before the clip
        start = max(start, (prev_end + first.start) / 2)
    end = last.end + pad_end
    if next_start is not None:  # ... and in the first half of the silence after it
        end = min(end, (last.end + next_start) / 2)
    start = max(start, 0.0)
    return round(start, 3), round(end, 3)


def find_clips(
    transcript: Transcript, config: FinderConfig | None = None, extra: ScoreFn | None = None
) -> list[Clip]:
    """Run segmentation, scoring and selection; return up to ``config.top`` clips."""
    config = config or FinderConfig()
    config.validate()
    sentences = segment_sentences(
        transcript.words, pause_split=config.pause_split, max_sentence=config.max_seconds / 2
    )
    candidates = generate_candidates(sentences, config.min_seconds, config.max_seconds)
    ranked = rank_candidates(candidates, config, extra)
    picked = select_non_overlapping(ranked, config.top)
    clips = []
    for rank, c in enumerate(picked, 1):
        start, end = padded_bounds(c, transcript.words, config.pad_start, config.pad_end)
        hook, alt = hook_lines(c)
        clips.append(
            Clip(
                rank=rank,
                candidate=c,
                start=start,
                end=end,
                hook_line=hook,
                title=draft_title(c),
                alt_hook=alt,
            )
        )
    return clips
