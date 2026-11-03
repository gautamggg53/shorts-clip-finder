"""Transparent heuristic scoring for clip candidates.

Every feature returns a value in ``[0, 1]`` plus short human-readable notes. The final score
is the weighted mean of the features scaled to 0-100. All word lists live in this module so
they are easy to read, test and tune. See ``docs/scoring.md`` for the rationale.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

from .models import Candidate, Sentence
from .textutil import contains_phrase, is_number, normalize, strip_trailing_quotes, tokens

DEFAULT_WEIGHTS: dict[str, float] = {
    "hook": 0.30,
    "self_contained": 0.25,
    "payoff": 0.15,
    "density": 0.15,
    "specifics": 0.15,
}

CURIOSITY_WORDS = frozenset(
    """why how what secret secrets nobody never mistake mistakes truth wrong myth myths
    surprising surprisingly actually however instead biggest worst best only stop
    warning lie lies hidden real reason""".split()
)
CURIOSITY_PHRASES = (
    "turns out",
    "most people",
    "here's why",
    "here's how",
    "here's what",
    "the reason",
    "what if",
    "the problem",
    "no one",
    "the truth",
    "you won't",
    "the one thing",
)
SECOND_PERSON = frozenset({"you", "your", "you're", "yours", "you've", "you'll", "yourself"})
FILLER_OPENERS = frozenset({"um", "uh", "so", "okay", "ok", "well", "anyway", "alright", "like"})
DANGLING_OPENERS = frozenset(
    """this that these those it it's he she they him her them his its their that's he's she's
    they're they've there""".split()
)
CONJUNCTION_OPENERS = frozenset(
    "and but so because or which also then plus although though yet nor".split()
)
BACK_REFERENCES = (
    "as i said",
    "as i mentioned",
    "like i said",
    "as we saw",
    "as we discussed",
    "mentioned earlier",
    "we talked about",
    "the previous",
    "the above",
    "remember when",
    "as you can see",
    "on the screen",
    "in this chart",
)
FORWARD_REFERENCES = (
    "next up",
    "up next",
    "next time",
    "in a moment",
    "coming up",
    "let's look at",
    "we'll see",
    "stay tuned",
    "in the next",
    "later in",
    "after the break",
    "let me show you",
)
CONCLUSION_MARKERS = (
    "that's why",
    "that is why",
    "which means",
    "the lesson",
    "bottom line",
    "in other words",
    "in short",
    "the answer",
    "the point",
    "the result",
    "this is why",
    "so remember",
    "the takeaway",
    "in the end",
)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def hook_strength(sentence: Sentence) -> tuple[float, list[str]]:
    """How likely the opening sentence is to stop a scroll."""
    text = sentence.text
    toks = tokens(text)
    notes: list[str] = []
    v = 0.2
    if strip_trailing_quotes(text).endswith("?"):
        v += 0.3
        notes.append("hook: question")
    if any(is_number(w.text) for w in sentence.words):
        v += 0.2
        notes.append("hook: number")
    body = toks[1:] if toks and toks[0] in CONJUNCTION_OPENERS else toks
    if any(t in CURIOSITY_WORDS for t in body) or any(
        contains_phrase(toks, p) for p in CURIOSITY_PHRASES
    ):
        v += 0.2
        notes.append("hook: contrast/curiosity word")
    if any(t in SECOND_PERSON for t in toks):
        v += 0.15
        notes.append("hook: speaks to 'you'")
    n = len(toks)
    if 5 <= n <= 18:
        v += 0.1
    elif n > 30:
        v -= 0.15
        notes.append("hook: long opening sentence")
    elif n < 4:
        v -= 0.1
    if toks and toks[0] in FILLER_OPENERS:
        v -= 0.15
        notes.append("hook: opens with filler")
    return _clamp(v), notes


def self_containedness(c: Candidate) -> tuple[float, list[str]]:
    """Penalise clips that depend on context outside themselves."""
    notes: list[str] = []
    v = 1.0
    first = tokens(c.sentences[0].text)
    if first and first[0] in DANGLING_OPENERS:
        v -= 0.35
        notes.append(f"context: opens with '{first[0]}'")
    elif first and first[0] in CONJUNCTION_OPENERS:
        v -= 0.25
        notes.append(f"context: opens with '{first[0]}'")
    if any(contains_phrase(first, p) for p in BACK_REFERENCES):
        v -= 0.25
        notes.append("context: refers back")
    last = tokens(c.sentences[-1].text)
    if any(contains_phrase(last, p) for p in FORWARD_REFERENCES):
        v -= 0.25
        notes.append("context: ends on a teaser for later")
    return _clamp(v), notes


def payoff(c: Candidate) -> tuple[float, list[str]]:
    """Does the clip land? Full stop at the end, ideally with a conclusion marker."""
    last = c.sentences[-1]
    text = strip_trailing_quotes(last.text)
    notes: list[str] = []
    if text.endswith(("...", "…")):
        v = 0.4
        notes.append("payoff: trails off")
    elif text.endswith((".", "!")):
        v = 0.75
    elif text.endswith("?"):
        v = 0.3
        notes.append("payoff: ends on a question")
    else:
        v = 0.0
        notes.append("payoff: no closing punctuation")
    toks = tokens(text)
    if any(contains_phrase(toks, p) for p in CONCLUSION_MARKERS):
        v += 0.25
        notes.append("payoff: conclusion marker")
    return _clamp(v), notes


def speech_density(c: Candidate) -> tuple[float, list[str]]:
    """Words per second near a natural voiceover pace, with few dead-air pauses."""
    words = c.words
    dur = max(c.duration, 1e-6)
    wps = len(words) / dur
    if 2.2 <= wps <= 3.6:
        v = 1.0
    elif wps < 2.2:
        v = _clamp((wps - 1.0) / 1.2)
    else:
        v = max(0.4, 1.0 - (wps - 3.6) / 2.0)
    dead = sum(max(0.0, b.start - a.end - 1.0) for a, b in pairwise(words))
    v -= min(0.5, 2.0 * dead / dur)
    notes = [f"density: {wps:.1f} words/s"]
    if dead > 0.5:
        notes.append(f"density: {dead:.1f}s of long pauses")
    return _clamp(v), notes


def specifics(c: Candidate) -> tuple[float, list[str]]:
    """Numbers and named things per 10 seconds (saturates at 1.5)."""
    count = 0
    for s in c.sentences:
        for i, w in enumerate(s.words):
            t = normalize(w.text)
            capitalised = w.text.lstrip("\"'“‘(")[:1].isupper()
            # Mid-sentence capitalised words are a cheap proxy for names and places.
            if is_number(w.text) or (i > 0 and t and capitalised and t != "i"):
                count += 1
    rate = count / max(c.duration / 10.0, 1e-6)
    return _clamp(rate / 1.5), [f"specifics: {count} numbers/names"]


FEATURES: dict[str, Callable[[Candidate], tuple[float, list[str]]]] = {
    "hook": lambda c: hook_strength(c.sentences[0]),
    "self_contained": self_containedness,
    "payoff": payoff,
    "density": speech_density,
    "specifics": specifics,
}


@dataclass
class HeuristicScorer:
    """Scores candidates with the documented features and weights."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown weight(s): {', '.join(sorted(unknown))}")
        if sum(self.weights.values()) <= 0:
            raise ValueError("weights must sum to more than 0")

    def score(self, c: Candidate) -> Candidate:
        feats: dict[str, float] = {}
        notes: list[str] = []
        for name, fn in FEATURES.items():
            if self.weights.get(name, 0.0) == 0.0:
                continue
            value, n = fn(c)
            feats[name] = value
            notes.extend(n)
        total_w = sum(self.weights.get(k, 0.0) for k in feats)
        c.features = feats
        c.notes = notes
        c.score = 100.0 * sum(self.weights[k] * v for k, v in feats.items()) / total_w
        return c


def parse_weights(spec: str) -> dict[str, float]:
    """Parse ``"hook=0.4,payoff=0.2"`` on top of the defaults."""
    weights = dict(DEFAULT_WEIGHTS)
    for part in filter(None, (p.strip() for p in spec.split(","))):
        if "=" not in part:
            raise ValueError(f"bad weight {part!r}; expected name=value")
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in FEATURES:
            raise ValueError(f"unknown feature {k!r}; choose from {', '.join(FEATURES)}")
        weights[k] = float(v)
        if weights[k] < 0:
            raise ValueError("weights must be >= 0")
    return weights
