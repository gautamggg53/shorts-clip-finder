"""Draft a hook line and an on-screen title from a clip, without any language model."""

from __future__ import annotations

import re

from .models import Candidate, Sentence
from .scoring import CONJUNCTION_OPENERS, FILLER_OPENERS, hook_strength
from .textutil import normalize, strip_trailing_quotes

SMALL_WORDS = frozenset("a an the and but or for nor on at to by of in vs via as per".split())
TRAILING_STOPWORDS = SMALL_WORDS | frozenset(
    "is are was were be that this with your my our".split()
)
_LEADING_FILLERS = (
    FILLER_OPENERS | CONJUNCTION_OPENERS | {"now", "look", "listen", "right", "ah", "oh"}
)


def _strip_leading_fillers(words: list[str]) -> list[str]:
    while len(words) > 3 and normalize(words[0]) in _LEADING_FILLERS:
        words = words[1:]
    if len(words) > 3 and [normalize(w) for w in words[:2]] == ["you", "know"]:
        words = words[2:]
    return words


def title_case(words: list[str]) -> str:
    out = []
    for i, w in enumerate(words):
        core = normalize(w)
        if 0 < i < len(words) - 1 and core in SMALL_WORDS:
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def _shorten(words: list[str], max_words: int) -> list[str]:
    if len(words) <= max_words:
        return words
    # Prefer cutting at a clause boundary (allowing two words of slack over the limit).
    for i in range(min(max_words + 2, len(words)) - 1, 2, -1):
        if re.search(r"[,;:—–]$", words[i]):
            return words[: i + 1]
    words = words[:max_words]
    while len(words) > 3 and normalize(words[-1]) in TRAILING_STOPWORDS:
        words = words[:-1]
    return words


QUESTION_STARTS = frozenset(
    "what why how who when where which is are do does did can could would should will".split()
)
_PUNCT = " ,;:—–-.!?\"'”’"


def _is_question(s: Sentence) -> bool:
    return strip_trailing_quotes(s.text).endswith("?")


def question_words(s: Sentence, max_words: int) -> list[str] | None:
    """The question as title words: the whole sentence if short, else its final question
    clause ("Ah, my boys, he cries, what is the use...?" -> "what is the use...?")."""
    words = _strip_leading_fillers(s.text.split())
    if len(words) <= max_words:
        return words
    for i in range(1, len(words)):
        tail = words[i:]
        if (
            words[i - 1].endswith(",")
            and 4 <= len(tail) <= max_words
            and normalize(tail[0]) in QUESTION_STARTS
        ):
            return tail
    return None


def pick_title_sentence(c: Candidate, max_words: int = 12) -> Sentence:
    head = c.sentences[:3]
    for s in head:
        if _is_question(s) and question_words(s, max_words):
            return s
    return max(head, key=lambda s: (hook_strength(s)[0], -s.index))


def _fit_chars(words: list[str], max_chars: int) -> list[str]:
    while len(title_case(words).rstrip(_PUNCT)) > max_chars and len(words) > 3:
        words = words[:-1]
        while len(words) > 3 and normalize(words[-1]) in TRAILING_STOPWORDS:
            words = words[:-1]
    return words


def draft_title(c: Candidate, *, max_words: int = 8, max_chars: int = 60) -> str:
    """Pick a short question from the opening if there is one (a whole question of up to
    ``max_words + 4`` words, or its final question clause), otherwise the opening sentence
    with the strongest hook, trimmed at a clause boundary or the word limit."""
    s = pick_title_sentence(c, max_words + 4)
    q = question_words(s, max_words + 4) if _is_question(s) else None
    words = q if q is not None else _shorten(_strip_leading_fillers(s.text.split()), max_words)
    fitted = _fit_chars(words, max_chars)
    title = title_case(fitted).rstrip(_PUNCT)
    if q is not None and fitted == q:
        title += "?"
    return title[:1].upper() + title[1:]


def hook_lines(c: Candidate) -> tuple[str, str | None]:
    """Return (opening line, stronger alternative from inside the clip or None)."""
    first = c.sentences[0]
    first_score = hook_strength(first)[0]
    alt = None
    best = max(c.sentences[1:], key=lambda s: hook_strength(s)[0], default=None)
    if best is not None and hook_strength(best)[0] >= first_score + 0.2:
        alt = best.text
    return first.text, alt
