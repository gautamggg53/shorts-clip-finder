"""Split a word stream into sentences.

A sentence ends when a word ends in ``.``, ``!``, ``?`` or ``…`` (optionally followed by a
closing quote or bracket) and is not a known abbreviation, or when the speaker pauses for at
least ``pause_split`` seconds. Sentences longer than ``max_sentence`` seconds (common when a
transcript has little punctuation) are split recursively at their longest internal pause,
preferring pauses after a comma.
"""

from __future__ import annotations

import re

from .models import Sentence, Word

ABBREVIATIONS = frozenset(
    """mr. mrs. ms. dr. prof. sr. jr. st. vs. etc. e.g. i.e. inc. ltd. co. corp. no. approx.
    jan. feb. mar. apr. jun. jul. aug. sep. sept. oct. nov. dec. u.s. u.k. a.m. p.m. fig.""".split()
)
_END = re.compile(r"[.!?…]+[\"'”’)\]»]*$")
_INITIALS = re.compile(r"^(?:[a-z]\.){2,}$")


def is_sentence_end(word: Word) -> bool:
    t = word.text.lower()
    if not _END.search(t):
        return False
    return t not in ABBREVIATIONS and not _INITIALS.match(t)


def _split_long(words: list[Word], max_sentence: float) -> list[list[Word]]:
    if len(words) < 2 or words[-1].end - words[0].start <= max_sentence:
        return [words]
    best_i, best_score = None, -1.0
    for i in range(len(words) - 1):
        gap = words[i + 1].start - words[i].end
        score = gap + (0.25 if words[i].text.endswith((",", ";", ":", "—", "-")) else 0.0)
        # Keep both halves non-trivial.
        if i >= 1 and i <= len(words) - 3 and score > best_score:
            best_i, best_score = i, score
    if best_i is None:
        best_i = len(words) // 2 - 1
    return _split_long(words[: best_i + 1], max_sentence) + _split_long(
        words[best_i + 1 :], max_sentence
    )


def segment_sentences(
    words: list[Word], *, pause_split: float = 1.5, max_sentence: float = 20.0
) -> list[Sentence]:
    """Group *words* into :class:`Sentence` objects in time order."""
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt.start - w.end) if nxt else 0.0
        t = w.text.lower()
        boundary = nxt is None or is_sentence_end(w) or gap >= pause_split
        # An abbreviation at the very end of a sentence ("... in the U.S.") followed by a pause.
        if not boundary and (t in ABBREVIATIONS or _INITIALS.match(t)) and gap >= 0.6:
            boundary = True
        if boundary:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    out: list[Sentence] = []
    for g in groups:
        for part in _split_long(g, max_sentence):
            out.append(Sentence(index=len(out), words=tuple(part)))
    return out
