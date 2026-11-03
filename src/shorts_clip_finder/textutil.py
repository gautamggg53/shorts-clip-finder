"""Small text helpers used by scoring and title drafting."""

from __future__ import annotations

import re

_TOKEN_STRIP = re.compile(r"^[^\w$%£€₹]+|[^\w%]+$")

NUMBER_WORDS = frozenset(
    """zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen
    fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty
    ninety hundred thousand million billion trillion half double triple twice thrice percent
    dozen""".split()
)
_DIGIT = re.compile(r"\d")


def normalize(token: str) -> str:
    """Lower-case a word and strip surrounding punctuation ("You're," -> "you're")."""
    return _TOKEN_STRIP.sub("", token.lower().replace("’", "'"))


def tokens(text: str) -> list[str]:
    out = []
    for raw in text.split():
        t = normalize(raw)
        if t:
            out.append(t)
    return out


def is_number(raw_token: str) -> bool:
    t = normalize(raw_token)
    return (
        bool(_DIGIT.search(raw_token)) or t in NUMBER_WORDS or any(c in raw_token for c in "$%£€₹")
    )


def contains_phrase(toks: list[str], phrase: str) -> bool:
    """True if the space-separated *phrase* appears as consecutive tokens."""
    p = phrase.split()
    n = len(p)
    return any(toks[i : i + n] == p for i in range(len(toks) - n + 1))


def strip_trailing_quotes(text: str) -> str:
    return text.rstrip().rstrip("\"'”’)]»")
