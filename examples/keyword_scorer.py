"""Example custom scorer: boost clips that mention topics you care about.

shorts-clip-finder talk.srt --scorer examples/keyword_scorer.py:KeywordScorer
"""

from __future__ import annotations

TOPICS = {"budget", "invest", "saving", "savings", "debt", "interest"}


class KeywordScorer:
    def score(self, candidate) -> float:
        words = {w.text.lower().strip(".,!?;:\"'") for w in candidate.words}
        return min(1.0, len(words & TOPICS) / 3)
