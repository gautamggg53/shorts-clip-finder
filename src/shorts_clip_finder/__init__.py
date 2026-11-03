"""Find self-contained short-form clip candidates in long-form speech."""

from .models import Candidate, Clip, Sentence, Transcript, Word
from .pipeline import FinderConfig, find_clips
from .scoring import DEFAULT_WEIGHTS, HeuristicScorer
from .segment import segment_sentences
from .transcripts import load_transcript

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_WEIGHTS",
    "Candidate",
    "Clip",
    "FinderConfig",
    "HeuristicScorer",
    "Sentence",
    "Transcript",
    "Word",
    "__version__",
    "find_clips",
    "load_transcript",
    "segment_sentences",
]
