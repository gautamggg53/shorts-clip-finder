import itertools

import pytest

from conftest import make_words
from shorts_clip_finder import FinderConfig, find_clips, load_transcript
from shorts_clip_finder.models import Transcript
from shorts_clip_finder.pipeline import generate_candidates, padded_bounds, rank_candidates
from shorts_clip_finder.segment import segment_sentences


@pytest.fixture
def talk(fixtures) -> Transcript:
    return load_transcript(fixtures / "talk_whisper.json")


def test_candidates_respect_length_bounds(talk):
    sents = segment_sentences(talk.words)
    cands = generate_candidates(sents, 10, 25)
    assert cands
    assert all(10 <= c.duration <= 25 for c in cands)


def test_fixture_ranking_is_deterministic_and_sensible(talk):
    cfg = FinderConfig(min_seconds=10, max_seconds=25, top=4)
    clips = find_clips(talk, cfg)
    again = find_clips(talk, cfg)
    assert [c.to_dict() for c in clips] == [c.to_dict() for c in again]
    assert [c.rank for c in clips] == [1, 2, 3, 4]
    scores = [c.score for c in clips]
    assert scores == sorted(scores, reverse=True)
    # The question hook wins; filler and teaser sentences are never chosen as openers.
    assert clips[0].hook_line == "Why do most people never save a single rupee of their raise?"
    openers = {c.hook_line.split()[0] for c in clips}
    assert not openers & {"This", "And", "Anyway,", "It", "Next"}


def test_selected_clips_never_overlap(talk):
    clips = find_clips(talk, FinderConfig(min_seconds=8, max_seconds=20, top=10))
    spans = sorted((c.start, c.end) for c in clips)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in itertools.pairwise(spans))


def test_padding_snaps_to_word_boundaries(talk):
    clips = find_clips(talk, FinderConfig(min_seconds=10, max_seconds=25, top=3))
    for c in clips:
        first, last = c.candidate.words[0], c.candidate.words[-1]
        assert c.start <= first.start and c.end >= last.end
        assert first.start - c.start <= 0.15 + 1e-9 and c.end - last.end <= 0.35 + 1e-9
        inside = [w for w in talk.words if w.start < c.end and w.end > c.start]
        assert inside == list(c.candidate.words)


def test_padding_clamps_to_zero_and_gap_midpoint():
    words = make_words("Hi there. Next.", start=0.05, gap=0.2)
    sents = segment_sentences(words)
    cands = generate_candidates(sents, 0.1, 1.0)
    first = next(c for c in cands if c.first_index == 0 and c.last_index == 0)
    start, end = padded_bounds(first, words, 0.15, 0.35)
    assert start == 0.0
    assert end == pytest.approx((words[1].end + words[2].start) / 2, abs=1e-3)


def test_custom_scorer_blends_and_reorders(talk):
    cfg = FinderConfig(min_seconds=10, max_seconds=25, top=1, scorer_weight=1.0)

    def likes_compounding(c):
        return 1.0 if "compound" in c.text.lower() else 0.0

    clip = find_clips(talk, cfg, likes_compounding)[0]
    assert "compound" in clip.candidate.text.lower()
    assert clip.candidate.features["custom"] == 1.0


def test_custom_scorer_must_return_unit_interval(talk):
    with pytest.raises(ValueError):
        find_clips(talk, FinderConfig(min_seconds=10, max_seconds=25), lambda c: 5)


def test_custom_scorer_only_sees_top_k(talk):
    seen = []
    cfg = FinderConfig(min_seconds=10, max_seconds=25, scorer_top_k=3)
    sents = segment_sentences(talk.words)
    rank_candidates(generate_candidates(sents, 10, 25), cfg, lambda c: seen.append(c) or 0.5)
    assert len(seen) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_seconds": 0},
        {"min_seconds": 30, "max_seconds": 20},
        {"top": 0},
        {"pad_start": -1},
        {"scorer_weight": 2},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        FinderConfig(**kwargs).validate()


def test_too_short_input_returns_nothing(talk):
    assert find_clips(talk, FinderConfig(min_seconds=500, max_seconds=600)) == []
