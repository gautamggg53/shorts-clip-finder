import pytest

from conftest import make_candidate
from shorts_clip_finder.models import Candidate
from shorts_clip_finder.scoring import (
    DEFAULT_WEIGHTS,
    HeuristicScorer,
    hook_strength,
    parse_weights,
    payoff,
    self_containedness,
    specifics,
    speech_density,
)


def hook(text: str) -> float:
    return hook_strength(make_candidate(text).sentences[0])[0]


def test_hook_rewards_questions_numbers_curiosity_and_you():
    plain = hook("The weather was mild during the long walk home.")
    assert hook("Why do you lose money every single month?") > plain
    assert hook("Most people waste 30 percent of their income.") > plain
    assert hook("Here's why your budget keeps failing.") > plain


def test_hook_penalises_filler_and_rambling():
    assert hook("So the weather was mild during the walk.") < hook(
        "The weather was mild during the walk."
    )
    long = " ".join(["word"] * 35) + "."
    assert hook(long) < hook("A short clear opening line here.")


def test_self_contained_penalises_dangling_starts_and_teasers():
    clean = self_containedness(make_candidate("Rent is your biggest cost.", "Cut it first."))[0]
    assert clean == 1.0
    assert self_containedness(make_candidate("That is the problem.", "Fix it."))[0] < clean
    assert self_containedness(make_candidate("And then it happened.", "Done."))[0] < clean
    assert self_containedness(make_candidate("As I said, rent matters.", "Ok."))[0] < clean
    assert self_containedness(make_candidate("Rent matters.", "Next up, food costs."))[0] < clean


@pytest.mark.parametrize(
    ("ending", "expected"),
    [
        ("That's why you start early.", 1.0),
        ("You start early.", 0.75),
        ("Do you start early?", 0.3),
        ("You start early...", 0.4),
        ("you start early", 0.0),
    ],
)
def test_payoff(ending, expected):
    assert payoff(make_candidate("Opening line here.", ending))[0] == pytest.approx(expected)


def test_density_prefers_natural_pace():
    natural = speech_density(make_candidate("one two three four five six seven eight.", wps=2.8))
    slow = speech_density(make_candidate("one two three four five six seven eight.", wps=1.2))
    assert natural[0] == 1.0 and slow[0] < 0.3
    assert "words/s" in natural[1][0]


def test_specifics_counts_numbers_and_names():
    vague = specifics(make_candidate("things got better over time for people."))[0]
    sharp = specifics(make_candidate("In 2019 Priya saved 40 percent in Pune."))[0]
    assert sharp > vague


def test_scorer_is_bounded_and_deterministic():
    c = make_candidate("Why do you overspend?", "Because spending is automatic.")
    s1 = HeuristicScorer().score(c).score
    s2 = (
        HeuristicScorer()
        .score(make_candidate("Why do you overspend?", "Because spending is automatic."))
        .score
    )
    assert 0 <= s1 <= 100 and s1 == s2
    assert set(c.features) == set(DEFAULT_WEIGHTS)


def test_zero_weight_drops_feature():
    c = HeuristicScorer({**DEFAULT_WEIGHTS, "specifics": 0.0}).score(make_candidate("Hi there."))
    assert "specifics" not in c.features


def test_parse_weights():
    w = parse_weights("hook=0.5, payoff=0")
    assert w["hook"] == 0.5 and w["payoff"] == 0 and w["density"] == DEFAULT_WEIGHTS["density"]
    for bad in ("nope=1", "hook", "hook=-1"):
        with pytest.raises(ValueError):
            parse_weights(bad)
    with pytest.raises(ValueError):
        HeuristicScorer({"bogus": 1.0})
    with pytest.raises(ValueError):
        HeuristicScorer(dict.fromkeys(DEFAULT_WEIGHTS, 0.0))


def test_candidate_properties():
    c: Candidate = make_candidate("One two.", "Three four.")
    assert c.text == "One two. Three four." and len(c.words) == 4
    assert c.first_index == 0 and c.last_index == 1 and c.duration > 0
