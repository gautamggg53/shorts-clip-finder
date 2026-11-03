from conftest import make_candidate
from shorts_clip_finder.titles import draft_title, hook_lines, title_case


def test_title_case_keeps_small_words_lower():
    assert title_case("the cost of a bad habit".split()) == "The Cost of a Bad Habit"
    assert title_case(["SIP", "vs", "FD"]) == "SIP vs FD"


def test_short_question_becomes_title():
    c = make_candidate("So why do budgets fail?", "Because nobody tracks them.")
    assert draft_title(c) == "Why Do Budgets Fail?"


def test_long_sentence_is_cut_at_clause_or_word_limit():
    c = make_candidate(
        "Most families spend twenty percent more than they think, and the gap grows every year."
    )
    title = draft_title(c)
    assert title == "Most Families Spend Twenty Percent More Than They Think"
    assert len(title) <= 60


def test_no_clause_falls_back_to_word_limit_without_dangling_stopword():
    c = make_candidate("Every single month the bank quietly takes a fee from your account balance.")
    assert draft_title(c) == "Every Single Month the Bank Quietly Takes"


def test_clause_boundary_preferred():
    c = make_candidate("Compound growth is slow, then suddenly it is not at all.")
    assert draft_title(c) == "Compound Growth Is Slow"


def test_hook_lines_offers_stronger_alternative():
    c = make_candidate("It was a long day.", "Why do 90 percent of you never save?", "Fine.")
    first, alt = hook_lines(c)
    assert first == "It was a long day."
    assert alt == "Why do 90 percent of you never save?"
    assert hook_lines(make_candidate("Why do you never save?", "ok."))[1] is None


def test_long_question_uses_its_final_question_clause():
    c = make_candidate("Ah, my friends, he cries, what is the point of saving if you never spend?")
    assert draft_title(c) == "What Is the Point of Saving If You Never Spend?"


def test_char_limit_ignores_trailing_punctuation():
    c = make_candidate("On the occasion of my first appearance before Queen Victoria, he spoke.")
    assert draft_title(c) == "On the Occasion of My First Appearance Before Queen Victoria"
