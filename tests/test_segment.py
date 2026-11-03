from conftest import make_words
from shorts_clip_finder.models import Word
from shorts_clip_finder.segment import is_sentence_end, segment_sentences


def texts(sentences):
    return [s.text for s in sentences]


def test_splits_on_terminal_punctuation():
    words = make_words("One two three. Is it four? Yes! “Quoted end.” Last")
    assert texts(segment_sentences(words)) == [
        "One two three.",
        "Is it four?",
        "Yes!",
        "“Quoted end.”",
        "Last",
    ]


def test_abbreviations_and_decimals_do_not_split():
    words = make_words("Dr. Rao moved to the U.S. in 2010 and earned 3.5 percent. Then more.")
    assert texts(segment_sentences(words)) == [
        "Dr. Rao moved to the U.S. in 2010 and earned 3.5 percent.",
        "Then more.",
    ]
    assert not is_sentence_end(Word("e.g.", 0, 1))


def test_abbreviation_followed_by_long_pause_ends_sentence():
    words = [
        Word("in", 0, 0.2),
        Word("the", 0.3, 0.5),
        Word("U.S.", 0.6, 1.0),
        Word("Next", 1.8, 2.0),
        Word("one.", 2.1, 2.4),
    ]
    assert texts(segment_sentences(words)) == ["in the U.S.", "Next one."]


def test_long_pause_splits_unpunctuated_speech():
    words = [
        Word("no", 0, 0.3),
        Word("punctuation", 0.4, 0.9),
        Word("here", 3.0, 3.3),
        Word("either", 3.4, 3.8),
    ]
    assert texts(segment_sentences(words, pause_split=1.5)) == ["no punctuation", "here either"]


def test_overlong_sentence_split_at_best_pause():
    words = []
    t = 0.0
    for i in range(60):
        text = f"w{i}," if i == 29 else f"w{i}"
        words.append(Word(text, t, t + 0.3))
        t += 0.5 if i != 29 else 0.9  # a slightly longer pause after the comma
    sents = segment_sentences(words, max_sentence=20.0)
    assert len(sents) == 2
    assert sents[0].words[-1].text == "w29,"
    assert all(s.duration <= 20.0 for s in sents)
    assert [s.index for s in sents] == [0, 1]


def test_empty_input():
    assert segment_sentences([]) == []


def test_join_glues_split_number_pieces():
    from shorts_clip_finder.models import join_words

    words = [Word("Give", 0, 1), Word("$20", 1, 2), Word(",000", 2, 3), Word("now.", 3, 4)]
    assert join_words(words) == "Give $20,000 now."
