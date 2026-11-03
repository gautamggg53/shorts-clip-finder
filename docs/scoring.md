# How clips are scored

Everything here is a heuristic. It ranks candidates so you review the promising ones first;
it does not know whether a clip will perform. All word lists are in
`src/shorts_clip_finder/scoring.py`.

## Pipeline

1. **Words.** From faster-whisper (exact word times) or a transcript. SRT/VTT cues and
   segment-only JSON have no word times, so they are spread across each cue by word length
   (`word_timing: "interpolated"` in the output).
2. **Sentences.** Split after `.` `!` `?` `…` (not after abbreviations like `Dr.` or `U.S.`),
   or at a pause of at least 1.5 s. A sentence longer than half of `--max` is split at its
   longest pause, preferring a pause after a comma.
3. **Candidates.** Every run of consecutive whole sentences whose speech lasts between
   `--min` and `--max` seconds.
4. **Scoring.** Five features, each 0-1, combined as a weighted mean and scaled to 0-100.
5. **Selection.** Highest score first; a candidate is skipped if it shares a sentence with a
   clip already picked. Ties go to the length closest to the middle of the range, then the
   earlier clip.
6. **Cut points.** Start = first word − `--pad-start` (0.15 s), end = last word + `--pad-end`
   (0.35 s), but never past the midpoint of the silence to the neighbouring word, so a clip
   never includes part of another sentence.

## Features and default weights

| Feature | Weight | What raises it | What lowers it |
|---|---|---|---|
| `hook` | 0.30 | Opening sentence is a question (+0.30), has a number (+0.20), a contrast/curiosity word such as *why, secret, mistake, actually, most people* (+0.20), speaks to *you* (+0.15), is 5-18 words (+0.10). Base 0.20. | Opens with filler (*so, um, well, okay*) −0.15, over 30 words −0.15, under 4 words −0.10 |
| `self_contained` | 0.25 | Starts at 1.0 | Opens with a pronoun that points back (*this, that, it, he, they…*) −0.35, or a conjunction (*and, but, so, because…*) −0.25; back-references like *as I said* −0.25; ends on a teaser (*next up, in the next video, stay tuned*) −0.25 |
| `payoff` | 0.15 | Ends with `.` or `!` (0.75); conclusion marker in the last sentence (*that's why, which means, the lesson, in other words*) +0.25 | Ends with `…` (0.4), `?` (0.3), or no punctuation (0.0) |
| `density` | 0.15 | 2.2-3.6 words per second = 1.0 | Slower pace falls linearly to 0 at 1 word/s; faster than 3.6 drops gently (floor 0.4); dead air (pauses over 1 s) subtracts up to 0.5 |
| `specifics` | 0.15 | Numbers, currency/percent signs and capitalised mid-sentence words (a cheap proxy for names), per 10 s; 1.5 per 10 s = 1.0 | Vague, name-free speech |

Change weights with `--weights hook=0.4,specifics=0.05`; a weight of 0 turns a feature off.

## Custom scorers

`--scorer module:attr` (or `path/to/file.py:attr`) loads a callable or a class with a
`score(candidate) -> float` method returning 0-1. Only the top `--scorer-top-k` (40)
heuristic candidates are sent to it, and the final score is
`(1 - w) * heuristic + w * 100 * custom` with `w = --scorer-weight` (0.5). The value appears
as `custom` in the breakdown. See `examples/`.

## Known blind spots

- English word lists only. Other languages still get density, payoff and specifics, but the
  hook and context rules mostly won't fire.
- Capitalised words are counted as "names", so title-cased transcripts inflate `specifics`.
- Pronoun openers are always penalised, even when the reference is resolved in the same
  sentence ("This one habit saved me…").
- It cannot see the video, hear tone or tell if something is funny.
