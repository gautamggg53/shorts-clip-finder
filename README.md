# shorts-clip-finder

[![CI](https://github.com/gautamggg53/shorts-clip-finder/actions/workflows/ci.yml/badge.svg)](https://github.com/gautamggg53/shorts-clip-finder/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Find the best self-contained 30-60 second moments in a long voiceover, podcast or video, ready
to review and cut for YouTube Shorts, Instagram Reels or TikTok.

- **Local and free.** Transcribes on your machine with
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper). No API keys, no uploads.
- **Or skip transcription.** Feed an existing SRT, WebVTT or Whisper/WhisperX JSON
  transcript. That path has **no ML dependencies** at all.
- **Transparent scoring.** Every clip comes with a score breakdown and plain-English notes
  ("hook: question", "context: opens with 'that'"). No black box, no LLM required.
- **Editor-friendly output.** JSON, CSV, Markdown, a CMX3600 EDL for Premiere/Resolve, an
  ffmpeg cut script, per-clip SRT captions, and optional one-step cutting with 9:16 crop and
  burned-in captions.

> **What it is not:** it can't predict virality. It ranks candidates with heuristics so you
> watch the promising ones first. You still pick, and you still edit.

## Install

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) (for transcription
and cutting).

It is not on PyPI yet, so install from GitHub:

```bash
# with local transcription
pip install "shorts-clip-finder[whisper] @ git+https://github.com/gautamggg53/shorts-clip-finder"
# transcript-only (SRT/VTT/JSON), no ML dependencies
pip install "git+https://github.com/gautamggg53/shorts-clip-finder"
```

The first transcription downloads the Whisper model (`small.en` is about 250 MB).

## Quick start

```bash
# Media in: transcribe locally, find 5 clips, write reports to ./talk_shorts/
shorts-clip-finder talk.mp4

# Also cut them: exact re-encode, vertical 1080x1920, burned-in captions
shorts-clip-finder talk.mp4 --export-clips --vertical --burn-captions

# Already have captions? Skip Whisper entirely
shorts-clip-finder talk.srt
shorts-clip-finder talk.mp4 -t talk.srt --export-clips     # use the SRT, cut the video

# Re-rank without re-transcribing (the transcript is saved on the first run)
shorts-clip-finder talk.mp4 -t talk_shorts/transcript.json --min 20 --max 45 -n 8
```

## A real run

Input: Chapter 7 of P. T. Barnum's *The Art of Money Getting*, read for
[LibriVox](https://librivox.org/) (public domain, 10 min 16 s, mono MP3). The audio is not
included in this repo. To reproduce:

```bash
curl -L -o art-of-money-getting-ch07.mp3 \
  https://archive.org/download/art_of_money_getting_librivox/artofmoneygetting_07_barnum_64kb.mp3
shorts-clip-finder art-of-money-getting-ch07.mp3 -o out --export-clips --cut-mode exact
```

Output (8-core CPU, no GPU, `small.en`; transcription took 29 s):

```text
transcribing art-of-money-getting-ch07.mp3 with faster-whisper 'small.en' (local)...
saved transcript to out/transcript.json (29s); reuse it with -t out/transcript.json
cutting 5 clip(s) with ffmpeg (exact)...
 #  start        end            len  score  title
 1  00:01:03.970 00:02:03.070 59.1s   87.7  How Are You Going to Get It?
    hook: How are you going to get it?
 2  00:00:25.450 00:01:03.370 37.9s   82.1  What Is the Object of Learning My Trade
    hook: Chapter 7 Don't Get Above Your Business Young men, after they get through their business training or apprenticeship, instead of pursuing their avocation and rising in their business, will often lie about doing nothing.
 3  00:06:35.830 00:07:19.100 43.3s   81.7  What Is the Use of Being Rich If You Can't Enjoy Yourself?
    hook: Ah, my boys, he cries, what is the use of being rich if you can't enjoy yourself?
 4  00:02:59.510 00:03:43.280 43.8s   81.2  Stephen Girard Started Life as a Poor Cabin
    hook: Stephen Girard started life as a poor cabin boy and died worth nine million dollars.
 5  00:07:51.070 00:08:37.040 46.0s   78.2  On the Occasion of My First Appearance Before Queen Victoria
    hook: On the occasion of my first appearance before Queen Victoria, the Duke of Wellington asked me what sphere in life General Tom Thumb's parents were in.

wrote 10 file(s) to out
```

Things to notice, good and bad: clip 2 starts with the chapter heading because Whisper didn't
punctuate it ("...Your Business Young men,"), and title 4 is cut at the 8-word limit. This is
why the output is a shortlist to review, not a finished edit.

From `out/clips.md`:

```markdown
## 1. How Are You Going to Get It?

- **Cut:** 00:01:03.970 → 00:02:03.070 (59.1s)
- **Score:** 87.7 — hook 0.95, self_contained 1.00, payoff 0.75, density 0.96, specifics 0.57
- **Hook line:** How are you going to get it?
- **Notes:** hook: question; hook: contrast/curiosity word; hook: speaks to 'you'; density: 2.9 words/s; density: 1.1s of long pauses; specifics: 5 numbers/names
```

## Output files

| File | What it is |
|---|---|
| `clips.json` | Everything: cut points, speech span, score, breakdown, notes, title, hook line, text |
| `clips.csv` | One row per clip, for spreadsheets |
| `clips.md` | Human-readable review sheet |
| `clips.edl` | CMX3600 EDL (one event per clip, `--fps` default 30) for Premiere Pro / DaVinci Resolve |
| `cut_clips.sh` | The exact ffmpeg commands, so you can tweak and re-run |
| `clips/NN-title.srt` | Per-clip captions, times relative to the clip (short 4-word cues) |
| `clips/NN-title.mp4` | The cut clips, with `--export-clips` |
| `transcript.json` | Word-timed transcript when the input was transcribed |

Cut points are snapped to word boundaries: 0.15 s before the first word and 0.35 s after the
last, but never past the midpoint of the silence to a neighbouring word.

**Cut modes.** `--cut-mode copy` (default) is fast and lossless but starts on the nearest
earlier keyframe, so a video clip may begin slightly early. `--cut-mode exact` re-encodes
(H.264/AAC) for frame-accurate cuts. `--vertical` (scale and center-crop to 1080x1920) and
`--burn-captions` always re-encode.

## How scoring works

Candidates are runs of whole sentences lasting between `--min` and `--max` seconds. Each is
scored 0-100 as a weighted mean of five features:

| Feature | Weight | In short |
|---|---|---|
| `hook` | 0.30 | Opening sentence: question, number, curiosity/contrast word, "you", sensible length; filler openers lose points |
| `self_contained` | 0.25 | Penalises openers that point back ("that", "he", "and", "as I said") and endings that tease what's next |
| `payoff` | 0.15 | Ends on a full stop, ideally with a conclusion ("that's why", "the lesson") |
| `density` | 0.15 | 2.2-3.6 words/s, little dead air |
| `specifics` | 0.15 | Numbers and names per 10 s |

The top clips are picked greedily without overlap. Every rule, word list and threshold is
documented in [docs/scoring.md](docs/scoring.md). Tune with
`--weights hook=0.4,specifics=0.05`.

## Bring your own scorer (optional)

Plug in any function that returns 0-1 for a candidate, for example a local LLM:

```bash
shorts-clip-finder talk.srt --scorer examples/local_llm_scorer.py:score --scorer-weight 0.4
```

```python
# my_scorer.py
def score(candidate) -> float:  # candidate.text, .words, .start, .end, .sentences
    return 1.0 if "budget" in candidate.text.lower() else 0.0
```

Only the top `--scorer-top-k` (default 40) heuristic candidates are sent to your scorer. The
package itself never makes network calls. `examples/local_llm_scorer.py` talks to a local
[Ollama](https://ollama.com) server; `examples/keyword_scorer.py` is a no-dependency example.

## Python API

```python
from shorts_clip_finder import FinderConfig, find_clips, load_transcript

transcript = load_transcript("talk.srt")
for clip in find_clips(transcript, FinderConfig(min_seconds=20, max_seconds=45, top=3)):
    print(clip.rank, round(clip.start, 2), round(clip.end, 2), round(clip.score, 1), clip.title)
```

## Supported inputs

- **Media:** anything ffmpeg can read. Audio is decoded with the ffmpeg binary to 16 kHz
  mono float32 and passed to faster-whisper with word timestamps and VAD on. No
  `initial_prompt` is used, because prompts can make Whisper hallucinate text.
  Options: `--model` (default `small.en`), `--device`, `--compute-type`, `--language`,
  `--no-vad`.
- **SRT / WebVTT:** cue text is cleaned of tags; repeated rolling lines in auto-generated
  captions are dropped. Word times are interpolated inside each cue.
- **JSON:** Whisper/faster-whisper/WhisperX `{"segments": [...]}` with or without `words`,
  a bare list of segments or words, or this tool's `transcript.json`.

## Limitations

- **Heuristics, English-first.** The hook and context word lists are English. Other
  languages still transcribe (pick a multilingual model like `small`) and still get
  density/payoff/specifics, but ranking will be weaker.
- **It reads words, not the room.** It can't see the video, hear tone, or tell if something
  is funny or visually interesting.
- **Punctuation matters.** Sentence splitting depends on the transcript's punctuation
  (Whisper's is usually good; unpunctuated captions fall back to pauses).
- **SRT/VTT timing is approximate.** Without word timestamps, cut points are estimated
  within each cue; transcribe the media for word-accurate cuts.
- **Simple pronoun rule.** Any clip opening with "this/that/it/he…" is penalised, even when
  the reference is resolved in the same sentence.
- Titles are trimmed sentences, not rewritten copy. Edit before posting.

## Development

```bash
pip install -e ".[dev,whisper]"
ruff check . && ruff format --check .
pytest                 # fast suite; generates synthetic media with ffmpeg
pytest -m slow         # real faster-whisper run (needs espeak-ng or SCF_SPEECH_SAMPLE)
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE) © 2026 Gautam Goyal
