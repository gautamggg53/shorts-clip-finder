# Contributing

Thanks for helping. Bug reports, transcript samples that score badly, and small focused pull
requests are all welcome.

## Setup

```bash
git clone https://github.com/gautamggg53/shorts-clip-finder
cd shorts-clip-finder
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # add ,whisper to work on transcription
```

You also need `ffmpeg` and `ffprobe` on your PATH for the integration tests.

## Checks

```bash
ruff check . && ruff format --check .
pytest                      # fast suite, no ML models needed
pytest -m slow              # real faster-whisper run (needs [whisper] and espeak-ng
                            # or SCF_SPEECH_SAMPLE=/path/to/speech.wav)
```

## Changing the scoring

Scoring lives in `src/shorts_clip_finder/scoring.py` and is documented in
`docs/scoring.md`. Please:

- keep every rule explainable in one sentence and add a note string so users can see why a
  clip scored the way it did;
- add a test in `tests/test_scoring.py` with example sentences that should move up or down;
- update `docs/scoring.md`.

The fixture transcripts in `tests/fixtures/` are generated from `talk.txt` by
`make_fixtures.py`. Regenerate them rather than editing the SRT/VTT/JSON by hand.

## Ground rules

- No network calls or API keys in the package. LLM scoring belongs in user-supplied scorers.
- Don't commit audio or video. Tests generate their own media with ffmpeg.
- Be kind in issues and reviews.
