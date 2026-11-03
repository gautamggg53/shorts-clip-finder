# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-03

### Added
- `shorts-clip-finder` CLI and Python library.
- Inputs: any ffmpeg-readable media (local transcription with faster-whisper, optional
  `[whisper]` extra) or an existing SRT, WebVTT or JSON transcript (Whisper, faster-whisper,
  WhisperX and the tool's own `transcript.json`).
- Sentence segmentation with abbreviation handling, pause splitting and long-sentence splitting.
- Heuristic scoring: hook strength, self-containedness, payoff, speech density, specifics.
  Weights are configurable with `--weights`.
- Non-overlapping top-N selection, cut points snapped to word boundaries with padding.
- Heuristic hook line, stronger-hook suggestion and on-screen title (no LLM).
- Optional pluggable scorer (`--scorer module:attr` or `file.py:attr`).
- Exports: JSON, CSV, Markdown, CMX3600 EDL, `cut_clips.sh` ffmpeg script, per-clip SRT.
- `--export-clips` with stream copy or exact re-encode, `--vertical` 9:16 center crop and
  `--burn-captions`.

[0.1.0]: https://github.com/gautamggg53/shorts-clip-finder/releases/tag/v0.1.0
