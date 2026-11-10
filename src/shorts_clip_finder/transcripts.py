"""Load transcripts (SRT, WebVTT, JSON) into a flat list of timed words.

Supported JSON shapes:

* this tool's own ``transcript.json`` (``{"words": [...]}``),
* Whisper / faster-whisper / WhisperX style ``{"segments": [{"start", "end", "text",
  "words": [{"word" | "text", "start", "end", "probability" | "score"}]}]}``,
* a bare list of segments, or a bare list of words.

When a source only has cue or segment times (SRT, VTT, segment-only JSON), word times are
interpolated across each cue in proportion to word length and the transcript is marked
``word_timing="interpolated"``.
"""

from __future__ import annotations

import codecs
import html
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import Transcript, Word

TRANSCRIPT_SUFFIXES = (".srt", ".vtt", ".json")
TRANSCRIPT_FORMAT = "shorts-clip-finder.transcript"

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})")
_TAG = re.compile(r"<[^>]*>|\{\\[^}]*\}")


class TranscriptError(ValueError):
    """Raised when a transcript file cannot be parsed."""


def parse_timestamp(text: str) -> float:
    m = _TS.search(text)
    if not m:
        raise TranscriptError(f"bad timestamp: {text!r}")
    h, mi, s, frac = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(frac.ljust(3, "0")) / 1000


def _clean_cue_text(text: str) -> str:
    text = html.unescape(_TAG.sub("", text))
    text = re.sub(r"^\s*(?:>>\s*|-(?:\s+|$))", "", text)
    return " ".join(text.split())


def interpolate_words(text: str, start: float, end: float) -> list[Word]:
    """Spread the words of *text* across [start, end] proportionally to their length."""
    parts = text.split()
    if not parts:
        return []
    end = max(end, start)
    weights = [len(p) + 1 for p in parts]
    total = sum(weights)
    out, t = [], start
    for p, w in zip(parts, weights):
        d = (end - start) * w / total
        out.append(Word(p, round(t, 3), round(t + d, 3)))
        t += d
    return out


def _parse_cues(content: str, *, vtt: bool) -> list[tuple[float, float, list[str]]]:
    cues = []
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").replace("\r", "\n").strip())
    for block in blocks:
        lines = block.split("\n")
        if vtt and lines[0].lstrip("\ufeff").startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        left, right = lines[idx].split("-->", 1)
        start = parse_timestamp(left)
        end = parse_timestamp(right.strip().split()[0])
        text_lines = [_clean_cue_text(ln) for ln in lines[idx + 1 :]]
        cues.append((start, end, [ln for ln in text_lines if ln]))
    return cues


def _words_from_cues(cues: list[tuple[float, float, list[str]]]) -> list[Word]:
    words: list[Word] = []
    prev_lines: list[str] = []
    for start, end, lines in cues:
        # Auto-generated captions (e.g. YouTube) repeat the previous line in every cue.
        new_lines = [ln for ln in lines if ln not in prev_lines]
        if lines:
            prev_lines = lines
        words.extend(interpolate_words(" ".join(new_lines), start, end))
    return words


def parse_srt(content: str) -> list[Word]:
    return _words_from_cues(_parse_cues(content, vtt=False))


def parse_vtt(content: str) -> list[Word]:
    return _words_from_cues(_parse_cues(content, vtt=True))


def _word_from_dict(d: dict[str, Any]) -> tuple[str, float | None, float | None, float | None]:
    text = str(d.get("word", d.get("text", ""))).strip()
    prob = d.get("probability", d.get("score", d.get("confidence")))
    start, end = d.get("start"), d.get("end")
    return (
        text,
        None if start is None else float(start),
        None if end is None else float(end),
        None if prob is None else float(prob),
    )


def _fill_missing(
    raw: list[tuple[str, float | None, float | None, float | None]], lo: float, hi: float
) -> list[Word]:
    """Fill words lacking timestamps (WhisperX leaves some numerals untimed)."""
    out: list[Word] = []
    n = len(raw)
    for i, (text, start, end, prob) in enumerate(raw):
        if start is None or end is None:
            prev_end = out[-1].end if out else lo
            nxt = next((r[1] for r in raw[i + 1 :] if r[1] is not None), hi)
            gap_words = 1 + next(
                (k for k, r in enumerate(raw[i + 1 :]) if r[1] is not None), n - i - 1
            )
            step = max(0.0, (nxt - prev_end) / gap_words)
            start = prev_end if start is None else start
            end = start + step if end is None else end
        out.append(Word(text, start, max(end, start), prob))
    return out


def parse_json_data(data: Any) -> tuple[list[Word], str | None, bool]:
    """Return (words, language, exact_word_timing)."""
    language = data.get("language") if isinstance(data, dict) else None
    if isinstance(data, dict) and isinstance(data.get("words"), list) and "segments" not in data:
        raw = [_word_from_dict(w) for w in data["words"]]
        exact = all(r[1] is not None for r in raw)
        lo = (raw[0][1] or 0.0) if raw else 0.0
        hi = max((r[2] for r in raw if r[2] is not None), default=lo)
        return [w for w in _fill_missing(raw, lo, hi) if w.text], language, exact
    if isinstance(data, dict):
        segments = data.get("segments")
    elif isinstance(data, list):
        segments = data
    else:
        segments = None
    if not isinstance(segments, list):
        raise TranscriptError("JSON transcript needs a 'segments' or 'words' list")
    if segments and isinstance(segments[0], dict) and "word" in segments[0]:
        return parse_json_data({"words": segments, "language": language})

    words: list[Word] = []
    exact = True
    for seg in segments:
        seg_words = seg.get("words") or []
        s0 = float(seg.get("start", words[-1].end if words else 0.0))
        s1 = float(seg.get("end", s0))
        if seg_words:
            raw = [_word_from_dict(w) for w in seg_words]
            if any(r[1] is None for r in raw):
                exact = False
            words.extend(w for w in _fill_missing(raw, s0, s1) if w.text)
        else:
            exact = False
            words.extend(interpolate_words(str(seg.get("text", "")).strip(), s0, s1))
    return words, language, exact


_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def read_transcript_text(path: str | Path) -> str:
    """Read a transcript as text: UTF-8, or UTF-16/UTF-32 when the file starts with a BOM.

    Subtitle editors on Windows often save SRT files as UTF-16 with a byte order mark;
    those are decoded transparently. Anything else that is not valid UTF-8 raises a
    TranscriptError telling the user to re-save the file as UTF-8, rather than guessing a
    legacy code page and silently garbling the words.
    """
    p = Path(path)
    data = p.read_bytes()
    encoding = next((enc for bom, enc in _BOMS if data.startswith(bom)), "utf-8")
    try:
        return data.decode(encoding)
    except UnicodeDecodeError as e:
        raise TranscriptError(
            f"{p}: not valid {encoding.removesuffix('-sig').upper()} text "
            f"(byte {e.start}); re-save the file as UTF-8 and try again"
        ) from e


def load_transcript(path: str | Path) -> Transcript:
    """Load an SRT, VTT or JSON transcript from disk."""
    p = Path(path)
    content = read_transcript_text(p)
    suffix = p.suffix.lower()
    language = None
    if suffix == ".srt":
        words, exact = parse_srt(content), False
    elif suffix == ".vtt":
        words, exact = parse_vtt(content), False
    elif suffix == ".json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise TranscriptError(f"{p}: invalid JSON ({e})") from e
        words, language, exact = parse_json_data(data)
        if isinstance(data, dict) and data.get("word_timing") == "interpolated":
            exact = False
    else:
        raise TranscriptError(f"unsupported transcript type {suffix!r} (use .srt, .vtt or .json)")
    words = sorted(words, key=lambda w: (w.start, w.end))
    if not words:
        raise TranscriptError(f"{p}: transcript contains no words")
    return Transcript(
        words=words,
        source=str(p),
        language=language,
        word_timing="exact" if exact else "interpolated",
    )


def transcript_to_dict(t: Transcript, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {
        "format": TRANSCRIPT_FORMAT,
        "version": 1,
        "source": t.source,
        "language": t.language,
        "word_timing": t.word_timing,
        "words": [_word_dict(w) for w in t.words],
    }
    if extra:
        d.update(extra)
    return d


def _word_dict(w: Word) -> dict[str, Any]:
    d: dict[str, Any] = {"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3)}
    if w.probability is not None:
        d["probability"] = round(w.probability, 3)
    return d


def save_transcript(t: Transcript, path: str | Path, extra: dict[str, Any] | None = None) -> None:
    Path(path).write_text(
        json.dumps(transcript_to_dict(t, extra), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def is_transcript_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TRANSCRIPT_SUFFIXES


def words_from_iterable(items: Iterable[tuple[str, float, float]]) -> list[Word]:
    """Convenience for tests and library users: build words from (text, start, end)."""
    return [Word(t, float(s), float(e)) for t, s, e in items]
