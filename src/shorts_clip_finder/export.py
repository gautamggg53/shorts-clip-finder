"""Write clip lists as JSON, CSV, Markdown, CMX3600 EDL, and per-clip SRT captions."""

from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .models import Clip, Word, join_words
from .scoring import FEATURES


def timecode(seconds: float, sep: str = ".") -> str:
    """``HH:MM:SS.mmm`` (``sep=","`` for SRT)."""
    ms = round(max(seconds, 0.0) * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def frames_timecode(seconds: float, fps: int) -> str:
    """Non-drop-frame ``HH:MM:SS:FF`` for EDLs."""
    total = round(max(seconds, 0.0) * fps)
    f = total % fps
    s = total // fps
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{f:02d}"


def slugify(text: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > max_len:  # cut at a word boundary when possible
        head = slug[: max_len + 1]
        cut = head.rsplit("-", 1)[0] if "-" in head else ""
        slug = cut if len(cut) >= max_len // 2 else slug[:max_len]
    return slug.rstrip("-") or "clip"


def clip_basename(clip: Clip) -> str:
    return f"{clip.rank:02d}-{slugify(clip.title)}"


def results_dict(clips: Sequence[Clip], meta: dict[str, Any]) -> dict[str, Any]:
    return {**meta, "clips": [c.to_dict() for c in clips]}


def to_json(clips: Sequence[Clip], meta: dict[str, Any]) -> str:
    return json.dumps(results_dict(clips, meta), ensure_ascii=False, indent=2) + "\n"


def to_csv(clips: Sequence[Clip]) -> str:
    feats = list(FEATURES) + (
        ["custom"] if any("custom" in c.candidate.features for c in clips) else []
    )
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        [
            "rank",
            "start",
            "end",
            "start_tc",
            "end_tc",
            "duration",
            "score",
            *feats,
            "title",
            "hook_line",
            "text",
        ]
    )
    for c in clips:
        f = c.candidate.features
        w.writerow(
            [
                c.rank,
                f"{c.start:.3f}",
                f"{c.end:.3f}",
                timecode(c.start),
                timecode(c.end),
                f"{c.duration:.2f}",
                f"{c.score:.1f}",
                *[f"{f[k]:.3f}" if k in f else "" for k in feats],
                c.title,
                c.hook_line,
                c.candidate.text,
            ]
        )
    return buf.getvalue()


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(clips: Sequence[Clip], meta: dict[str, Any]) -> str:
    lines = [f"# Clip candidates: {Path(str(meta.get('source', ''))).name}", ""]
    lines.append(
        f"Found {len(clips)} clip(s) between {meta.get('min_seconds')}s and "
        f"{meta.get('max_seconds')}s. Scores are heuristic (0-100); review before "
        "posting."
    )
    lines += ["", "| # | Start | End | Length | Score | Title |", "|---|---|---|---|---|---|"]
    for c in clips:
        lines.append(
            f"| {c.rank} | {timecode(c.start)} | {timecode(c.end)} | "
            f"{c.duration:.1f}s | {c.score:.1f} | {_md_cell(c.title)} |"
        )
    for c in clips:
        f = c.candidate.features
        lines += [
            "",
            f"## {c.rank}. {c.title}",
            "",
            f"- **Cut:** {timecode(c.start)} → {timecode(c.end)} ({c.duration:.1f}s)",
            f"- **Score:** {c.score:.1f} — " + ", ".join(f"{k} {v:.2f}" for k, v in f.items()),
            f"- **Hook line:** {c.hook_line}",
        ]
        if c.alt_hook:
            lines.append(f"- **Stronger hook inside the clip:** {c.alt_hook}")
        if c.candidate.notes:
            lines.append(f"- **Notes:** {'; '.join(c.candidate.notes)}")
        lines += ["", f"> {c.candidate.text}"]
    return "\n".join(lines) + "\n"


def to_edl(
    clips: Sequence[Clip], source_name: str, fps: int = 30, title: str = "shorts-clip-finder"
) -> str:
    """CMX3600 EDL with one event per clip, laid end to end on the record side."""
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    rec = 0  # record-side position in frames
    for c in clips:
        src_in, src_out = round(c.start * fps), round(c.end * fps)
        rec_out = rec + (src_out - src_in)
        lines.append(
            f"{c.rank:03d}  AX       AA/V  C        {frames_timecode(src_in / fps, fps)} "
            f"{frames_timecode(src_out / fps, fps)} {frames_timecode(rec / fps, fps)} "
            f"{frames_timecode(rec_out / fps, fps)}"
        )
        lines.append(f"* FROM CLIP NAME: {source_name}")
        lines.append(f"* COMMENT: {c.title}")
        lines.append("")
        rec = rec_out
    return "\n".join(lines)


def caption_cues(
    words: Sequence[Word],
    offset: float,
    *,
    max_words: int = 4,
    max_chars: int = 32,
    clip_end: float | None = None,
) -> list[tuple[float, float, str]]:
    """Group words into short caption cues with times relative to *offset*."""
    cues: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        text_len = len(join_words([*cur, w]))
        if cur and (len(cur) >= max_words or text_len > max_chars):
            cues.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?…][\"'”’)]*$", w.text):
            cues.append(cur)
            cur = []
    if cur:
        cues.append(cur)
    out = []
    for i, cue in enumerate(cues):
        start = max(0.0, cue[0].start - offset)
        end = cue[-1].end - offset
        if i + 1 < len(cues):
            nxt = cues[i + 1][0].start - offset
            if nxt - end < 0.3:  # close tiny gaps so captions don't flicker
                end = nxt
        elif clip_end is not None:
            end = min(end + 0.3, clip_end - offset)  # hold the last caption briefly
        out.append((round(start, 3), round(max(end, start + 0.05), 3), join_words(cue)))
    return out


def to_srt(cues: Sequence[tuple[float, float, str]]) -> str:
    blocks = []
    for i, (s, e, text) in enumerate(cues, 1):
        blocks.append(f"{i}\n{timecode(s, ',')} --> {timecode(e, ',')}\n{text}\n")
    return "\n".join(blocks)


def clip_srt(clip: Clip, *, max_words: int = 4) -> str:
    return to_srt(
        caption_cues(clip.candidate.words, clip.start, max_words=max_words, clip_end=clip.end)
    )


FORMATS = ("json", "csv", "md", "edl", "sh")


def write_reports(
    clips: Sequence[Clip],
    meta: dict[str, Any],
    out_dir: Path,
    formats: Sequence[str],
    fps: int = 30,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    source_name = Path(str(meta.get("source", "source"))).name
    contents = {
        "json": lambda: to_json(clips, meta),
        "csv": lambda: to_csv(clips),
        "md": lambda: to_markdown(clips, meta),
        "edl": lambda: to_edl(clips, source_name, fps),
    }
    names = {"json": "clips.json", "csv": "clips.csv", "md": "clips.md", "edl": "clips.edl"}
    for fmt in formats:
        if fmt in contents:
            p = out_dir / names[fmt]
            p.write_text(contents[fmt](), encoding="utf-8")
            written.append(p)
    return written
