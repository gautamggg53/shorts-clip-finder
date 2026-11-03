"""Command-line interface: ``shorts-clip-finder INPUT [options]``."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import __version__
from .cutter import CutOptions, cut_script, plan_cuts, run_cuts, write_caption_files
from .export import FORMATS, timecode, write_reports
from .media import MediaError, MediaInfo, probe
from .pipeline import FinderConfig, find_clips
from .plugins import load_scorer
from .scoring import parse_weights
from .transcripts import TranscriptError, is_transcript_path, load_transcript, save_transcript


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shorts-clip-finder",
        description="Find self-contained short-form clip candidates in a long voiceover or "
        "video. INPUT is a media file (transcribed locally with faster-whisper) or a "
        "transcript (.srt, .vtt, .json).",
    )
    p.add_argument("input", help="media file or transcript")
    p.add_argument(
        "-t",
        "--transcript",
        help="use this transcript instead of transcribing "
        "INPUT (INPUT is then only used for cutting clips)",
    )
    p.add_argument("-o", "--out", help="output folder (default: ./<input-name>_shorts)")
    p.add_argument("-n", "--top", type=int, default=5, help="number of clips (default 5)")
    p.add_argument(
        "--min",
        dest="min_seconds",
        type=float,
        default=30.0,
        help="minimum clip length in seconds (default 30)",
    )
    p.add_argument(
        "--max",
        dest="max_seconds",
        type=float,
        default=60.0,
        help="maximum clip length in seconds (default 60)",
    )
    p.add_argument("--pad-start", type=float, default=0.15, help="seconds before first word")
    p.add_argument("--pad-end", type=float, default=0.35, help="seconds after last word")
    p.add_argument(
        "--weights",
        default="",
        help="override feature weights, e.g. "
        "'hook=0.4,specifics=0.1' (features: hook, self_contained, payoff, "
        "density, specifics)",
    )
    p.add_argument(
        "--format",
        default=",".join(FORMATS),
        help=f"comma-separated outputs from {','.join(FORMATS)} (default: all)",
    )
    p.add_argument("--fps", type=int, default=30, help="frame rate for the EDL (default 30)")

    g = p.add_argument_group("transcription (needs the [whisper] extra)")
    g.add_argument("--model", default="small.en", help="faster-whisper model (default small.en)")
    g.add_argument("--device", default="auto", help="auto, cpu or cuda")
    g.add_argument(
        "--compute-type", default="auto", help="auto (fastest for the device), int8, float16, ..."
    )
    g.add_argument("--language", help="language code (default: en for *.en models, else auto)")
    g.add_argument("--no-vad", action="store_true", help="disable voice activity detection")

    c = p.add_argument_group("cutting (needs ffmpeg)")
    c.add_argument("--export-clips", action="store_true", help="cut the clips with ffmpeg")
    c.add_argument(
        "--cut-mode",
        choices=["copy", "exact"],
        default="copy",
        help="copy = fast stream copy (starts on a keyframe); exact = re-encode",
    )
    c.add_argument("--vertical", action="store_true", help="9:16 center crop to 1080x1920")
    c.add_argument("--burn-captions", action="store_true", help="burn word captions into video")
    c.add_argument("--caption-words", type=int, default=4, help="max words per caption cue")

    s = p.add_argument_group("custom scorer (optional, e.g. your own LLM)")
    s.add_argument("--scorer", help="'module:attr' or 'file.py:attr' returning 0-1 per clip")
    s.add_argument(
        "--scorer-weight",
        type=float,
        default=0.5,
        help="blend weight of the custom scorer (default 0.5)",
    )
    s.add_argument(
        "--scorer-top-k",
        type=int,
        default=40,
        help="only the top K heuristic candidates are sent to the custom scorer",
    )

    p.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _log(args: argparse.Namespace, msg: str) -> None:
    if not args.quiet:
        print(msg, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (TranscriptError, MediaError, ValueError, FileNotFoundError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(f"no such file: {src}")
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    bad = set(formats) - set(FORMATS)
    if bad:
        raise ValueError(f"unknown format(s): {', '.join(sorted(bad))}")
    out_dir = Path(args.out) if args.out else Path.cwd() / f"{src.stem}_shorts"
    media: Path | None = None if is_transcript_path(src) and not args.transcript else src
    if args.export_clips and media is None:
        raise ValueError("--export-clips needs a media INPUT (pass the transcript with -t)")

    config = FinderConfig(
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        top=args.top,
        pad_start=args.pad_start,
        pad_end=args.pad_end,
        weights=parse_weights(args.weights),
        scorer_weight=args.scorer_weight,
        scorer_top_k=args.scorer_top_k,
    )
    config.validate()
    extra = load_scorer(args.scorer) if args.scorer else None

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if args.transcript or is_transcript_path(src):
        transcript = load_transcript(args.transcript or src)
        _log(args, f"loaded {len(transcript.words)} words from {transcript.source}")
    else:
        from .transcribe import WhisperUnavailable, transcribe

        _log(args, f"transcribing {src} with faster-whisper '{args.model}' (local)...")
        try:
            transcript = transcribe(
                src,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                language=args.language,
                vad=not args.no_vad,
            )
        except WhisperUnavailable as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        tpath = out_dir / "transcript.json"
        save_transcript(transcript, tpath, {"model": args.model})
        _log(
            args,
            f"saved transcript to {tpath} ({time.monotonic() - t0:.0f}s); reuse it with -t {tpath}",
        )
    if transcript.word_timing == "interpolated":
        _log(
            args,
            "note: this transcript has no per-word times; word times are estimated "
            "within each cue, so cut points are approximate",
        )

    clips = find_clips(transcript, config, extra)
    if not clips:
        _log(
            args,
            f"no {args.min_seconds:g}-{args.max_seconds:g}s windows found; try a "
            "smaller --min or a longer input",
        )
        return 1

    meta = {
        "tool": "shorts-clip-finder",
        "version": __version__,
        "source": str(media or transcript.source),
        "transcript": transcript.source,
        "word_timing": transcript.word_timing,
        "min_seconds": args.min_seconds,
        "max_seconds": args.max_seconds,
        "weights": config.weights,
        "custom_scorer": args.scorer,
    }
    written = write_reports(clips, meta, out_dir, formats, fps=args.fps)

    opts = CutOptions(
        mode=args.cut_mode,
        vertical=args.vertical,
        burn_captions=args.burn_captions,
        caption_words=args.caption_words,
    )
    if media is not None and ("sh" in formats or args.export_clips):
        info = (
            probe(media)
            if args.export_clips or args.vertical or args.burn_captions
            else _guess_info(media)
        )
        plans = plan_cuts(media, clips, info, opts)
        clips_dir = out_dir / "clips"
        write_caption_files(plans, clips_dir, args.caption_words)
        if "sh" in formats:
            sh = out_dir / "cut_clips.sh"
            sh.write_text(cut_script(plans), encoding="utf-8")
            os.chmod(sh, 0o755)
            written.append(sh)
        if args.export_clips:
            _log(
                args,
                f"cutting {len(plans)} clip(s) with ffmpeg ({opts.mode}"
                f"{', vertical' if opts.vertical else ''}"
                f"{', captions' if opts.burn_captions else ''})...",
            )
            written += run_cuts(plans, clips_dir)

    if not args.quiet:
        _print_table(clips)
        print(f"\nwrote {len(written)} file(s) to {out_dir}")
    return 0


def _guess_info(media: Path) -> MediaInfo:
    """Avoid requiring ffprobe just to write a copy-mode script."""
    try:
        return probe(media)
    except MediaError:
        audio_only = media.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
        return MediaInfo(duration=None, has_video=not audio_only, has_audio=True)


def _print_table(clips: list) -> None:
    print(f"{'#':>2}  {'start':<12} {'end':<12} {'len':>5}  {'score':>5}  title")
    for c in clips:
        print(
            f"{c.rank:>2}  {timecode(c.start):<12} {timecode(c.end):<12} "
            f"{c.duration:>4.1f}s  {c.score:>5.1f}  {c.title}"
        )
        print(f"    hook: {c.hook_line}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
