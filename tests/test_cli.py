import json

import pytest

from shorts_clip_finder import __version__
from shorts_clip_finder.cli import main

ARGS = ["--min", "10", "--max", "25", "-n", "3"]


def test_transcript_only_run(tmp_path, fixtures, capsys):
    out = tmp_path / "out"
    rc = main([str(fixtures / "talk.vtt"), "-o", str(out), *ARGS])
    assert rc == 0
    assert {p.name for p in out.iterdir()} == {"clips.json", "clips.csv", "clips.md", "clips.edl"}
    data = json.loads((out / "clips.json").read_text())
    assert data["word_timing"] == "interpolated" and len(data["clips"]) == 3
    captured = capsys.readouterr()
    assert "hook:" in captured.out and "approximate" in captured.err


def test_formats_and_quiet(tmp_path, fixtures, capsys):
    out = tmp_path / "o"
    assert main([str(fixtures / "talk.srt"), "-o", str(out), "--format", "json", "-q", *ARGS]) == 0
    assert [p.name for p in out.iterdir()] == ["clips.json"]
    assert capsys.readouterr().out == ""


def test_default_out_dir(tmp_path, fixtures, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main([str(fixtures / "talk.srt"), "-q", *ARGS]) == 0
    assert (tmp_path / "talk_shorts" / "clips.json").exists()


def test_custom_scorer_from_file(tmp_path, fixtures):
    plugin = tmp_path / "myscorer.py"
    plugin.write_text(
        "class Scorer:\n"
        "    def score(self, candidate):\n"
        "        return 1.0 if 'credit' in candidate.text.lower() else 0.0\n"
    )
    out = tmp_path / "o"
    rc = main(
        [
            str(fixtures / "talk_whisper.json"),
            "-o",
            str(out),
            "--format",
            "json",
            "-q",
            "--scorer",
            f"{plugin}:Scorer",
            "--scorer-weight",
            "0.9",
            *ARGS,
        ]
    )
    assert rc == 0
    data = json.loads((out / "clips.json").read_text())
    assert "custom" in data["clips"][0]["breakdown"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--format", "pdf"],
        ["--weights", "magic=1"],
        ["--export-clips"],
        ["--min", "-1"],
        ["--scorer", "nocolon"],
        ["--scorer", "missing_file.py:x"],
    ],
)
def test_errors_exit_2(tmp_path, fixtures, extra, capsys):
    rc = main([str(fixtures / "talk.srt"), "-o", str(tmp_path / "o"), *extra])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_missing_input(tmp_path):
    assert main([str(tmp_path / "nope.srt")]) == 2


def test_no_clips_found_exit_1(tmp_path, fixtures):
    assert (
        main(
            [str(fixtures / "talk.srt"), "-o", str(tmp_path), "--min", "500", "--max", "600", "-q"]
        )
        == 1
    )


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_shipped_example_scorer(tmp_path, fixtures):
    example = fixtures.parent.parent / "examples" / "keyword_scorer.py"
    rc = main(
        [
            str(fixtures / "talk.srt"),
            "-o",
            str(tmp_path),
            "--format",
            "json",
            "-q",
            "--scorer",
            f"{example}:KeywordScorer",
            *ARGS,
        ]
    )
    assert rc == 0


def test_fractional_fps_for_edl(tmp_path, fixtures):
    out = tmp_path / "o"
    rc = main([str(fixtures / "talk.srt"), "-o", str(out), "--fps", "29.97", "-q", *ARGS])
    assert rc == 0
    assert "FCM: NON-DROP FRAME" in (out / "clips.edl").read_text()


@pytest.mark.parametrize("bad", ["0", "abc"])
def test_bad_fps_is_a_usage_error(tmp_path, fixtures, capsys, bad):
    with pytest.raises(SystemExit) as e:
        main([str(fixtures / "talk.srt"), "-o", str(tmp_path / "o"), "--fps", bad])
    assert e.value.code == 2 and "--fps" in capsys.readouterr().err
