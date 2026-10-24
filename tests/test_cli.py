from __future__ import annotations

import json
from pathlib import Path

import pytest

from ankan.cli import main
from ankan.ocr.preprocess import decode_image

from .conftest import DATA, FULL_ANSWER


def test_grade_text_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    answer = tmp_path / "answer.txt"
    answer.write_text(FULL_ANSWER, encoding="utf-8")
    rubric = str(DATA / "rubric_photosynthesis.json")
    assert (
        main(["grade", "--rubric", rubric, "--text-file", str(answer), "--grader", "lexical"]) == 0
    )
    out = capsys.readouterr().out
    assert "Proposed score: 5 / 5" in out
    assert "A teacher makes the final decision." in out

    assert (
        main(
            [
                "grade",
                "--rubric",
                rubric,
                "--text-file",
                str(answer),
                "--grader",
                "lexical",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["grade"]["score"] == 5


def test_demo_sheet(tmp_path: Path) -> None:
    output = tmp_path / "sheet.png"
    assert main(["demo-sheet", str(output), "--angle", "0"]) == 0
    assert decode_image(output.read_bytes()).shape[2] == 3


def test_calibrate_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    examples = tmp_path / "labels.jsonl"
    examples.write_text(
        '{"key_point": "Oxygen is released", "answer": "oxygen is released", "label": "covered"}\n'
        '{"key_point": "Oxygen is released", "answer": "sugar is made", "label": "missed"}\n',
        encoding="utf-8",
    )
    assert main(["calibrate", str(examples), "--matcher", "lexical"]) == 0
    assert "accuracy 100.0%" in capsys.readouterr().out


def test_errors_are_reported_not_raised(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = str(tmp_path / "missing.json")
    assert main(["grade", "--rubric", missing, "--text-file", missing, "--grader", "lexical"]) == 1
    assert capsys.readouterr().err.startswith("error:")
