from __future__ import annotations

import pytest
from pydantic import ValidationError

from ankan.config import Settings
from ankan.schemas import Rubric, TeacherDecision


def test_sample_rubric_is_valid(rubric: Rubric) -> None:
    assert rubric.max_marks == sum(kp.marks for kp in rubric.key_points)
    assert [kp.id for kp in rubric.key_points][-1] == "oxygen"


def test_marks_must_add_up_to_max(rubric_data: dict[str, object]) -> None:
    rubric_data["max_marks"] = 6
    with pytest.raises(ValidationError, match="add up to 5"):
        Rubric.model_validate(rubric_data)


def test_key_point_ids_must_be_unique(rubric_data: dict[str, object]) -> None:
    points = rubric_data["key_points"]
    assert isinstance(points, list)
    points[1]["id"] = points[0]["id"]
    with pytest.raises(ValidationError, match="duplicate key point ids: location"):
        Rubric.model_validate(rubric_data)


def test_keyword_alternatives_are_trimmed_and_checked(rubric_data: dict[str, object]) -> None:
    points = rubric_data["key_points"]
    assert isinstance(points, list)
    points[0]["keywords"] = [" chloroplast | leaf "]
    assert Rubric.model_validate(rubric_data).key_points[0].keywords == ["chloroplast|leaf"]
    points[0]["keywords"] = ["chloroplast||leaf"]
    with pytest.raises(ValidationError, match="empty alternative"):
        Rubric.model_validate(rubric_data)


def test_ids_reject_path_characters(rubric_data: dict[str, object]) -> None:
    rubric_data["question_id"] = "../etc"
    with pytest.raises(ValidationError):
        Rubric.model_validate(rubric_data)


def test_override_needs_score_and_reason() -> None:
    with pytest.raises(ValidationError, match="final_score"):
        TeacherDecision(action="override", reviewer="Ms Rao", comment="harsh")
    with pytest.raises(ValidationError, match="comment"):
        TeacherDecision(action="override", reviewer="Ms Rao", final_score=3)
    decision = TeacherDecision(action="override", reviewer="Ms Rao", final_score=3, comment="ok")
    assert decision.final_score == 3


def test_decision_action_is_restricted() -> None:
    with pytest.raises(ValidationError):
        TeacherDecision(action="delete", reviewer="x")


def test_settings_validate_choices() -> None:
    with pytest.raises(ValueError, match="grader"):
        Settings(grader="magic")
    with pytest.raises(ValueError, match="ocr_engine"):
        Settings(ocr_engine="tesseract")


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANKAN_GRADER", "LEXICAL")
    monkeypatch.setenv("ANKAN_LOW_OCR_CONFIDENCE", "0.7")
    settings = Settings.from_env()
    assert settings.grader == "lexical"
    assert settings.low_ocr_confidence == 0.7
    monkeypatch.setenv("ANKAN_LOW_OCR_CONFIDENCE", "high")
    with pytest.raises(ValueError, match="must be a number"):
        Settings.from_env()


def test_missing_backend_message_gives_a_working_install_command() -> None:
    from ankan.errors import BackendUnavailableError

    message = str(BackendUnavailableError("trocr", "trocr"))
    assert message.startswith("Reading photos of handwriting needs packages")
    assert 'pip install -e ".[trocr]"' in message  # local install, not a PyPI lookup


def test_password_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANKAN_PASSWORD", "  open sesame ")
    assert Settings.from_env().password == "open sesame"
    monkeypatch.setenv("ANKAN_PASSWORD", "")
    assert Settings.from_env().password is None


def test_fast_ocr_from_env_and_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    from ankan.cli import _settings, build_parser

    monkeypatch.setenv("ANKAN_FAST_OCR", "yes")
    assert Settings.from_env().fast_ocr is True
    monkeypatch.delenv("ANKAN_FAST_OCR")
    assert Settings.from_env().fast_ocr is False
    args = build_parser().parse_args(["serve", "--fast"])
    assert _settings(args).fast_ocr is True
