from __future__ import annotations

import pytest

from ankan.errors import ConflictError, NotFoundError
from ankan.grading import KeyPointGrader, LexicalMatcher
from ankan.ocr.preprocess import encode_png
from ankan.pipeline import GradingPipeline
from ankan.schemas import FlagCode, OCRResult, Rubric, SubmissionStatus, TeacherDecision
from ankan.store import ReviewStore
from ankan.synthetic import render_sheet

from .conftest import FULL_ANSWER, FakeOCR

# --------------------------------------------------------------------------- pipeline


def test_image_pipeline_deskews_before_ocr(rubric: Rubric) -> None:
    ocr = FakeOCR()
    pipeline = GradingPipeline(ocr, KeyPointGrader(LexicalMatcher()))
    result = pipeline.grade_image(encode_png(render_sheet(["Some writing here"], angle=4)), rubric)
    assert result.ocr.skew_angle == pytest.approx(-4, abs=0.3)
    assert ocr.pages[0] is result.page
    assert result.grade.score == 5
    assert result.grade.flags == []


def test_low_confidence_lines_are_flagged(rubric: Rubric) -> None:
    ocr = FakeOCR([("Photosynthesis happens in the chloroplasts.", 0.95), ("smudged text", 0.31)])
    pipeline = GradingPipeline(ocr, KeyPointGrader(LexicalMatcher()), low_ocr_confidence=0.6)
    result = pipeline.grade_image(encode_png(render_sheet(["x"])), rubric)
    (flag,) = [f for f in result.grade.flags if f.code is FlagCode.LOW_OCR_CONFIDENCE]
    assert "line(s) 2" in flag.message


def test_blank_answer_is_flagged(rubric: Rubric) -> None:
    pipeline = GradingPipeline(FakeOCR([]), KeyPointGrader(LexicalMatcher()))
    grade = pipeline.grade_text("", rubric)
    assert grade.score == 0
    assert grade.flags[0].code is FlagCode.EMPTY_ANSWER


# --------------------------------------------------------------------------- store


def _submit(store: ReviewStore, rubric: Rubric, text: str, pipeline: GradingPipeline):  # type: ignore[no-untyped-def]
    _, version = store.get_rubric(rubric.question_id)
    return store.create_submission(
        rubric=rubric,
        rubric_version=version,
        student_ref="roll-17",
        ocr=OCRResult(engine="typed"),
        answer_text=text,
        proposed=pipeline.grade_text(text, rubric),
    )


def test_rubric_versions_increase(store: ReviewStore, rubric: Rubric) -> None:
    assert store.save_rubric(rubric) == 1
    assert store.save_rubric(rubric) == 2
    saved, version = store.get_rubric(rubric.question_id)
    assert (saved, version) == (rubric, 2)
    with pytest.raises(NotFoundError):
        store.get_rubric("unknown")


def test_approve_flow_and_audit(
    store: ReviewStore, rubric: Rubric, pipeline: GradingPipeline
) -> None:
    store.save_rubric(rubric)
    sub = _submit(store, rubric, FULL_ANSWER, pipeline)
    assert sub.status is SubmissionStatus.PENDING
    decided = store.decide(sub.id, TeacherDecision(action="approve", reviewer="Ms Rao"))
    assert decided.status is SubmissionStatus.APPROVED
    assert decided.final_score == 5
    assert [a.action for a in decided.audit] == ["graded", "approved"]
    assert decided.audit[1].actor == "Ms Rao"
    with pytest.raises(ConflictError, match="already approved"):
        store.decide(sub.id, TeacherDecision(action="approve", reviewer="Mr Sen"))


def test_override_is_bounded_by_max_marks(
    store: ReviewStore, rubric: Rubric, pipeline: GradingPipeline
) -> None:
    store.save_rubric(rubric)
    sub = _submit(store, rubric, "Plants need water.", pipeline)
    too_high = TeacherDecision(action="override", reviewer="t", final_score=6, comment="x")
    with pytest.raises(ValueError, match="exceeds max marks"):
        store.decide(sub.id, too_high)
    ok = TeacherDecision(action="override", reviewer="t", final_score=2, comment="diagram shown")
    decided = store.decide(sub.id, ok)
    assert decided.status is SubmissionStatus.OVERRIDDEN
    assert (decided.final_score, decided.comment) == (2, "diagram shown")


def test_regrade_only_while_pending(
    store: ReviewStore, rubric: Rubric, pipeline: GradingPipeline
) -> None:
    store.save_rubric(rubric)
    sub = _submit(store, rubric, "Plants need water.", pipeline)
    regraded = store.regrade(
        sub.id,
        answer_text=FULL_ANSWER,
        proposed=pipeline.grade_text(FULL_ANSWER, rubric),
        rubric=rubric,
        rubric_version=1,
        actor="Ms Rao",
    )
    assert regraded.proposed.score == 5
    assert [a.action for a in regraded.audit] == ["graded", "regraded"]
    store.decide(sub.id, TeacherDecision(action="approve", reviewer="Ms Rao"))
    with pytest.raises(ConflictError):
        store.regrade(
            sub.id,
            answer_text="",
            proposed=regraded.proposed,
            rubric=rubric,
            rubric_version=1,
            actor="x",
        )
    with pytest.raises(NotFoundError):
        store.decide("0" * 32, TeacherDecision(action="approve", reviewer="x"))


def test_queue_puts_flagged_pending_answers_first(
    store: ReviewStore, rubric: Rubric, pipeline: GradingPipeline
) -> None:
    store.save_rubric(rubric)
    clean = _submit(store, rubric, FULL_ANSWER, pipeline)
    flagged = _submit(store, rubric, "", pipeline)
    done = _submit(store, rubric, FULL_ANSWER, pipeline)
    store.decide(done.id, TeacherDecision(action="approve", reviewer="t"))
    order = [s.id for s in store.list_submissions()]
    assert order == [flagged.id, clean.id, done.id]
    pending = store.list_submissions(status=SubmissionStatus.PENDING)
    assert {s.id for s in pending} == {flagged.id, clean.id}


def test_page_image_is_stored_and_paths_are_safe(
    store: ReviewStore, rubric: Rubric, pipeline: GradingPipeline
) -> None:
    store.save_rubric(rubric)
    sub = store.create_submission(
        rubric=rubric,
        rubric_version=1,
        student_ref=None,
        ocr=OCRResult(engine="typed"),
        answer_text=FULL_ANSWER,
        proposed=pipeline.grade_text(FULL_ANSWER, rubric),
        page_png=b"\x89PNG fake",
    )
    assert store.image_path(sub.id).read_bytes() == b"\x89PNG fake"
    with pytest.raises(NotFoundError):
        store.image_path("../../etc/passwd")


def test_failed_insert_leaves_no_orphan_image(
    store: ReviewStore, rubric: Rubric, pipeline: GradingPipeline
) -> None:
    # The rubric was never saved, so the foreign key rejects the submission.
    with pytest.raises(Exception, match="FOREIGN KEY"):
        store.create_submission(
            rubric=rubric,
            rubric_version=1,
            student_ref=None,
            ocr=OCRResult(engine="typed"),
            answer_text="x",
            proposed=pipeline.grade_text("x", rubric),
            page_png=b"png",
        )
    assert list(store.image_dir.iterdir()) == []


def test_delete_question_removes_its_answers_and_images(
    store: ReviewStore, rubric: Rubric, pipeline: GradingPipeline
) -> None:
    store.save_rubric(rubric)
    other = rubric.model_copy(update={"question_id": "other-q"})
    store.save_rubric(other)
    kept = _submit(store, other, FULL_ANSWER, pipeline)
    with_image = store.create_submission(
        rubric=rubric,
        rubric_version=1,
        student_ref=None,
        ocr=OCRResult(engine="typed"),
        answer_text=FULL_ANSWER,
        proposed=pipeline.grade_text(FULL_ANSWER, rubric),
        page_png=b"png",
    )
    decided = _submit(store, rubric, FULL_ANSWER, pipeline)
    store.decide(decided.id, TeacherDecision(action="approve", reviewer="t"))

    assert store.delete_rubric(rubric.question_id) == 2
    assert [r.question_id for r in store.list_rubrics()] == ["other-q"]
    assert [s.id for s in store.list_submissions()] == [kept.id]
    assert not (store.image_dir / f"{with_image.id}.png").exists()
    with pytest.raises(NotFoundError):
        store.delete_rubric(rubric.question_id)
