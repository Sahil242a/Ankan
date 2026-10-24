"""Data models shared by OCR, grading, storage and the API.

Design rule: graders only decide *coverage* (covered / partial / missed) per key point.
Marks are always computed by deterministic code from the rubric, so two runs with the
same coverage decisions can never produce different scores.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PARTIAL_CREDIT_FRACTION = 0.5
_FLOAT_TOLERANCE = 1e-6


class Coverage(str, Enum):
    COVERED = "covered"
    PARTIAL = "partial"
    MISSED = "missed"


class FlagCode(str, Enum):
    """Reasons a submission should be looked at carefully by the teacher."""

    EMPTY_ANSWER = "empty_answer"
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    BORDERLINE_KEY_POINT = "borderline_key_point"
    NEGATION_IN_EVIDENCE = "negation_in_evidence"
    UNVERIFIED_EVIDENCE = "unverified_evidence"
    GRADER_DISAGREEMENT = "grader_disagreement"
    GRADER_FALLBACK = "grader_fallback"


class SubmissionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    OVERRIDDEN = "overridden"


# --------------------------------------------------------------------------- rubric


class KeyPoint(BaseModel):
    """One idea the answer must contain to earn marks."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str = Field(min_length=3, max_length=500)
    marks: float = Field(gt=0, le=100)
    # Optional marking-scheme keywords used by the lexical matcher. Each entry is one
    # required idea; alternatives are separated by "|", e.g. "carbon dioxide|co2".
    keywords: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("keywords")
    @classmethod
    def _check_keywords(cls, value: list[str]) -> list[str]:
        cleaned = []
        for entry in value:
            alternatives = [alt.strip() for alt in entry.split("|")]
            if not all(alternatives):
                raise ValueError(f"keyword {entry!r} has an empty alternative")
            cleaned.append("|".join(alternatives))
        return cleaned


class Rubric(BaseModel):
    """A teacher-authored marking scheme for one question."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    question: str = Field(min_length=3, max_length=2000)
    model_answer: str = Field(default="", max_length=10000)  # optional; helps the LLM grader
    max_marks: float = Field(gt=0, le=100)
    key_points: list[KeyPoint] = Field(min_length=1, max_length=30)
    allow_partial: bool = True
    score_step: float = Field(default=0.5, gt=0, le=10)

    @model_validator(mode="after")
    def _check_consistency(self) -> Rubric:
        ids = [kp.id for kp in self.key_points]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate key point ids: {', '.join(duplicates)}")
        total = sum(kp.marks for kp in self.key_points)
        if not math.isclose(total, self.max_marks, abs_tol=_FLOAT_TOLERANCE):
            raise ValueError(
                f"key point marks add up to {total:g} but max_marks is {self.max_marks:g}"
            )
        return self


# --------------------------------------------------------------------------- OCR


class BoundingBox(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)


class OCRLine(BaseModel):
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    box: BoundingBox | None = None


class OCRResult(BaseModel):
    engine: str
    lines: list[OCRLine] = Field(default_factory=list)
    skew_angle: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text.strip())

    @property
    def mean_confidence(self) -> float | None:
        scores = [line.confidence for line in self.lines if line.confidence is not None]
        return sum(scores) / len(scores) if scores else None


# --------------------------------------------------------------------------- grading


class ReviewFlag(BaseModel):
    code: FlagCode
    message: str
    key_point_id: str | None = None


class KeyPointResult(BaseModel):
    key_point_id: str
    description: str
    status: Coverage
    marks_possible: float
    marks_awarded: float
    similarity: float | None = Field(default=None, ge=0, le=1)
    evidence: str | None = None
    rationale: str | None = None


class GradeResult(BaseModel):
    question_id: str
    grader: str
    score: float
    max_marks: float
    key_points: list[KeyPointResult]
    flags: list[ReviewFlag] = Field(default_factory=list)


# --------------------------------------------------------------------------- review


class TeacherDecision(BaseModel):
    """The teacher's final call. An override must carry a score and a reason."""

    model_config = ConfigDict(str_strip_whitespace=True)

    action: Literal["approve", "override"]
    reviewer: str = Field(min_length=1, max_length=100)
    final_score: float | None = Field(default=None, ge=0)
    comment: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def _check_override(self) -> TeacherDecision:
        if self.action == "override":
            if self.final_score is None:
                raise ValueError("an override needs a final_score")
            if not self.comment:
                raise ValueError("an override needs a comment explaining the change")
        return self


class AuditEntry(BaseModel):
    action: str
    actor: str
    at: datetime
    details: dict[str, object] = Field(default_factory=dict)


class Submission(BaseModel):
    id: str
    question_id: str
    student_ref: str | None
    status: SubmissionStatus
    ocr: OCRResult
    answer_text: str
    proposed: GradeResult
    final_score: float | None = None
    reviewer: str | None = None
    comment: str | None = None
    has_image: bool = False
    created_at: datetime
    decided_at: datetime | None = None
    audit: list[AuditEntry] = Field(default_factory=list)
