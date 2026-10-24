"""Turn per-key-point coverage decisions into marks. The only place marks are computed."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ankan.errors import GradingError
from ankan.schemas import (
    PARTIAL_CREDIT_FRACTION,
    Coverage,
    GradeResult,
    KeyPoint,
    KeyPointResult,
    ReviewFlag,
    Rubric,
)


@dataclass(frozen=True)
class Decision:
    status: Coverage
    similarity: float | None = None
    evidence: str | None = None
    rationale: str | None = None


def marks_for(status: Coverage, key_point: KeyPoint, *, allow_partial: bool) -> float:
    if status is Coverage.COVERED:
        return key_point.marks
    if status is Coverage.PARTIAL and allow_partial:
        return key_point.marks * PARTIAL_CREDIT_FRACTION
    return 0.0


def round_to_step(value: float, step: float) -> float:
    """Round half up to the nearest multiple of `step` (e.g. 0.5 marks)."""
    if step <= 0:
        raise ValueError("step must be positive")
    rounded = math.floor(value / step + 0.5 + 1e-9) * step
    return round(rounded, 6)


def build_result(
    rubric: Rubric,
    decisions: Mapping[str, Decision],
    *,
    grader: str,
    flags: Sequence[ReviewFlag] = (),
) -> GradeResult:
    expected = {kp.id for kp in rubric.key_points}
    if set(decisions) != expected:
        missing = sorted(expected - set(decisions))
        extra = sorted(set(decisions) - expected)
        raise GradingError(f"decisions do not match rubric (missing={missing}, extra={extra})")

    results: list[KeyPointResult] = []
    for kp in rubric.key_points:
        decision = decisions[kp.id]
        status = decision.status
        if status is Coverage.PARTIAL and not rubric.allow_partial:
            status = Coverage.MISSED
        results.append(
            KeyPointResult(
                key_point_id=kp.id,
                description=kp.description,
                status=status,
                marks_possible=kp.marks,
                marks_awarded=marks_for(status, kp, allow_partial=rubric.allow_partial),
                similarity=None if decision.similarity is None else round(decision.similarity, 4),
                evidence=decision.evidence,
                rationale=decision.rationale,
            )
        )
    raw_total = sum(r.marks_awarded for r in results)
    score = min(rubric.max_marks, round_to_step(raw_total, rubric.score_step))
    return GradeResult(
        question_id=rubric.question_id,
        grader=grader,
        score=score,
        max_marks=rubric.max_marks,
        key_points=results,
        flags=list(flags),
    )
