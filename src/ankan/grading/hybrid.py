"""Hybrid grader: the LLM proposes, the embedding grader cross-checks.

Two independent methods agreeing is evidence the proposal is sound; disagreement is
exactly the case a teacher should look at first. If the LLM is unavailable, the
similarity grade is used and the submission is flagged.
"""

from __future__ import annotations

from ankan.errors import GradingError
from ankan.grading.base import Grader
from ankan.schemas import Coverage, FlagCode, GradeResult, ReviewFlag, Rubric

_LEVEL = {Coverage.MISSED: 0, Coverage.PARTIAL: 1, Coverage.COVERED: 2}


class HybridGrader:
    def __init__(
        self, primary: Grader, check: Grader, *, disagreement_fraction: float = 0.25
    ) -> None:
        if not 0 < disagreement_fraction <= 1:
            raise ValueError("disagreement_fraction must be in (0, 1]")
        self.primary = primary
        self.check = check
        self.disagreement_fraction = disagreement_fraction

    @property
    def name(self) -> str:
        return f"hybrid({self.primary.name}+{self.check.name})"

    def warm_up(self) -> None:
        warm = getattr(self.check, "warm_up", None)
        if warm is not None:
            warm()

    def grade(self, answer_text: str, rubric: Rubric) -> GradeResult:
        check = self.check.grade(answer_text, rubric)
        try:
            primary = self.primary.grade(answer_text, rubric)
        except GradingError as exc:
            fallback = ReviewFlag(
                code=FlagCode.GRADER_FALLBACK,
                message=f"{self.primary.name} grader failed ({exc}); showing {check.grader} grade",
            )
            return check.model_copy(update={"flags": [*check.flags, fallback]})

        similarity = {kp.key_point_id: kp.similarity for kp in check.key_points}
        status = {kp.key_point_id: kp.status for kp in check.key_points}
        merged = [
            kp.model_copy(update={"similarity": similarity[kp.key_point_id]})
            for kp in primary.key_points
        ]
        flags = [*primary.flags]
        # Similarity-only flags (borderline/negation) are about the check grader's own
        # thresholds, so they are not copied; disagreement summarises them instead.
        far_apart = [
            kp.key_point_id
            for kp in primary.key_points
            if abs(_LEVEL[kp.status] - _LEVEL[status[kp.key_point_id]]) == 2
        ]
        gap = abs(primary.score - check.score)
        if gap > self.disagreement_fraction * rubric.max_marks or far_apart:
            detail = f"; opposite calls on {', '.join(far_apart)}" if far_apart else ""
            flags.append(
                ReviewFlag(
                    code=FlagCode.GRADER_DISAGREEMENT,
                    message=(
                        f"{primary.grader} gave {primary.score:g} but {check.grader} gave "
                        f"{check.score:g}{detail}"
                    ),
                )
            )
        return primary.model_copy(
            update={"grader": self.name, "key_points": merged, "flags": flags}
        )
