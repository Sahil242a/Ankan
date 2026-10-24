"""Key-point grader: match each rubric point to its best-supporting span of the answer."""

from __future__ import annotations

import numpy as np

from ankan.grading.matchers import Matcher
from ankan.grading.scoring import Decision, build_result
from ankan.grading.text import answer_body, candidate_spans, has_new_negation
from ankan.schemas import Coverage, FlagCode, GradeResult, ReviewFlag, Rubric


class KeyPointGrader:
    """Grades by similarity between each key point and the closest part of the answer."""

    def __init__(self, matcher: Matcher, *, max_span_words: int = 30) -> None:
        self.matcher = matcher
        self.max_span_words = max_span_words

    @property
    def name(self) -> str:
        return f"keypoint-{self.matcher.name}"

    def warm_up(self) -> None:
        warm = getattr(self.matcher, "warm_up", None)
        if warm is not None:
            warm()

    def grade(self, answer_text: str, rubric: Rubric) -> GradeResult:
        spans = candidate_spans(answer_body(answer_text), self.max_span_words)
        if spans:
            scores = self.matcher.similarity(rubric.key_points, spans)
        else:
            scores = np.zeros((len(rubric.key_points), 0))

        limits = self.matcher.thresholds
        decisions: dict[str, Decision] = {}
        flags: list[ReviewFlag] = []
        for row, kp in enumerate(rubric.key_points):
            if spans:
                score = float(scores[row].max())
                # Among equally good spans, quote the shortest: it is the most precise evidence.
                tied = np.flatnonzero(scores[row] >= score - 1e-9)
                best = min(tied, key=lambda j: len(spans[j].split()))
                evidence: str | None = spans[best]
            else:
                score, evidence = 0.0, None

            if score >= limits.covered:
                status = Coverage.COVERED
            elif score >= limits.partial:
                status = Coverage.PARTIAL
            else:
                status = Coverage.MISSED

            near = min(abs(score - limits.covered), abs(score - limits.partial))
            if near < limits.borderline_margin:
                flags.append(
                    ReviewFlag(
                        code=FlagCode.BORDERLINE_KEY_POINT,
                        key_point_id=kp.id,
                        message=f"'{kp.id}' scored {score:.2f}, close to a grading threshold",
                    )
                )
            negated = evidence is not None and has_new_negation(evidence, kp.description)
            if status is not Coverage.MISSED and negated:
                flags.append(
                    ReviewFlag(
                        code=FlagCode.NEGATION_IN_EVIDENCE,
                        key_point_id=kp.id,
                        message=(
                            f"evidence for '{kp.id}' contains a negation; similarity "
                            "models can miss that the student said the opposite"
                        ),
                    )
                )
            decisions[kp.id] = Decision(
                status=status,
                similarity=score,
                evidence=evidence if status is not Coverage.MISSED else None,
            )
        return build_result(rubric, decisions, grader=self.name, flags=flags)
