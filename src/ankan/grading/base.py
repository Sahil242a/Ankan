"""The grader interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ankan.schemas import GradeResult, Rubric


@runtime_checkable
class Grader(Protocol):
    @property
    def name(self) -> str: ...

    def grade(self, answer_text: str, rubric: Rubric) -> GradeResult: ...
