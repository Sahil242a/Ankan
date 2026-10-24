"""Choose similarity thresholds from teacher-labelled examples.

Input: JSON lines like {"key_point": "...", "answer": "...", "label": "covered"}
(optionally with "keywords": [...] for the lexical matcher).
For each example the best-span similarity is computed with the chosen matcher, then a
grid search picks the (partial, covered) pair that best reproduces the teacher's labels.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ankan.grading.matchers import Matcher, Thresholds
from ankan.grading.text import candidate_spans
from ankan.schemas import Coverage, KeyPoint

_CODES = {Coverage.MISSED: 0, Coverage.PARTIAL: 1, Coverage.COVERED: 2}


@dataclass(frozen=True)
class LabelledExample:
    key_point: KeyPoint
    answer: str
    label: Coverage


@dataclass(frozen=True)
class CalibrationReport:
    thresholds: Thresholds
    accuracy: float
    examples: int
    confusion: dict[str, dict[str, int]]  # confusion[true_label][predicted_label]


def load_examples(path: Path) -> list[LabelledExample]:
    examples = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key_point = KeyPoint(
                id=f"example-{number}",
                description=row["key_point"],
                marks=1,
                keywords=row.get("keywords", []),
            )
            examples.append(LabelledExample(key_point, row["answer"], Coverage(row["label"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{number}: invalid example ({exc})") from exc
    return examples


def best_span_scores(matcher: Matcher, examples: Iterable[LabelledExample]) -> list[float]:
    scores = []
    for example in examples:
        spans = candidate_spans(example.answer)
        if not spans:
            scores.append(0.0)
            continue
        scores.append(float(np.max(matcher.similarity([example.key_point], spans))))
    return scores


def _predict(scores: np.ndarray, partial: float, covered: float) -> np.ndarray:
    return np.where(scores >= covered, 2, np.where(scores >= partial, 1, 0))


def fit_thresholds(
    scores: Sequence[float], labels: Sequence[Coverage], step: float = 0.01
) -> CalibrationReport:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("need at least one labelled example")
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray([_CODES[label] for label in labels])
    grid = np.round(np.arange(step, 1.0 + 1e-9, step), 4)

    def margin(threshold: float) -> float:
        return float(np.min(np.abs(values - threshold)))

    # Maximise accuracy; among equally accurate pairs, keep both thresholds as far as possible
    # from every labelled score (max-margin, then total margin), so neither threshold is
    # pinned against the data.
    best_key, best_pair = (-1.0, -1.0, -1.0), (float(grid[0]), float(grid[1]))
    for i, partial in enumerate(grid[:-1]):
        partial_margin = margin(float(partial))
        for covered in grid[i + 1 :]:
            accuracy = float(np.mean(_predict(values, partial, covered) == truth))
            covered_margin = margin(float(covered))
            key = (
                accuracy,
                min(partial_margin, covered_margin),
                partial_margin + covered_margin,
            )
            if key > best_key:
                best_key, best_pair = key, (float(partial), float(covered))
    best_accuracy = best_key[0]

    partial, covered = best_pair
    predicted = _predict(values, partial, covered)
    names = {code: label.value for label, code in _CODES.items()}
    confusion = {
        names[t]: {names[p]: int(np.sum((truth == t) & (predicted == p))) for p in names}
        for t in names
    }
    return CalibrationReport(
        thresholds=Thresholds(covered=covered, partial=partial),
        accuracy=best_accuracy,
        examples=len(scores),
        confusion=confusion,
    )


def calibrate(matcher: Matcher, examples: Sequence[LabelledExample]) -> CalibrationReport:
    return fit_thresholds(best_span_scores(matcher, examples), [e.label for e in examples])
