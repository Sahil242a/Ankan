from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from ankan.errors import GradingError
from ankan.grading import EmbeddingMatcher, KeyPointGrader, LexicalMatcher, Thresholds
from ankan.grading.calibration import calibrate, fit_thresholds, load_examples
from ankan.grading.matchers import Matrix, phonetic, term_similarity
from ankan.grading.scoring import Decision, build_result, round_to_step
from ankan.grading.text import candidate_spans, has_new_negation, normalize, split_sentences
from ankan.schemas import Coverage, FlagCode, KeyPoint, Rubric

from .conftest import FULL_ANSWER

# --------------------------------------------------------------------------- text


def test_normalize_rejoins_hyphenated_line_breaks() -> None:
    assert normalize("photo-\n synthesis  happens\n\nhere") == "photosynthesis happens here"


def test_split_sentences() -> None:
    assert split_sentences("One. Two!  Three? Four; five") == [
        "One.",
        "Two!",
        "Three?",
        "Four;",
        "five",
    ]


def test_candidate_spans_include_pairs_and_windows() -> None:
    spans = candidate_spans("Alpha beta. Gamma delta.")
    assert spans == ["Alpha beta.", "Gamma delta.", "Alpha beta. Gamma delta."]

    long_sentence = " ".join(f"w{i}" for i in range(50))
    windows = candidate_spans(long_sentence, max_words=20)
    assert windows[0].split() == [f"w{i}" for i in range(20)]
    covered = {word for span in windows for word in span.split()}
    assert covered == {f"w{i}" for i in range(50)}  # nothing at the end is dropped
    assert candidate_spans("   ") == []


def test_negation_detection() -> None:
    assert has_new_negation("Plants do not release oxygen", "Oxygen is released")
    assert has_new_negation("Oxygen isn\u2019t released", "Oxygen is released")
    assert not has_new_negation("Oxygen is released", "Oxygen is released")
    assert not has_new_negation("No light, no food", "Without light there is no food")


# --------------------------------------------------------------------------- matchers


def test_phonetic_spelling_variants_match() -> None:
    assert phonetic("chlorophyll") == phonetic("chlorofil")
    assert term_similarity("oxygen", "oxigen") == 1.0
    assert term_similarity("glucose", "glucoze") == 1.0


def test_short_and_different_words_do_not_match() -> None:
    assert term_similarity("light", "night") == 0.0
    assert term_similarity("co2", "co") == 0.0
    assert term_similarity("chlorophyll", "chloroplast") == 0.0


def coverage(key_point: KeyPoint, span: str) -> float:
    return float(LexicalMatcher().similarity([key_point], [span])[0, 0])


def _kp(description: str, keywords: list[str] | None = None) -> KeyPoint:
    return KeyPoint(id="kp", description=description, marks=1, keywords=keywords or [])


def test_lexical_keywords_accept_any_alternative() -> None:
    point = _kp("Carbon dioxide is a raw material", ["carbon dioxide|co2"])
    assert coverage(point, "plants take in CO2") == 1.0
    assert coverage(point, "plants take in carbon dioxide") == 1.0
    assert coverage(point, "plants take in carbon") == 0.5
    assert coverage(point, "plants take in nitrogen") == 0.0


def test_lexical_without_keywords_uses_description_words() -> None:
    assert coverage(_kp("Chlorophyll absorbs light"), "chlorofil absorbed the light") == 1.0
    assert coverage(_kp("Chlorophyll absorbs light"), "") == 0.0


def test_thresholds_are_validated() -> None:
    with pytest.raises(ValueError):
        Thresholds(covered=0.4, partial=0.6)
    with pytest.raises(ValueError):
        Thresholds(covered=0.6, partial=0.4, borderline_margin=0.6)


class _FakeEncoder:
    """Maps known sentences to fixed vectors, like a sentence-transformers model would."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table
        self.calls: list[dict[str, object]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(kwargs)
        vectors = np.array([self.table[t] for t in texts], dtype=np.float64)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_embedding_matcher_uses_cosine_and_clips_negatives() -> None:
    encoder = _FakeEncoder(
        {"Point A": [1, 0], "Point B": [0, 1], "same as A": [1, 0], "opposite of B": [0, -1]}
    )
    matcher = EmbeddingMatcher(model=encoder)
    matrix = matcher.similarity([_kp("Point A"), _kp("Point B")], ["same as A", "opposite of B"])
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 1] == 0.0  # cosine -1 clipped to 0
    assert encoder.calls[0]["normalize_embeddings"] is True
    assert matcher.similarity([_kp("Point A")], []).shape == (1, 0)


# --------------------------------------------------------------------------- scoring


@pytest.mark.parametrize(
    ("value", "step", "expected"),
    [(3.24, 0.5, 3.0), (3.25, 0.5, 3.5), (3.75, 0.5, 4.0), (2.5, 1, 3.0), (0.1 + 0.2, 0.1, 0.3)],
)
def test_round_to_step(value: float, step: float, expected: float) -> None:
    assert round_to_step(value, step) == expected


def test_partial_credit_and_rounding(rubric: Rubric) -> None:
    decisions = {kp.id: Decision(status=Coverage.PARTIAL) for kp in rubric.key_points}
    result = build_result(rubric, decisions, grader="test")
    assert [kp.marks_awarded for kp in result.key_points] == [0.5] * 5
    assert result.score == 2.5


def test_partial_counts_as_missed_when_not_allowed(rubric: Rubric) -> None:
    strict = rubric.model_copy(update={"allow_partial": False})
    decisions = {kp.id: Decision(status=Coverage.PARTIAL) for kp in strict.key_points}
    result = build_result(strict, decisions, grader="test")
    assert result.score == 0
    assert {kp.status for kp in result.key_points} == {Coverage.MISSED}


def test_decisions_must_cover_the_rubric(rubric: Rubric) -> None:
    with pytest.raises(GradingError, match="missing"):
        build_result(rubric, {"location": Decision(status=Coverage.COVERED)}, grader="test")


# --------------------------------------------------------------------------- key point grader


def test_complete_answer_gets_full_marks(rubric: Rubric) -> None:
    result = KeyPointGrader(LexicalMatcher()).grade(FULL_ANSWER, rubric)
    assert result.score == 5
    assert all(kp.evidence for kp in result.key_points)
    assert result.flags == []


def test_answer_with_ocr_misspellings(rubric: Rubric) -> None:
    answer = (
        "Photosynthesis hapens in the chloroplasts of leaves. The green pigment chlorofil "
        "absorbs light from the sun.\nPlants take in water and carbon dioxide to make glucoze."
    )
    result = KeyPointGrader(LexicalMatcher()).grade(answer, rubric)
    assert result.score == 4
    assert [kp.key_point_id for kp in result.key_points if kp.status is Coverage.MISSED] == [
        "oxygen"
    ]


def test_blank_answer_scores_zero(rubric: Rubric) -> None:
    result = KeyPointGrader(LexicalMatcher()).grade("", rubric)
    assert result.score == 0
    assert all(kp.status is Coverage.MISSED and kp.evidence is None for kp in result.key_points)


def test_negated_evidence_is_flagged(rubric: Rubric) -> None:
    result = KeyPointGrader(LexicalMatcher()).grade("Plants do not release oxygen.", rubric)
    codes = {(flag.code, flag.key_point_id) for flag in result.flags}
    assert (FlagCode.NEGATION_IN_EVIDENCE, "oxygen") in codes


class _ScriptedMatcher:
    name = "scripted"
    thresholds = Thresholds(covered=0.6, partial=0.4, borderline_margin=0.05)

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def similarity(self, key_points: Sequence[KeyPoint], spans: Sequence[str]) -> Matrix:
        return np.array([[s] * len(spans) for s in self.scores], dtype=np.float64)


def test_thresholds_and_borderline_flags(rubric: Rubric) -> None:
    grader = KeyPointGrader(_ScriptedMatcher([0.9, 0.62, 0.5, 0.38, 0.1]))
    result = grader.grade("Some answer text.", rubric)
    assert [kp.status for kp in result.key_points] == [
        Coverage.COVERED,
        Coverage.COVERED,
        Coverage.PARTIAL,
        Coverage.MISSED,
        Coverage.MISSED,
    ]
    assert result.score == 2.5
    borderline = [f.key_point_id for f in result.flags if f.code is FlagCode.BORDERLINE_KEY_POINT]
    assert borderline == ["pigment", "glucose"]


# --------------------------------------------------------------------------- calibration


def test_fit_thresholds_separates_labels() -> None:
    scores = [0.1, 0.2, 0.45, 0.5, 0.8, 0.9]
    labels = [Coverage.MISSED] * 2 + [Coverage.PARTIAL] * 2 + [Coverage.COVERED] * 2
    report = fit_thresholds(scores, labels)
    assert report.accuracy == 1.0
    # Max-margin tie-break: each threshold sits in the middle of its gap between labels.
    assert report.thresholds.partial == pytest.approx(0.325, abs=0.006)
    assert report.thresholds.covered == pytest.approx(0.65, abs=0.006)
    assert report.confusion["covered"]["covered"] == 2


def test_fit_thresholds_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        fit_thresholds([], [])
    with pytest.raises(ValueError):
        fit_thresholds([0.5], [])


def test_calibrate_from_file(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"key_point": "Oxygen is released", "answer": "oxygen is given out", '
        '"keywords": ["oxygen", "released|given out"], "label": "covered"}\n'
        "\n"
        '{"key_point": "Oxygen is released", "answer": "it makes sugar", "label": "missed"}\n',
        encoding="utf-8",
    )
    examples = load_examples(path)
    assert len(examples) == 2
    report = calibrate(LexicalMatcher(), examples)
    assert report.accuracy == 1.0


def test_load_examples_reports_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text('{"key_point": "x y z", "answer": "a", "label": "maybe"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":1: invalid example"):
        load_examples(path)


def test_evidence_prefers_the_shortest_equally_good_span(rubric: Rubric) -> None:
    answer = "Oxygen is released. Leaves are green."
    result = KeyPointGrader(LexicalMatcher()).grade(answer, rubric)
    oxygen = next(kp for kp in result.key_points if kp.key_point_id == "oxygen")
    assert oxygen.evidence == "Oxygen is released."  # not the two-sentence pair span


def test_failed_embedding_download_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    def offline(*args: object, **kwargs: object) -> object:
        raise RuntimeError("403 Forbidden")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=offline)
    )
    with pytest.raises(GradingError, match="check the internet connection"):
        EmbeddingMatcher().similarity([_kp("Point A")], ["an answer"])
