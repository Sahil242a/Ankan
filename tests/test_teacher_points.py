"""Grading points written the way teachers write them, without keyword lists.

The Stack/Queue sheet is a real case: the answer deserves 3/4 because "stack of plates" is
not a queue application.
"""

from __future__ import annotations

import json

import pytest

from ankan.grading import KeyPointGrader, LexicalMatcher
from ankan.grading.matchers import parse_description, stem
from ankan.grading.text import answer_body, has_new_negation
from ankan.schemas import Coverage, FlagCode, Rubric

from .conftest import DATA


@pytest.fixture
def stack_queue() -> Rubric:
    return Rubric.model_validate(
        json.loads((DATA / "rubric_stack_queue.json").read_text(encoding="utf-8"))
    )


def grade(rubric: Rubric, text: str) -> tuple[float, str]:
    result = KeyPointGrader(LexicalMatcher()).grade(text, rubric)
    return result.score, "".join(kp.status.value[0] for kp in result.key_points)


def test_real_sheet_gets_three_out_of_four(stack_queue: Rubric) -> None:
    text = (DATA / "stack_queue_ocr.txt").read_text(encoding="utf-8")
    result = KeyPointGrader(LexicalMatcher()).grade(text, stack_queue)
    assert result.score == 3
    assert [kp.status for kp in result.key_points] == [
        Coverage.COVERED,
        Coverage.COVERED,
        Coverage.COVERED,
        Coverage.MISSED,  # "stack of plates" is not a queue application
    ]
    assert result.key_points[2].evidence == "Example of stack - undo button in MS Word."
    assert result.flags == []  # no false "close call" or "negation" from "Roll No."
    assert all("Aman" not in (kp.evidence or "") for kp in result.key_points)


@pytest.mark.parametrize(
    ("text", "score", "statuses"),
    [
        (
            "Stack works on LIFO. Queue works on FIFO. Example of stack - undo. "
            "Example of queue - printer queue.",
            4,
            "cccc",
        ),
        (  # the classic swap: wrong principles, right examples
            "Stack works on FIFO, first in first out. Queue works on LIFO, last in first out. "
            "Stack example: undo in Word. Queue example: ticket line at a cinema.",
            2,
            "mmcc",
        ),
        (
            "In a stack the last in first out rule applies. A queue is first in first out.",
            2,
            "ccmm",
        ),
        ("Stack is LIFO. Example of stack - ticket line.", 1, "cmmm"),
        ("Stack and queue are data structures.", 0, "mmmm"),
        (  # handwriting-recognition slips: O read as 0, missing letters
            "Stak works on LIF0 principle. Queue is FIF0. Exmple of stack - undo buton. "
            "Queue exampel - printer.",
            4,
            "cccc",
        ),
        ("Think of a stack. It works on LIFO.", 1, "cmmm"),
    ],
)
def test_common_answers(stack_queue: Rubric, text: str, score: float, statuses: str) -> None:
    assert grade(stack_queue, text) == (score, statuses)


def test_parse_description() -> None:
    lifo = parse_description("Stack follows LIFO (Last In First Out).")
    assert [[(alt.terms, alt.mode) for alt in group] for group in lifo.groups] == [
        [(("lifo",), "all"), (("last", "in", "first", "out"), "phrase")],
        [(("stack",), "all")],
    ]  # "follows" is how the teacher phrased it, not something to look for
    assert lifo.examples is None

    application = parse_description(
        "Gives a valid queue application, such as a printer queue, CPU scheduling, or BFS."
    )
    assert [[alt.terms for alt in group] for group in application.groups] == [[(stem("queue"),)]]
    assert application.examples is not None
    assert [alt.terms for alt in application.examples] == [
        ("printer",),
        ("cpu", "schedul"),
        ("bfs",),
    ]

    short_form = parse_description("Uses water and carbon dioxide (CO2)")
    assert [alt.terms for alt in short_form.groups[0]] == [("carbon", "dioxid"), ("co2",)]
    acronym = parse_description("The Central Processing Unit (CPU) executes instructions")
    assert acronym.groups[0][0].terms == ("central", "process", "unit")
    bracket_examples = parse_description("Inheritance reuses code (e.g. Dog extends Animal)")
    assert bracket_examples.examples is not None


def test_short_forms_in_either_direction() -> None:
    rubric = Rubric(
        question_id="cpu",
        question="What does the CPU do?",
        max_marks=2,
        key_points=[
            {
                "id": "p1",
                "description": "The Central Processing Unit (CPU) runs instructions",
                "marks": 1,
            },
            {"id": "p2", "description": "Carbon dioxide (CO2) is released", "marks": 1},
        ],
    )
    assert grade(rubric, "The CPU runs instructions. CO2 is released.")[0] == 2
    assert (
        grade(rubric, "The central processing unit runs instructions. Carbon dioxide is released.")[
            0
        ]
        == 2
    )


def test_sheet_headers_and_labels_are_not_graded() -> None:
    text = "Name: Aman Verma  Roll No.- 22CS1041\n#\nQ2. Ans: Stack is LIFO\nClass is a blueprint"
    assert answer_body(text) == "Stack is LIFO\nClass is a blueprint"


def test_number_abbreviation_is_not_a_negation() -> None:
    assert not has_new_negation("Roll No.-22CS41 Stack works", "Stack follows LIFO")
    assert not has_new_negation("Q No 5: stack", "stack")
    assert has_new_negation("There is no tilt", "The Earth is tilted")


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("use", "used"),
        ("uses", "use"),
        ("remove", "removed"),
        ("tree", "trees"),
        ("class", "classes"),
        ("process", "processing"),
        ("gas", "gases"),
        ("degree", "degrees"),
        ("agree", "agreed"),
        ("study", "studies"),
    ],
)
def test_word_forms_match(a: str, b: str) -> None:
    assert stem(a) == stem(b)


def test_teacher_keywords_still_take_priority(rubric: Rubric) -> None:
    # The photosynthesis rubric lists keywords, so the wording parser is not used.
    result = KeyPointGrader(LexicalMatcher()).grade("Plants do not release oxygen.", rubric)
    assert (FlagCode.NEGATION_IN_EVIDENCE, "oxygen") in {
        (flag.code, flag.key_point_id) for flag in result.flags
    }
