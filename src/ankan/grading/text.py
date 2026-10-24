"""Text helpers: normalise OCR output and cut it into candidate evidence spans."""

from __future__ import annotations

import re
import unicodedata
from itertools import pairwise

_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

NEGATIONS = frozenset(
    {"not", "no", "never", "none", "nothing", "neither", "nor", "without", "cannot"}
)

# Form fields printed or written at the top of answer sheets ("Name: ...", "Roll No.- ...").
# A label only counts when punctuation follows it, so "Class is a blueprint" is kept.
_SHEET_FIELD = re.compile(
    r"^\s*(?:name|roll\s*(?:no|number)?|reg(?:istration)?\s*(?:no|number)?|"
    r"enrol+ment\s*(?:no|number)?|seat\s*(?:no|number)|class|section|date|subject)"
    r"\s*[.:\-\u2013]",
    re.IGNORECASE,
)
# Question labels at the start of a line: "Q2. Ans:", "Q.3)", "Answer -". Kept text follows.
_QUESTION_LABEL = re.compile(
    r"^\s*(?:q(?:uestion)?\s*\.?\s*\d+[a-z]?\s*[.):\-]*\s*)?(?:ans(?:wer)?\s*[.:\-\u2013]+\s*)?",
    re.IGNORECASE,
)
# "No." and "No:" as an abbreviation for number, e.g. "Roll No. 41" or "Q No: 2".
_NUMBER_ABBREVIATION = re.compile(r"\bno\s*[.:#]|\bno\s*(?=\d)", re.IGNORECASE)


def normalize(text: str) -> str:
    """NFKC, re-join words hyphenated across lines, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return _WHITESPACE.sub(" ", text).strip()


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("\u2019", "'"))


def answer_body(text: str) -> str:
    """Drop sheet header lines (name, roll number) and question labels before grading.

    The full text is still stored and shown to the teacher; this only decides what is
    compared with the marking scheme and sent to an AI grader.
    """
    kept = []
    for line in text.splitlines():
        if _SHEET_FIELD.match(line) or not _WORD.search(line.lower()):
            continue  # a form field, or only symbols ("#", "---") misread from the margin
        kept.append(_QUESTION_LABEL.sub("", line, count=1))
    return "\n".join(kept)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(normalize(text)) if s.strip()]


def candidate_spans(text: str, max_words: int = 30) -> list[str]:
    """Sentences, windows over over-long sentences, and adjacent sentence pairs.

    Students often run ideas across sentences or skip punctuation entirely, so a key point
    is matched against several granularities and the best-scoring span wins.
    """
    if max_words < 4:
        raise ValueError("max_words must be at least 4")
    pieces: list[str] = []
    for sentence in split_sentences(text):
        tokens = sentence.split()
        if len(tokens) <= max_words:
            pieces.append(sentence)
            continue
        stride = max_words // 2
        for start in range(0, len(tokens) - stride, stride):
            pieces.append(" ".join(tokens[start : start + max_words]))

    spans = list(pieces)
    for first, second in pairwise(pieces):
        if len(first.split()) + len(second.split()) <= 2 * max_words:
            spans.append(f"{first} {second}")
    return list(dict.fromkeys(spans))  # de-duplicate, keep order


def has_new_negation(evidence: str, reference: str) -> bool:
    """True if the evidence negates something while the reference statement does not."""

    def negations(text: str) -> set[str]:
        text = _NUMBER_ABBREVIATION.sub(" ", text)
        return {w for w in words(text) if w in NEGATIONS or w.endswith("n't")}

    return bool(negations(evidence) - negations(reference))


def comparable(text: str) -> str:
    """Lower-case words only, for 'is this quote really in the answer?' checks."""
    return " ".join(words(normalize(text)))
