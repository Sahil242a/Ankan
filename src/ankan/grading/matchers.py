"""Similarity backends that score how well an answer span expresses a key point.

* EmbeddingMatcher - sentence-transformers cosine similarity; understands paraphrase.
* LexicalMatcher   - fuzzy, spelling-tolerant keyword coverage; needs no model download
                     and every score can be explained word by word.

Thresholds live with the matcher because each backend has its own similarity scale.
Use `ankan calibrate` on teacher-labelled examples to tune them for your subject.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from ankan.errors import BackendUnavailableError, GradingError, model_unavailable
from ankan.grading.text import split_sentences, words
from ankan.schemas import KeyPoint

Matrix = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Thresholds:
    covered: float
    partial: float
    borderline_margin: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.partial < self.covered <= 1:
            raise ValueError("thresholds must satisfy 0 < partial < covered <= 1")
        if not 0 <= self.borderline_margin < 0.5:
            raise ValueError("borderline_margin must be in [0, 0.5)")


class Matcher(Protocol):
    name: str
    thresholds: Thresholds

    def similarity(self, key_points: Sequence[KeyPoint], spans: Sequence[str]) -> Matrix:
        """Return a (len(key_points), len(spans)) matrix of scores in [0, 1]."""
        ...


# --------------------------------------------------------------------------- lexical

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from into onto as is are was were be
    been being am do does did has have had it its this that these those there their they them
    he she his her we our you your i me my so such than then thus also can could will would
    shall should may might must which who whom whose what when where why how all any each
    some very more most less much many via per about
    take takes taken took place places occur occurs occurred happen happens happened called
    known way ways thing things important""".split()
)

_PHONETIC_RULES = (("ph", "f"), ("ck", "k"), ("z", "s"), ("y", "i"), ("q", "k"))
# Digits that handwriting recognition confuses with letters inside words ("LIF0", "Ce11").
_OCR_DIGITS = str.maketrans({"0": "o", "1": "l", "5": "s"})
_DOUBLE_LETTER = re.compile(r"(.)\1+")


@lru_cache(maxsize=8192)
def stem(word: str) -> str:
    """Tiny suffix stripper: 'absorbs', 'absorbed', 'absorbing' -> 'absorb'; 'use', 'used',
    'uses' -> 'use'; 'remove', 'removed' -> 'remov'."""
    for suffix, replacement in (
        ("ies", "y"),
        ("ing", ""),
        ("ed", ""),
        ("es", ""),
        ("d", ""),
        ("s", ""),
    ):
        if suffix == "d" and not word.endswith("ed"):
            continue  # only "used" -> "use"; never "bad" -> "ba"
        if suffix == "s" and word.endswith("ss"):
            continue  # "class", "process" are not plurals
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    if word.endswith("e") and len(word) > 3:
        return word[:-1]  # "remove" matches "removed" -> "remov", "tree" matches "trees"
    return word


@lru_cache(maxsize=8192)
def phonetic(term: str) -> str:
    """Collapse common spelling/OCR variants: chlorophyll -> chlorofil, oxygen -> oxigen."""
    if any(ch.isalpha() for ch in term) and not term.isdigit():
        term = term.translate(_OCR_DIGITS)
    for old, new in _PHONETIC_RULES:
        term = term.replace(old, new)
    return _DOUBLE_LETTER.sub(r"\1", term)


def content_terms(text: str, drop: frozenset[str] = frozenset()) -> list[str]:
    seen: dict[str, None] = {}
    for word in words(text):
        if word not in STOPWORDS and word not in drop and len(word) > 1:
            seen[stem(word)] = None
    return list(seen)


@lru_cache(maxsize=65536)
def term_similarity(a: str, b: str) -> float:
    """1.0 for the same term; fuzzy ratio for longer words; 0 for different short words."""
    if a == b or phonetic(a) == phonetic(b):
        return 1.0
    shortest = min(len(a), len(b))
    if shortest < 4:  # short tokens (co2, dna, ph) must match exactly
        return 0.0
    ratio = max(
        SequenceMatcher(None, a, b).ratio(), SequenceMatcher(None, phonetic(a), phonetic(b)).ratio()
    )
    cutoff = 0.8 if shortest >= 6 else 0.85  # short words: 'light' vs 'night' is not a typo
    return ratio if ratio >= cutoff else 0.0


# Words teachers use to phrase a point, not part of what the student must write:
# "Gives a valid example ...", "Correctly explains that ...", "Stack follows LIFO".
INSTRUCTION_WORDS = frozenset(
    """give gives given giving mention mentions mentioned mentioning explain explains explained
    describe describes described define defines defined write writes written identify
    identifies identified show shows shown follow follows followed valid correct correctly
    proper properly appropriate suitable relevant clearly any one least student answer
    use uses used using""".split()
)
# Words that only announce the list of examples that follows them ("an application, such as").
EXAMPLE_NOUNS = frozenset(
    """example examples application applications use uses usage instance instances case cases
    real world life""".split()
)
_EXAMPLES_IN_BRACKETS = re.compile(
    r"\(\s*(?:e\.?\s*g\.?|eg|for example|for instance|such as|like)[\s,:]+([^()]*)\)", re.I
)
_EXAMPLES_AFTER = re.compile(
    r"(?:,\s*|\s+)(?:such as|for example|for instance|e\.\s*g\.|eg\.|including|,\s*like)"
    r"[\s,:]+(.+)$",
    re.I,
)
_EXAMPLE_SEPARATORS = re.compile(r",|;|/|\bor\b|\band\b|\betc\b\.?", re.I)
_BRACKET = re.compile(r"\(([^()]*)\)")
_RAW_WORD = re.compile(r"[A-Za-z0-9][\w'-]*")


@dataclass(frozen=True)
class Alternative:
    """One way of expressing an idea.

    mode "fraction": partial credit for each term found (teacher keywords).
    mode "all": every term must appear, in any order (listed examples).
    mode "phrase": the words must appear in this order ("last in first out"), because a
    near-miss like "first in first out" means the opposite.
    """

    terms: tuple[str, ...]
    mode: str = "fraction"


Group = tuple[Alternative, ...]  # one required idea: any one alternative will do


@dataclass(frozen=True)
class Requirements:
    """What a span must contain to express a key point."""

    groups: tuple[Group, ...]  # every group is needed; partly covered gives a partial score
    examples: Group | None = None  # if set, one listed example is needed as well

    def terms(self) -> set[str]:
        groups = (*self.groups, self.examples) if self.examples else self.groups
        return {term for group in groups for alt in group for term in alt.terms}

    def subject_terms(self) -> set[str]:
        """Single words the point is about ("stack"), as opposed to phrases or examples."""
        return {
            group[0].terms[0]
            for group in self.groups
            if len(group) == 1 and len(group[0].terms) == 1
        }


def _phrase(text: str) -> tuple[str, ...]:
    return tuple(stem(word) for word in words(text) if word not in INSTRUCTION_WORDS)


def _either_wordings(text: str) -> tuple[str, list[Group]]:
    """Turn "WORDS (OTHER WORDS)" into one idea that either wording expresses.

    "LIFO (Last In First Out)": the bracket expands the word before it.
    "carbon dioxide (CO2)", "Central Processing Unit (CPU)": the bracket is a short form of
    the phrase before it; the phrase is found from the initials, or else as up to two words
    back to the previous small word ("and", "the").
    """
    groups: list[Group] = []
    kept: list[str] = []
    position = 0
    stop = STOPWORDS | INSTRUCTION_WORDS | EXAMPLE_NOUNS
    for bracket in _BRACKET.finditer(text):
        before = text[position : bracket.start()]
        inner = bracket.group(1)
        previous = list(_RAW_WORD.finditer(before))
        inner_words = words(inner)
        position = bracket.end()
        if not previous or not inner_words:
            kept.append(f"{before} {inner}")
            continue
        count, long_mode = 1, "phrase"
        if len(inner_words) == 1:
            short = inner_words[0]
            initials = [
                n
                for n in range(min(6, len(previous)), 1, -1)
                if "".join(m.group(0)[0].lower() for m in previous[-n:]) == short
            ]
            if initials:
                count = initials[0]
            else:
                count, long_mode = 0, "all"
                for match in reversed(previous[-2:]):
                    if match.group(0).lower() in stop:
                        break
                    count += 1
                count = max(count, 1)
        long_text = before[previous[-count].start() :]
        kept.append(before[: previous[-count].start()])
        options = []
        for part, mode in ((long_text, long_mode), (inner, "phrase")):
            terms = _phrase(part)
            if terms:
                options.append(Alternative(terms, mode if len(terms) > 1 else "all"))
        if options:
            groups.append(tuple(options))
    kept.append(text[position:])
    return " ".join(kept), groups


@lru_cache(maxsize=1024)
def parse_description(description: str) -> Requirements:
    """Read a point the way a teacher writes it.

    * Instruction words ("gives", "valid", "follows") are ignored.
    * "WORD (other words)" means either wording: "LIFO (Last In First Out)".
    * "such as A, B, or C" and "(e.g. A, B)" mean any one of the listed examples.
    """
    text = description
    example_text = None
    bracketed = _EXAMPLES_IN_BRACKETS.search(text)
    if bracketed:
        example_text = bracketed.group(1)
        text = text[: bracketed.start()] + " " + text[bracketed.end() :]
    else:
        trailing = _EXAMPLES_AFTER.search(text)
        if trailing:
            example_text = trailing.group(1)
            text = text[: trailing.start()]

    text, groups = _either_wordings(text)
    drop = INSTRUCTION_WORDS | EXAMPLE_NOUNS if example_text else INSTRUCTION_WORDS
    groups.extend((Alternative((term,), "all"),) for term in content_terms(text, drop))

    examples: Group | None = None
    if example_text:
        subject = {term for group in groups for alt in group for term in alt.terms}
        options = []
        for part in _EXAMPLE_SEPARATORS.split(example_text):
            terms = content_terms(part, INSTRUCTION_WORDS | EXAMPLE_NOUNS)
            # "printer queue" in a point about queues: the distinctive part is "printer".
            distinctive = [term for term in terms if term not in subject] or terms
            if distinctive:
                options.append(Alternative(tuple(distinctive), "all"))
        examples = tuple(options) or None
    return Requirements(groups=tuple(groups), examples=examples)


def requirements_for(key_point: KeyPoint) -> Requirements:
    """Keywords if the teacher gave them, otherwise read from the point's wording."""
    if key_point.keywords:
        groups = []
        for keyword in key_point.keywords:
            options = [content_terms(alt) or words(alt) for alt in keyword.split("|")]
            group = tuple(Alternative(tuple(terms)) for terms in options if terms)
            if group:
                groups.append(group)
        return Requirements(groups=tuple(groups))
    return parse_description(key_point.description)


def _alternative_score(alt: Alternative, present: Sequence[str], sequence: Sequence[str]) -> float:
    if alt.mode == "phrase":
        n = len(alt.terms)
        for start in range(len(sequence) - n + 1):
            window = sequence[start : start + n]
            if all(term_similarity(a, b) > 0 for a, b in zip(alt.terms, window, strict=True)):
                return 1.0
        return 0.0
    scores = [max((term_similarity(term, p) for p in present), default=0.0) for term in alt.terms]
    if alt.mode == "all" and min(scores) == 0:
        return 0.0
    return sum(scores) / len(scores)


class LexicalMatcher:
    name = "lexical"

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self.thresholds = thresholds or Thresholds(covered=0.7, partial=0.4)

    @classmethod
    def _score(
        cls,
        needs: Requirements,
        weights: Sequence[float],
        span: str,
        own_topics: frozenset[str] = frozenset(),
        other_topics: frozenset[str] = frozenset(),
    ) -> float:
        present = content_terms(span) or words(span)
        if other_topics and set(present) & other_topics:
            sentences = split_sentences(span)
            if len(sentences) > 1:
                # Joining "Stack is FIFO." with "Queue is LIFO." would credit the stack point
                # with the queue sentence's LIFO: judge sentences on their own instead.
                return max(
                    cls._score(needs, weights, sentence, own_topics, other_topics)
                    for sentence in sentences
                )
            if not set(present) & own_topics:
                return 0.0  # the sentence is about another point's topic ("Queue is LIFO")
        sequence = [stem(word) for word in words(span)]
        if not present or not (needs.groups or needs.examples):
            return 0.0

        def group_score(group: Group) -> float:
            return max(_alternative_score(alt, present, sequence) for alt in group)

        main = (
            sum(w * group_score(g) for w, g in zip(weights, needs.groups, strict=True))
            / sum(weights)
            if needs.groups
            else 1.0
        )
        if needs.examples is None:
            return main
        # Naming the topic is not enough: "Example of queue - stack of plates" gives no
        # listed queue application, so the weaker of the two parts decides.
        return min(main, group_score(needs.examples))

    def similarity(self, key_points: Sequence[KeyPoint], spans: Sequence[str]) -> Matrix:
        needs = [requirements_for(kp) for kp in key_points]
        # A word in several points of the same question ("stack", "queue") names the topic;
        # it counts for less than the idea that is unique to a point ("LIFO").
        frequency = Counter(term for need in needs for term in need.terms())
        subjects = Counter(term for need in needs for term in need.subject_terms())
        topics = frozenset(term for term, count in subjects.items() if count > 1)
        matrix = np.zeros((len(key_points), len(spans)), dtype=np.float64)
        for i, need in enumerate(needs):
            weights = [
                1.0 / min(frequency[t] for alt in group for t in alt.terms) for group in need.groups
            ]
            own_topics = topics & need.subject_terms()
            other_topics = topics - need.terms()
            for j, span in enumerate(spans):
                matrix[i, j] = self._score(need, weights, span, own_topics, other_topics)
        return matrix


# --------------------------------------------------------------------------- embeddings


class EmbeddingMatcher:
    name = "embedding"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        thresholds: Thresholds | None = None,
        model: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.thresholds = thresholds or Thresholds(covered=0.6, partial=0.42)
        self._model = model
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        self._load()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise BackendUnavailableError("embedding", "embeddings") from exc
                try:
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                except Exception as exc:  # download or network failures surface as many types
                    raise GradingError(
                        model_unavailable("embedding model", self.model_name, exc)
                    ) from exc
            return self._model

    def _encode(self, texts: Sequence[str]) -> Matrix:
        vectors = self._load().encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float64)

    def similarity(self, key_points: Sequence[KeyPoint], spans: Sequence[str]) -> Matrix:
        if not key_points or not spans:
            return np.zeros((len(key_points), len(spans)), dtype=np.float64)
        vectors = self._encode([kp.description for kp in key_points] + list(spans))
        kp_vectors, span_vectors = vectors[: len(key_points)], vectors[len(key_points) :]
        # Vectors are unit length, so the dot product is the cosine similarity.
        return np.asarray(np.clip(kp_vectors @ span_vectors.T, 0.0, 1.0), dtype=np.float64)
