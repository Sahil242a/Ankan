"""End-to-end pipeline: page image -> OCR -> rubric grading -> review flags."""

from __future__ import annotations

from dataclasses import dataclass

from ankan.config import Settings
from ankan.grading import (
    AnthropicClient,
    EmbeddingMatcher,
    Grader,
    HybridGrader,
    KeyPointGrader,
    LexicalMatcher,
    LLMRubricGrader,
)
from ankan.grading.text import comparable
from ankan.ocr import OCREngine, PaddleOCREngine, TrOCREngine
from ankan.ocr.preprocess import Image, decode_image, deskew
from ankan.schemas import FlagCode, GradeResult, OCRResult, ReviewFlag, Rubric


@dataclass(frozen=True)
class PageGrade:
    page: Image  # the deskewed page the OCR boxes refer to
    ocr: OCRResult
    grade: GradeResult


class GradingPipeline:
    def __init__(self, ocr: OCREngine, grader: Grader, *, low_ocr_confidence: float = 0.6) -> None:
        self.ocr = ocr
        self.grader = grader
        self.low_ocr_confidence = low_ocr_confidence

    def warm_up(self) -> None:
        """Load the models in advance so the first answer is not slowed down by loading."""
        for part in (self.ocr, self.grader):
            warm = getattr(part, "warm_up", None)
            if warm is not None:
                warm()

    def grade_image(self, data: bytes, rubric: Rubric) -> PageGrade:
        page, angle = deskew(decode_image(data))
        ocr = self.ocr.read(page).model_copy(update={"skew_angle": angle})
        return PageGrade(page=page, ocr=ocr, grade=self.grade_text(ocr.text, rubric, ocr=ocr))

    def grade_text(self, text: str, rubric: Rubric, *, ocr: OCRResult | None = None) -> GradeResult:
        """Grade already-extracted text (typed answers, or OCR text the teacher corrected)."""
        result = self.grader.grade(text, rubric)
        flags = [*self._answer_flags(text, ocr), *result.flags]
        return result.model_copy(update={"flags": flags})

    def _answer_flags(self, text: str, ocr: OCRResult | None) -> list[ReviewFlag]:
        flags: list[ReviewFlag] = []
        if not comparable(text):
            flags.append(
                ReviewFlag(
                    code=FlagCode.EMPTY_ANSWER,
                    message="no answer text was found; check the scan for a blank or "
                    "unreadable page",
                )
            )
        if ocr is not None:
            weak = [
                number
                for number, line in enumerate(ocr.lines, start=1)
                if line.confidence is not None and line.confidence < self.low_ocr_confidence
            ]
            if weak:
                flags.append(
                    ReviewFlag(
                        code=FlagCode.LOW_OCR_CONFIDENCE,
                        message=(
                            f"handwriting recognition was unsure on line(s) "
                            f"{', '.join(map(str, weak))}; compare the text with the scan "
                            "before approving"
                        ),
                    )
                )
        return flags


def build_ocr(settings: Settings) -> OCREngine:
    if settings.ocr_engine == "paddle":
        return PaddleOCREngine(lang=settings.paddle_lang)
    return TrOCREngine(settings.trocr_model, device=settings.device, fast=settings.fast_ocr)


def build_grader(settings: Settings) -> Grader:
    if settings.grader == "lexical":
        return KeyPointGrader(LexicalMatcher())
    embedding = KeyPointGrader(EmbeddingMatcher(settings.embedding_model, device=settings.device))
    if settings.grader == "embedding":
        return embedding
    llm = LLMRubricGrader(AnthropicClient(settings.llm_model))
    if settings.grader == "llm":
        return llm
    return HybridGrader(llm, embedding, disagreement_fraction=settings.disagreement_fraction)


def build_pipeline(settings: Settings) -> GradingPipeline:
    return GradingPipeline(
        build_ocr(settings),
        build_grader(settings),
        low_ocr_confidence=settings.low_ocr_confidence,
    )
