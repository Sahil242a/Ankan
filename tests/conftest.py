from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ankan.api import create_app
from ankan.config import Settings
from ankan.grading import KeyPointGrader, LexicalMatcher
from ankan.ocr.preprocess import Image
from ankan.pipeline import GradingPipeline
from ankan.schemas import BoundingBox, OCRLine, OCRResult, Rubric
from ankan.store import ReviewStore

DATA = Path(__file__).resolve().parent / "data"

FULL_ANSWER = (
    "Photosynthesis happens in the chloroplasts. The green pigment chlorophyll absorbs "
    "sunlight. Plants use water and carbon dioxide to make glucose, and oxygen is released."
)


class FakeOCR:
    """Stands in for TrOCR/PaddleOCR: returns fixed lines, records the page it was given."""

    name = "fake"

    def __init__(self, lines: list[tuple[str, float]] | None = None) -> None:
        self.lines = lines if lines is not None else [(FULL_ANSWER, 0.93)]
        self.pages: list[Image] = []

    def read(self, page: Image) -> OCRResult:
        self.pages.append(page)
        return OCRResult(
            engine=self.name,
            lines=[
                OCRLine(
                    text=text,
                    confidence=confidence,
                    box=BoundingBox(x=10, y=10 + 40 * i, width=300, height=30),
                )
                for i, (text, confidence) in enumerate(self.lines)
            ],
        )


@pytest.fixture
def rubric_data() -> dict[str, object]:
    data: dict[str, object] = json.loads(
        (DATA / "rubric_photosynthesis.json").read_text(encoding="utf-8")
    )
    return data


@pytest.fixture
def rubric(rubric_data: dict[str, object]) -> Rubric:
    return Rubric.model_validate(rubric_data)


@pytest.fixture
def fake_ocr() -> FakeOCR:
    return FakeOCR()


@pytest.fixture
def pipeline(fake_ocr: FakeOCR) -> GradingPipeline:
    return GradingPipeline(fake_ocr, KeyPointGrader(LexicalMatcher()))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(grader="lexical", data_dir=tmp_path / "data", max_upload_mb=1)


@pytest.fixture
def store(settings: Settings) -> ReviewStore:
    return ReviewStore(settings.db_path, settings.image_dir)


@pytest.fixture
def client(
    settings: Settings, pipeline: GradingPipeline, store: ReviewStore
) -> Iterator[TestClient]:
    app = create_app(settings, pipeline=pipeline, store=store)
    with TestClient(app) as test_client:
        yield test_client
