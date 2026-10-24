"""PaddleOCR engine (detection + recognition in one model).

PaddleOCR returns text *fragments* with polygons. We merge fragments that sit on the same
baseline into lines, ordered left to right, so the output matches the TrOCR engine.
Supports the PaddleOCR 3.x `predict` result format and the 2.x nested-list format.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ankan.errors import BackendUnavailableError, OCRError, model_unavailable
from ankan.ocr.preprocess import Image
from ankan.schemas import BoundingBox, OCRLine, OCRResult


@dataclass(frozen=True)
class Fragment:
    text: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def _fragment(polygon: Any, text: Any, score: Any) -> Fragment | None:
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if points.size == 0 or not str(text).strip():
        return None
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    clipped = min(1.0, max(0.0, float(score)))
    return Fragment(str(text).strip(), clipped, float(x0), float(y0), float(x1), float(y1))


def parse_paddle_output(raw: Any) -> list[Fragment]:
    """Normalise PaddleOCR output (3.x dict-like pages or 2.x nested lists) to fragments."""
    fragments: list[Fragment] = []
    pages: Iterable[Any] = raw if isinstance(raw, list) else [raw]
    for page in pages:
        if page is None:
            continue
        if isinstance(page, Mapping):  # PaddleOCR 3.x
            texts = page.get("rec_texts")
            if texts is None:
                texts = []
            scores = page.get("rec_scores")
            polygons = page.get("rec_polys")
            if polygons is None:
                polygons = page.get("dt_polys")
            if scores is None or polygons is None:
                raise OCRError("unexpected PaddleOCR result: missing scores or polygons")
            if not len(texts) == len(scores) == len(polygons):
                raise OCRError("unexpected PaddleOCR result: field lengths differ")
            items: Iterable[tuple[Any, Any, Any]] = zip(polygons, texts, scores, strict=True)
        elif isinstance(page, Sequence):  # PaddleOCR 2.x: [[polygon, (text, score)], ...]
            items = ((entry[0], entry[1][0], entry[1][1]) for entry in page)
        else:
            raise OCRError(f"unexpected PaddleOCR result type: {type(page).__name__}")
        for polygon, text, score in items:
            fragment = _fragment(polygon, text, score)
            if fragment is not None:
                fragments.append(fragment)
    return fragments


def group_into_lines(fragments: list[Fragment]) -> list[OCRLine]:
    """Merge fragments whose vertical centres are close into reading-order lines."""
    if not fragments:
        return []
    tolerance = 0.5 * float(np.median([max(f.height, 1.0) for f in fragments]))
    groups: list[list[Fragment]] = []
    for fragment in sorted(fragments, key=lambda f: f.y_center):
        if groups:
            centre = sum(f.y_center for f in groups[-1]) / len(groups[-1])
            if abs(fragment.y_center - centre) <= tolerance:
                groups[-1].append(fragment)
                continue
        groups.append([fragment])

    lines: list[OCRLine] = []
    for group in groups:
        group.sort(key=lambda f: f.x0)
        weights = [max(len(f.text), 1) for f in group]
        confidence = sum(f.score * w for f, w in zip(group, weights, strict=True)) / sum(weights)
        x0 = max(0, int(min(f.x0 for f in group)))
        y0 = max(0, int(min(f.y0 for f in group)))
        x1 = int(np.ceil(max(f.x1 for f in group)))
        y1 = int(np.ceil(max(f.y1 for f in group)))
        lines.append(
            OCRLine(
                text=" ".join(f.text for f in group),
                confidence=confidence,
                box=BoundingBox(x=x0, y=y0, width=max(0, x1 - x0), height=max(0, y1 - y0)),
            )
        )
    return lines


class PaddleOCREngine:
    name = "paddle"

    def __init__(self, lang: str = "en", *, client: Any = None) -> None:
        self.lang = lang
        self._client = client
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._client is None:
                try:
                    from paddleocr import PaddleOCR
                except ImportError as exc:
                    raise BackendUnavailableError("paddle", "paddle") from exc
                try:
                    try:  # PaddleOCR 3.x: the page is already deskewed by us
                        self._client = PaddleOCR(
                            lang=self.lang,
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_textline_orientation=False,
                        )
                    except TypeError:  # PaddleOCR 2.x
                        self._client = PaddleOCR(lang=self.lang)
                except Exception as exc:  # model download or setup failure
                    raise OCRError(model_unavailable("PaddleOCR model", self.lang, exc)) from exc
            return self._client

    def read(self, page: Image) -> OCRResult:
        client = self._load()
        try:
            raw = client.predict(page) if hasattr(client, "predict") else client.ocr(page)
        except Exception as exc:  # third-party failure: surface as our own error type
            raise OCRError(f"PaddleOCR failed: {exc}") from exc
        return OCRResult(engine=self.name, lines=group_into_lines(parse_paddle_output(raw)))
