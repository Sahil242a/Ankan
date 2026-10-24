"""The OCR engine interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ankan.ocr.preprocess import Image
from ankan.schemas import OCRResult


@runtime_checkable
class OCREngine(Protocol):
    """Reads a deskewed page image (BGR) and returns its text lines in reading order."""

    name: str

    def read(self, page: Image) -> OCRResult: ...
