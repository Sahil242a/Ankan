"""OCR engines. Heavy dependencies are imported lazily, only when an engine is used."""

from ankan.ocr.base import OCREngine
from ankan.ocr.paddle import PaddleOCREngine
from ankan.ocr.trocr import TrOCREngine

__all__ = ["OCREngine", "PaddleOCREngine", "TrOCREngine"]
