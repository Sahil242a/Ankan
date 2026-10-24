"""Runtime settings, read from environment variables (all prefixed with ANKAN_)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

OCR_ENGINES = ("trocr", "paddle")
GRADERS = ("lexical", "embedding", "llm", "hybrid")


def _env(name: str, default: str) -> str:
    return os.environ.get(f"ANKAN_{name}", default).strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"ANKAN_{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    ocr_engine: str = "trocr"
    grader: str = "lexical"
    trocr_model: str = "microsoft/trocr-base-handwritten"
    paddle_lang: str = "en"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "claude-sonnet-5-5"
    device: str | None = None
    data_dir: Path = Path("data")
    low_ocr_confidence: float = 0.6
    disagreement_fraction: float = 0.25
    max_upload_mb: float = 10.0
    fast_ocr: bool = False  # 8-bit handwriting model: about 2x faster on CPU
    # When set, the browser asks for this password before showing anything (any username).
    password: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.ocr_engine not in OCR_ENGINES:
            raise ValueError(f"ocr_engine must be one of {OCR_ENGINES}, got {self.ocr_engine!r}")
        if self.grader not in GRADERS:
            raise ValueError(f"grader must be one of {GRADERS}, got {self.grader!r}")
        if not 0 <= self.low_ocr_confidence <= 1:
            raise ValueError("low_ocr_confidence must be between 0 and 1")
        if not 0 < self.disagreement_fraction <= 1:
            raise ValueError("disagreement_fraction must be in (0, 1]")
        if self.max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be positive")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ankan.sqlite3"

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "pages"

    @classmethod
    def from_env(cls) -> Settings:
        device = _env("DEVICE", "")
        return cls(
            ocr_engine=_env("OCR_ENGINE", cls.ocr_engine).lower(),
            grader=_env("GRADER", cls.grader).lower(),
            trocr_model=_env("TROCR_MODEL", cls.trocr_model),
            paddle_lang=_env("PADDLE_LANG", cls.paddle_lang),
            embedding_model=_env("EMBEDDING_MODEL", cls.embedding_model),
            llm_model=_env("LLM_MODEL", cls.llm_model),
            device=device or None,
            data_dir=Path(_env("DATA_DIR", "data")),
            low_ocr_confidence=_env_float("LOW_OCR_CONFIDENCE", cls.low_ocr_confidence),
            disagreement_fraction=_env_float("DISAGREEMENT_FRACTION", cls.disagreement_fraction),
            max_upload_mb=_env_float("MAX_UPLOAD_MB", cls.max_upload_mb),
            password=_env("PASSWORD", "") or None,
            fast_ocr=_env("FAST_OCR", "").lower() in ("1", "true", "yes", "on"),
        )
