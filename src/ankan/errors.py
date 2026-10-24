"""Exception hierarchy. Every error raised on purpose by Ankan derives from AnkanError."""

_FEATURES = {
    "trocr": "Reading photos of handwriting",
    "paddle": "Reading photos with PaddleOCR",
    "embedding": "The embedding grader",
    "llm": "The LLM grader",
}


class AnkanError(Exception):
    """Base class for all Ankan errors."""


class InvalidImageError(AnkanError):
    """The uploaded bytes could not be decoded as an image."""


class OCRError(AnkanError):
    """The OCR engine failed while reading a page."""


class GradingError(AnkanError):
    """The grader could not produce a valid result."""


class NotFoundError(AnkanError):
    """A rubric or submission does not exist."""


class ConflictError(AnkanError):
    """The request clashes with the current state (e.g. deciding an already-decided grade)."""


class BackendUnavailableError(AnkanError):
    """An optional dependency (torch, paddleocr, sentence-transformers, anthropic) is missing."""

    def __init__(self, backend: str, extra: str) -> None:
        feature = _FEATURES.get(backend, f"The '{backend}' backend")
        super().__init__(
            f"{feature} needs packages that are not installed. Stop the server, run "
            f'pip install -e ".[{extra}]" in the Ankan folder, then start the server again.'
        )


def model_unavailable(description: str, model: str, exc: BaseException) -> str:
    """Message for a model that could not be downloaded or loaded (usually no internet)."""
    return (
        f"Could not load the {description} ({model}). It is downloaded the first time it is "
        f"used, so check the internet connection and try again. Details: {str(exc)[:300]}"
    )
