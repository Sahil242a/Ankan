"""TrOCR engine: our own line segmentation + Microsoft's handwritten line recognizer.

TrOCR (ViT encoder + Transformer text decoder) reads one line at a time, so the page is
cut into lines by `analyze_layout` and the crops are recognised in batches.

Per-line confidence is the geometric mean of the generated tokens' probabilities, taken
from `compute_transition_scores` with padding after end-of-sequence masked out.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

from ankan.errors import BackendUnavailableError, OCRError, model_unavailable
from ankan.ocr.preprocess import Image, LineCrop, analyze_layout
from ankan.schemas import OCRLine, OCRResult

logger = logging.getLogger(__name__)


def _quantize(model: Any) -> Any:
    """Fast mode: store the linear layers' weights as 8-bit integers (about 2x faster on CPU).

    Falls back to the full-precision model if this PyTorch build cannot quantize.
    """
    import warnings

    import torch

    quantize: Any = torch.ao.quantization.quantize_dynamic  # untyped in PyTorch's stubs
    try:
        with warnings.catch_warnings():
            # Recent PyTorch prints a deprecation notice for this API on every load. If a
            # future release removes it, the except below keeps the app working.
            warnings.simplefilter("ignore")
            return quantize(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as exc:  # not available on every platform or PyTorch version
        logger.warning("fast mode unavailable, using the full-precision model: %s", exc)
        return model


class TrOCREngine:
    name = "trocr"

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        *,
        device: str | None = None,
        fast: bool = False,
        batch_size: int = 16,
        max_new_tokens: int = 64,
        processor: Any = None,
        model: Any = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if (processor is None) != (model is None):
            raise ValueError("pass both processor and model, or neither")
        self.model_name = model_name
        self.device = device
        self.fast = fast
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._processor = processor
        self._model = model
        self._lock = threading.Lock()
        if model is not None:
            self._prepare_model()

    # ------------------------------------------------------------------ loading

    def warm_up(self) -> None:
        """Load the model now (it is 1.3 GB), so the first upload does not wait for it."""
        self._load()

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:  # everything the reader needs, so a missing piece fails here, clearly
                import PIL  # noqa: F401
                import torch  # noqa: F401
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            except ImportError as exc:
                raise BackendUnavailableError("trocr", "trocr") from exc
            try:
                self._processor = TrOCRProcessor.from_pretrained(self.model_name)
                self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            except Exception as exc:  # download or network failures surface as many types
                raise OCRError(
                    model_unavailable("handwriting model", self.model_name, exc)
                ) from exc
            self._prepare_model()

    def _prepare_model(self) -> None:
        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)
        self._model.eval()
        if self.fast and self.device == "cpu":
            self._model = _quantize(self._model)

    # ------------------------------------------------------------------ inference

    def read(self, page: Image) -> OCRResult:
        crops = analyze_layout(page)
        recognised = self.recognize(crops)
        lines = [
            OCRLine(text=text, confidence=confidence, box=crop.box)
            for crop, (text, confidence) in zip(crops, recognised, strict=True)
        ]
        return OCRResult(engine=self.name, lines=lines)

    def recognize(self, crops: list[LineCrop]) -> list[tuple[str, float]]:
        if not crops:
            return []
        self._load()
        results: list[tuple[str, float]] = []
        for start in range(0, len(crops), self.batch_size):
            results.extend(self._recognize_batch(crops[start : start + self.batch_size]))
        return results

    def _recognize_batch(self, crops: list[LineCrop]) -> list[tuple[str, float]]:
        import torch
        from PIL import Image as PILImage

        processor, model = self._processor, self._model
        images = [PILImage.fromarray(crop.image).convert("RGB") for crop in crops]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        with torch.inference_mode():
            output = model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            # log-probabilities of each chosen token, shape (batch, generated_steps)
            token_logprobs = model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )
        texts = processor.batch_decode(output.sequences, skip_special_tokens=True)

        pad_id = processor.tokenizer.pad_token_id
        generated = output.sequences[:, -token_logprobs.shape[1] :]
        results: list[tuple[str, float]] = []
        for row, text in enumerate(texts):
            keep = generated[row] != pad_id if pad_id is not None else None
            scores = token_logprobs[row][keep] if keep is not None else token_logprobs[row]
            scores = scores[torch.isfinite(scores)]
            # No usable token scores means we cannot vouch for the line: report 0 so the
            # submission is flagged for the teacher instead of failing the whole page.
            confidence = math.exp(float(scores.mean())) if scores.numel() else 0.0
            results.append((text.strip(), min(1.0, max(0.0, confidence))))
        return results
