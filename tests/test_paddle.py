from __future__ import annotations

import numpy as np
import pytest

from ankan.errors import OCRError
from ankan.ocr import PaddleOCREngine
from ankan.ocr.paddle import group_into_lines, parse_paddle_output


def poly(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_parses_paddleocr3_pages() -> None:
    page = {
        "rec_texts": ["absorbs light", "Chlorophyll", ""],
        "rec_scores": np.array([0.9, 0.8, 0.99]),
        "rec_polys": [
            np.array(poly(300, 12, 520, 48)),
            np.array(poly(20, 10, 280, 50)),
            poly(0, 0, 1, 1),
        ],
    }
    fragments = parse_paddle_output([page])
    assert [f.text for f in fragments] == ["absorbs light", "Chlorophyll"]  # blanks dropped
    lines = group_into_lines(fragments)
    assert len(lines) == 1
    assert lines[0].text == "Chlorophyll absorbs light"  # reordered left to right
    assert lines[0].box is not None and lines[0].box.x == 20 and lines[0].box.width == 500
    assert lines[0].confidence == pytest.approx((0.8 * 11 + 0.9 * 13) / 24)


def test_parses_paddleocr2_nested_lists() -> None:
    raw = [
        [
            [poly(20, 100, 300, 140), ("second line", 0.7)],
            [poly(20, 10, 300, 50), ("first line", 0.95)],
        ]
    ]
    lines = group_into_lines(parse_paddle_output(raw))
    assert [line.text for line in lines] == ["first line", "second line"]


def test_empty_and_malformed_output() -> None:
    assert group_into_lines(parse_paddle_output([None])) == []
    with pytest.raises(OCRError, match="lengths differ"):
        parse_paddle_output([{"rec_texts": ["a"], "rec_scores": [], "rec_polys": []}])
    with pytest.raises(OCRError, match="unexpected PaddleOCR result type"):
        parse_paddle_output([42])


class FakePaddle:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, page: np.ndarray) -> list[dict[str, object]]:
        self.calls += 1
        return [{"rec_texts": ["hello"], "rec_scores": [0.5], "rec_polys": [poly(1, 1, 50, 20)]}]


def test_engine_uses_injected_client() -> None:
    client = FakePaddle()
    result = PaddleOCREngine(client=client).read(np.zeros((40, 80, 3), dtype=np.uint8))
    assert client.calls == 1
    assert result.engine == "paddle"
    assert result.text == "hello"


def test_failed_paddle_setup_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    def offline(**kwargs: object) -> object:
        raise RuntimeError("download failed")

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=offline))
    with pytest.raises(OCRError, match="check the internet connection"):
        PaddleOCREngine().read(np.zeros((40, 80, 3), dtype=np.uint8))
