from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from ankan.errors import InvalidImageError
from ankan.ocr.preprocess import (
    analyze_layout,
    binarize,
    decode_image,
    deskew,
    encode_png,
    estimate_skew,
    limit_size,
    remove_rules,
    segment_lines,
    to_gray,
)
from ankan.synthetic import render_sheet

LINES = [
    "Photosynthesis happens in the chloroplasts.",
    "Chlorophyll absorbs sunlight energy.",
    "Water and carbon dioxide are used",
    "to make glucose and oxygen is released.",
]


def test_decode_rejects_empty_and_garbage() -> None:
    with pytest.raises(InvalidImageError, match="empty"):
        decode_image(b"")
    with pytest.raises(InvalidImageError, match="PNG or JPEG"):
        decode_image(b"definitely not an image")


def test_png_round_trip() -> None:
    page = render_sheet(LINES)
    assert np.array_equal(decode_image(encode_png(page)), page)


def test_limit_size_keeps_aspect_ratio() -> None:
    image = np.zeros((1000, 4000, 3), dtype=np.uint8)
    small = limit_size(image, max_side=2000)
    assert small.shape[:2] == (500, 2000)
    assert limit_size(small, max_side=2000) is small


@pytest.mark.parametrize("angle", [0.0, 2.0, -3.5, 6.0, -8.0])
def test_deskew_recovers_rotation(angle: float) -> None:
    correction = estimate_skew(binarize(to_gray(render_sheet(LINES, angle=angle, noise=6))))
    assert correction == pytest.approx(-angle, abs=0.3)


def test_blank_page_is_not_rotated_and_has_no_lines() -> None:
    blank = np.full((800, 1200, 3), 250, dtype=np.uint8)
    page, angle = deskew(blank)
    assert angle == 0.0
    assert analyze_layout(page) == []


def test_ruled_paper_without_writing_has_no_lines() -> None:
    clean, rules = remove_rules(binarize(to_gray(render_sheet([], ruled=True))))
    assert np.count_nonzero(rules) > 0
    assert segment_lines(clean) == []


@pytest.mark.parametrize("count", [1, 4, 9])
def test_segments_each_written_line(count: int) -> None:
    lines = [f"Line {i} of the answer about plants" for i in range(count)]
    page, _ = deskew(render_sheet(lines, angle=3.0, noise=5))
    boxes = [crop.box for crop in analyze_layout(page)]
    assert len(boxes) == count
    tops = [box.y for box in boxes]
    assert tops == sorted(tops)
    for upper, lower in pairwise(boxes):
        assert upper.y + upper.height <= lower.y  # lines do not overlap


def test_line_crops_are_rgb_and_inside_page() -> None:
    page, _ = deskew(render_sheet(LINES))
    height, width = page.shape[:2]
    for crop in analyze_layout(page):
        assert crop.image.ndim == 3 and crop.image.shape[2] == 3
        assert crop.image.shape[:2] == (crop.box.height, crop.box.width)
        assert crop.box.x + crop.box.width <= width
        assert crop.box.y + crop.box.height <= height


def test_touching_lines_are_split() -> None:
    # Line height is estimated from the other lines on the page, so a pair of lines joined
    # by a descender (one run about twice the usual height) is cut at its thinnest row.
    page = np.zeros((420, 600), dtype=np.uint8)
    page[40:80, 50:550] = 255
    page[120:160, 50:550] = 255
    page[200:240, 50:550] = 255
    page[242:282, 50:550] = 255  # two pixels below the previous line
    page[240:242, 300:302] = 255  # a stroke touching both
    boxes = segment_lines(page)
    assert len(boxes) == 4
    for upper, lower in pairwise(boxes):
        assert upper.y + upper.height <= lower.y  # crops never overlap
