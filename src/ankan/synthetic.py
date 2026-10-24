"""Render synthetic 'handwritten' answer sheets: ruled paper, margin line, script font.

Used by the test suite and the demo so the image pipeline can be exercised without
real student data (which should never be committed to a repository).
"""

from __future__ import annotations

import cv2
import numpy as np

from ankan.ocr.preprocess import Image, rotate

RULE_SPACING = 72
TOP_MARGIN = 140
LEFT_MARGIN = 150


def render_sheet(
    lines: list[str],
    *,
    width: int = 1700,
    angle: float = 0.0,
    ruled: bool = True,
    noise: float = 0.0,
    seed: int = 0,
) -> Image:
    """Return a BGR image of the given lines written on ruled paper."""
    height = TOP_MARGIN + RULE_SPACING * (len(lines) + 3)
    page = np.full((height, width, 3), 250, dtype=np.uint8)
    if ruled:
        for y in range(TOP_MARGIN, height - RULE_SPACING // 2, RULE_SPACING):
            cv2.line(page, (0, y), (width, y), (225, 190, 170), 2)
        cv2.line(page, (LEFT_MARGIN - 30, 0), (LEFT_MARGIN - 30, height), (160, 160, 235), 2)
    for index, text in enumerate(lines):
        baseline = TOP_MARGIN + RULE_SPACING * (index + 1) - 14
        cv2.putText(
            page,
            text,
            (LEFT_MARGIN, baseline),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            1.35,
            (120, 45, 20),
            2,
            cv2.LINE_AA,
        )
    if noise > 0:
        rng = np.random.default_rng(seed)
        grain = rng.normal(0, noise, page.shape)
        page = np.clip(page.astype(np.float64) + grain, 0, 255).astype(np.uint8)
    return rotate(page, angle)
