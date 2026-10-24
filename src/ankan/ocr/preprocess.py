"""Image preprocessing for photographed or scanned answer sheets.

Pipeline: decode -> limit size -> deskew -> binarize -> remove ruled/margin lines ->
segment text lines. Line-level crops are what TrOCR expects (it is a single-line model).

Conventions: binary images use 255 for ink and 0 for paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from ankan.errors import InvalidImageError
from ankan.schemas import BoundingBox

Image = npt.NDArray[np.uint8]

MAX_SIDE = 2400
MIN_INK_FRACTION = 0.001
MIN_COMPONENT_AREA = 6


@dataclass(frozen=True)
class LineCrop:
    box: BoundingBox
    image: Image  # RGB crop of the cleaned page, ready for a line recognizer


# --------------------------------------------------------------------------- basics


def decode_image(data: bytes) -> Image:
    """Decode PNG/JPEG/... bytes into a BGR image."""
    if not data:
        raise InvalidImageError("the uploaded file is empty")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("could not read the file as an image; upload a PNG or JPEG")
    return np.asarray(image, dtype=np.uint8)


def encode_png(image: Image) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise InvalidImageError("could not encode image as PNG")
    return bytes(buffer.tobytes())


def limit_size(image: Image, max_side: int = MAX_SIDE) -> Image:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)


def to_gray(image: Image) -> Image:
    if image.ndim == 2:
        return image
    code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return np.asarray(cv2.cvtColor(image, code), dtype=np.uint8)


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def binarize(gray: Image) -> Image:
    """Adaptive threshold (robust to uneven phone-camera lighting), then drop speckles."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    block = _odd(max(15, min(gray.shape[:2]) // 40))
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 15
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    small = np.where(stats[1:count, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA)[0] + 1
    if small.size:
        binary[np.isin(labels, small)] = 0
    return np.asarray(binary, dtype=np.uint8)


def rotate(image: Image, angle: float, *, is_mask: bool = False) -> Image:
    """Rotate around the centre, keeping the size. Positive angle = counter-clockwise."""
    if angle == 0:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    fill: float | tuple[int, int, int]
    if is_mask:
        fill = 0
    elif image.ndim == 2:
        fill = 255
    else:
        fill = (255, 255, 255)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    return np.asarray(rotated, dtype=np.uint8)


# --------------------------------------------------------------------------- deskew


def estimate_skew(binary: Image, max_angle: float = 10.0) -> float:
    """Return the rotation (degrees) that makes text lines horizontal.

    Projection-profile search: when lines are horizontal, the row sums alternate sharply
    between ink and paper, which maximises their variance. Coarse 1-degree sweep, then a
    0.1-degree refinement. Does not depend on OpenCV's minAreaRect angle convention.
    """
    if np.count_nonzero(binary) < MIN_INK_FRACTION * binary.size:
        return 0.0
    height, width = binary.shape[:2]
    scale = min(1.0, 800 / width)
    small = binary
    if scale < 1:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = cv2.resize(binary, size, interpolation=cv2.INTER_AREA)
        small = np.where(resized > 127, 255, 0).astype(np.uint8)

    def score(angle: float) -> float:
        profile = rotate(small, angle, is_mask=True).sum(axis=1, dtype=np.float64)
        return float(np.var(profile))

    coarse = np.arange(-max_angle, max_angle + 1e-9, 1.0)
    best = float(max(coarse, key=score))
    fine = np.arange(best - 1.0, best + 1.0 + 1e-9, 0.1)
    best = float(max(fine, key=score))
    # Do not rotate a page that is already straight just because of noise.
    if score(best) <= score(0.0) * 1.001:
        return 0.0
    return round(best, 2)


def deskew(image: Image) -> tuple[Image, float]:
    """Downscale very large photos and straighten the page. Returns (image, angle_applied)."""
    image = limit_size(image)
    angle = estimate_skew(binarize(to_gray(image)))
    return rotate(image, angle), angle


# --------------------------------------------------------------------------- layout


def remove_rules(binary: Image) -> tuple[Image, Image]:
    """Remove ruled lines and margin lines printed on answer sheets.

    Long thin horizontal/vertical structures are found with morphological opening.
    Returns (clean_binary, rule_mask). Strokes cut by a removed rule are re-joined with a
    small vertical closing.
    """
    height, width = binary.shape[:2]
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 12), 1))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, height // 12)))
    rules = cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical),
    )
    rules = cv2.dilate(rules, np.ones((3, 3), np.uint8))
    clean = cv2.bitwise_and(binary, cv2.bitwise_not(rules))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    return np.asarray(clean, dtype=np.uint8), np.asarray(rules, dtype=np.uint8)


def _runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Contiguous True runs as half-open (start, end) intervals."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _split_tall(
    run: tuple[int, int], profile: npt.NDArray[np.float64], line_height: float
) -> list[tuple[int, int]]:
    """Split a run that probably contains several touching lines at its profile minima."""
    start, end = run
    if end - start <= 1.8 * line_height:
        return [run]
    margin = int(0.5 * line_height)
    lo, hi = start + margin, end - margin
    if hi <= lo:
        return [run]
    cut = lo + int(np.argmin(profile[lo:hi]))
    return _split_tall((start, cut), profile, line_height) + _split_tall(
        (cut, end), profile, line_height
    )


def segment_lines(binary: Image) -> list[BoundingBox]:
    """Find text lines with a horizontal projection profile. Boxes are top-to-bottom."""
    height, width = binary.shape[:2]
    profile = np.count_nonzero(binary, axis=1).astype(np.float64)
    if not profile.any():
        return []
    window = _odd(max(3, height // 300))
    smooth = np.convolve(profile, np.ones(window) / window, mode="same")
    positive = smooth[smooth > 0]
    threshold = max(1.0, 0.05 * float(np.percentile(positive, 90)))
    runs = _runs(smooth > threshold)
    if not runs:
        return []

    heights = [end - start for start, end in runs if end - start >= 6]
    line_height = float(np.median(heights)) if heights else float(max(e - s for s, e in runs))

    merged: list[tuple[int, int]] = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < 0.3 * line_height:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Attach thin fragments (i-dots, accents) to the nearest line, or drop them as noise.
    kept: list[tuple[int, int]] = []
    for index, (start, end) in enumerate(merged):
        if end - start >= 0.35 * line_height:
            kept.append((start, end))
            continue
        neighbours = []
        if kept:
            neighbours.append((start - kept[-1][1], "prev"))
        if index + 1 < len(merged):
            neighbours.append((merged[index + 1][0] - end, "next"))
        if not neighbours:
            continue
        gap, side = min(neighbours)
        if gap > 0.6 * line_height:
            continue
        if side == "prev":
            kept[-1] = (kept[-1][0], end)
        else:
            merged[index + 1] = (start, merged[index + 1][1])

    split: list[tuple[int, int]] = []
    for run in kept:
        split.extend(_split_tall(run, smooth, line_height))

    min_ink = 0.05 * line_height * line_height
    inked = [(s, e) for s, e in split if np.count_nonzero(binary[s:e]) >= min_ink]

    boxes: list[BoundingBox] = []
    pad_y = max(2, int(0.15 * line_height))
    pad_x = max(4, int(0.3 * line_height))
    for index, (start, end) in enumerate(inked):
        band = binary[start:end]
        columns = np.flatnonzero(np.count_nonzero(band, axis=0))
        x0 = max(0, int(columns[0]) - pad_x)
        x1 = min(width, int(columns[-1]) + 1 + pad_x)
        # Pad vertically, but never past the midpoint to a neighbouring line.
        upper = (inked[index - 1][1] + start) // 2 if index > 0 else 0
        lower = (end + inked[index + 1][0] + 1) // 2 if index + 1 < len(inked) else height
        y0 = max(upper, start - pad_y)
        y1 = min(lower, end + pad_y)
        boxes.append(BoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0))
    return boxes


def analyze_layout(page: Image) -> list[LineCrop]:
    """Clean a (deskewed) page and cut it into line crops, top to bottom, for a line reader."""
    gray = to_gray(page)
    clean_binary, rules = remove_rules(binarize(gray))
    clean_gray = gray.copy()
    clean_gray[rules > 0] = 255
    lines = []
    for box in segment_lines(clean_binary):
        crop = clean_gray[box.y : box.y + box.height, box.x : box.x + box.width]
        rgb = np.asarray(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
        lines.append(LineCrop(box=box, image=rgb))
    return lines
