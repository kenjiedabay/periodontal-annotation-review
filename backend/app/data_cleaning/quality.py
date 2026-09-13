"""Automated image quality signals for manual review triage.

These metrics are screening signals only. They do not establish clinical usability,
disease status, or diagnostic value.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityThresholds:
    blur_score: float = 20.0
    dark_brightness: float = 15.0
    bright_brightness: float = 240.0
    low_contrast: float = 10.0
    minimum_dimension: int = 64
    minimum_aspect_ratio: float = 0.2
    maximum_aspect_ratio: float = 5.0


@dataclass(frozen=True)
class QualityMetrics:
    blur_score: float
    brightness_score: float
    contrast_score: float
    quality_status: str
    manual_review_required: bool
    reasons: tuple[str, ...]


def inspect_image(image: Image.Image, thresholds: QualityThresholds) -> QualityMetrics:
    """Calculate non-clinical quality indicators and review flags."""
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = grayscale.shape[:2]
    aspect_ratio = width / height if height else 0.0
    blur_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    brightness_score = float(grayscale.mean())
    contrast_score = float(grayscale.std())

    reasons: list[str] = []
    if blur_score < thresholds.blur_score:
        reasons.append("low_sharpness_signal")
    if brightness_score < thresholds.dark_brightness:
        reasons.append("extremely_dark")
    if brightness_score > thresholds.bright_brightness:
        reasons.append("extremely_bright")
    if contrast_score < thresholds.low_contrast:
        reasons.append("very_low_contrast")
    if min(width, height) < thresholds.minimum_dimension:
        reasons.append("small_dimensions")
    if not thresholds.minimum_aspect_ratio <= aspect_ratio <= thresholds.maximum_aspect_ratio:
        reasons.append("unusual_aspect_ratio")
    if contrast_score < thresholds.low_contrast and blur_score < thresholds.blur_score:
        reasons.append("insufficient_visual_information_signal")

    return QualityMetrics(
        blur_score=round(blur_score, 4),
        brightness_score=round(brightness_score, 4),
        contrast_score=round(contrast_score, 4),
        quality_status="flagged_for_review" if reasons else "no_automated_flags",
        manual_review_required=bool(reasons),
        reasons=tuple(reasons),
    )
