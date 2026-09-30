"""Auditable crop, heatmap, and radiographic bone-loss geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .schema import RBL_SURFACES


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    crop_xyxy: tuple[float, float, float, float]
    output_size: int
    scale: float
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int

    def to_dict(self) -> dict:
        return asdict(self)


def normalized_box_to_xyxy(box: Iterable[float], width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = (float(value) for value in box)
    return ((cx - bw / 2) * width, (cy - bh / 2) * height,
            (cx + bw / 2) * width, (cy + bh / 2) * height)


def expand_box(box: Iterable[float], width: int, height: int, padding: float = 0.20) -> tuple[float, float, float, float]:
    """Add ``padding`` of box width/height on every side, clipped to the image."""
    x1, y1, x2, y2 = (float(value) for value in box)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("source box must be positive and inside image bounds")
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return max(0.0, x1 - dx), max(0.0, y1 - dy), min(float(width), x2 + dx), min(float(height), y2 + dy)


def make_letterbox_transform(crop_xyxy: Iterable[float], width: int, height: int, output_size: int = 256) -> LetterboxTransform:
    x1, y1, x2, y2 = (float(value) for value in crop_xyxy)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("crop must be positive and inside image bounds")
    crop_width, crop_height = x2 - x1, y2 - y1
    scale = min(output_size / crop_width, output_size / crop_height)
    resized_width = max(1, min(output_size, round(crop_width * scale)))
    resized_height = max(1, min(output_size, round(crop_height * scale)))
    return LetterboxTransform(width, height, (x1, y1, x2, y2), output_size, scale,
                              resized_width, resized_height,
                              (output_size - resized_width) // 2,
                              (output_size - resized_height) // 2)


def image_to_crop(point: Iterable[float], transform: LetterboxTransform) -> tuple[float, float]:
    x, y = (float(value) for value in point)
    x1, y1, _, _ = transform.crop_xyxy
    return ((x - x1) * transform.scale + transform.offset_x,
            (y - y1) * transform.scale + transform.offset_y)


def crop_to_image(point: Iterable[float], transform: LetterboxTransform) -> tuple[float, float]:
    x, y = (float(value) for value in point)
    x1, y1, _, _ = transform.crop_xyxy
    return ((x - transform.offset_x) / transform.scale + x1,
            (y - transform.offset_y) / transform.scale + y1)


def gaussian_heatmaps(points: list[tuple[float, float] | None], output_size: int = 256,
                      sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Return one Gaussian channel per landmark and its availability mask."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    heatmaps = np.zeros((len(points), output_size, output_size), dtype=np.float32)
    available = np.zeros((len(points),), dtype=np.float32)
    yy, xx = np.mgrid[:output_size, :output_size]
    for index, point in enumerate(points):
        if point is None:
            continue
        x, y = point
        if not (0 <= x < output_size and 0 <= y < output_size):
            raise ValueError(f"landmark {index} lies outside the prepared crop")
        heatmaps[index] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        available[index] = 1.0
    return heatmaps, available


def calculate_rbl(points: list[tuple[float, float] | None], surface: str,
                  *, minimum_root_pixels: float = 5.0,
                  maximum_projection_ratio: float = 1.25,
                  indices: tuple[int, int, int] | None = None) -> dict:
    """Calculate ground-truth RBL after conservative geometric validation."""
    if surface not in RBL_SURFACES:
        raise ValueError(f"unknown RBL surface: {surface}")
    indices = indices or RBL_SURFACES[surface]
    names = ("cej", "bone_level", "apex")
    selected = [points[index] for index in indices]
    missing = [name for name, point in zip(names, selected) if point is None]
    if missing:
        return {"surface": surface, "status": "not_assessable", "reason": "missing_landmarks", "missing": missing}
    cej, bone, apex = (np.asarray(point, dtype=np.float64) for point in selected)  # type: ignore[arg-type]
    root = apex - cej
    root_length = float(np.linalg.norm(root))
    if not np.isfinite(root_length) or root_length < minimum_root_pixels:
        return {"surface": surface, "status": "not_assessable", "reason": "invalid_root_length"}
    bone_vector = bone - cej
    bone_distance = float(np.linalg.norm(bone_vector))
    projection_ratio = float(np.dot(bone_vector, root) / (root_length ** 2))
    rbl_percent = 100.0 * bone_distance / root_length
    if not all(np.isfinite(value) for value in (bone_distance, projection_ratio, rbl_percent)):
        return {"surface": surface, "status": "not_assessable", "reason": "non_finite_geometry"}
    if projection_ratio < 0 or projection_ratio > maximum_projection_ratio:
        return {"surface": surface, "status": "not_assessable", "reason": "implausible_bone_projection",
                "projection_ratio": projection_ratio}
    return {"surface": surface, "status": "assessable", "rbl_percent": rbl_percent,
            "root_length_pixels": root_length, "cej_to_bone_pixels": bone_distance,
            "projection_ratio": projection_ratio}
