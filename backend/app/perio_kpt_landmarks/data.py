"""Read-only parser and deterministic preparation for Perio-KPT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

from .geometry import (LetterboxTransform, expand_box, image_to_crop,
                       make_letterbox_transform, normalized_box_to_xyxy)
from .schema import (EXPECTED_ROW_VALUES, LANDMARK_NAMES, OBJECT_CLASS_NAMES,
                     VALID_VISIBILITY)


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: int

    @property
    def available(self) -> bool:
        return self.visibility > 0


@dataclass(frozen=True)
class Annotation:
    image_id: str
    object_index: int
    class_id: int
    normalized_box: tuple[float, float, float, float]
    landmarks: tuple[Landmark, ...]
    source_label: Path
    source_line: int

    @property
    def record_id(self) -> str:
        return f"{self.image_id}:object-{self.object_index}"


class AnnotationError(ValueError):
    def __init__(self, reason: str, *, token_count: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.token_count = token_count


def parse_row(line: str, *, image_id: str, object_index: int, source_label: Path,
              source_line: int) -> Annotation:
    values = line.strip().split()
    if len(values) != EXPECTED_ROW_VALUES:
        raise AnnotationError("unexpected_value_count", token_count=len(values))
    try:
        numbers = [float(value) for value in values]
    except ValueError as error:
        raise AnnotationError("non_numeric_value") from error
    class_value = numbers[0]
    if not class_value.is_integer() or int(class_value) not in OBJECT_CLASS_NAMES:
        raise AnnotationError("invalid_class_id")
    box = tuple(numbers[1:5])
    cx, cy, width, height = box
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise AnnotationError("invalid_normalized_box")
    if cx - width / 2 < 0 or cy - height / 2 < 0 or cx + width / 2 > 1 or cy + height / 2 > 1:
        raise AnnotationError("box_outside_image")
    landmarks = []
    for index in range(len(LANDMARK_NAMES)):
        x, y, raw_visibility = numbers[5 + 3 * index:8 + 3 * index]
        if not raw_visibility.is_integer() or int(raw_visibility) not in VALID_VISIBILITY:
            raise AnnotationError("invalid_visibility")
        visibility = int(raw_visibility)
        if visibility == 0:
            if x != 0 or y != 0:
                raise AnnotationError("unavailable_landmark_has_coordinates")
        elif not (0 <= x <= 1 and 0 <= y <= 1):
            raise AnnotationError("landmark_outside_image")
        landmarks.append(Landmark(x, y, visibility))
    return Annotation(image_id, object_index, int(class_value), box, tuple(landmarks), source_label, source_line)


def parse_label(path: Path) -> tuple[list[Annotation], list[dict]]:
    annotations, quarantined = [], []
    object_index = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        object_index += 1
        try:
            annotations.append(parse_row(line, image_id=path.stem, object_index=object_index,
                                         source_label=path, source_line=line_number))
        except AnnotationError as error:
            quarantined.append({
                "record_id": f"{path.stem}:object-{object_index}",
                "image_id": path.stem,
                "source_label": str(path),
                "source_line": line_number,
                "reason": error.reason,
                "observed_value_count": error.token_count,
                "expected_value_count": EXPECTED_ROW_VALUES,
                "source_preserved": True,
            })
    return annotations, quarantined


def load_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"unreadable image: {path}")
    return image


def prepare_annotation(annotation: Annotation, image_width: int, image_height: int,
                       *, padding: float = 0.20, output_size: int = 256) -> tuple[LetterboxTransform, list[tuple[float, float] | None]]:
    box = normalized_box_to_xyxy(annotation.normalized_box, image_width, image_height)
    expanded = expand_box(box, image_width, image_height, padding)
    transform = make_letterbox_transform(expanded, image_width, image_height, output_size)
    points: list[tuple[float, float] | None] = []
    for landmark in annotation.landmarks:
        points.append(image_to_crop((landmark.x * image_width, landmark.y * image_height), transform)
                      if landmark.available else None)
    return transform, points


def annotation_points_original(annotation: Annotation, width: int, height: int) -> list[tuple[float, float] | None]:
    return [(landmark.x * width, landmark.y * height) if landmark.available else None
            for landmark in annotation.landmarks]


def find_image(image_id: str, directories: Iterable[Path]) -> Path | None:
    for directory in directories:
        candidate = directory / f"{image_id}.png"
        if candidate.is_file():
            return candidate
    return None
