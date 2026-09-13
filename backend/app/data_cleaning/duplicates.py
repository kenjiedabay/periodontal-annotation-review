"""Exact and perceptual duplicate detection."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DuplicateMatch:
    status: str
    reason: str = ""


@dataclass(frozen=True)
class _SeenImage:
    relative_name: str
    perceptual_hash: np.ndarray


def _perceptual_hash(image: np.ndarray) -> np.ndarray:
    grayscale = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(grayscale, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)[:8, :8]
    values = dct[1:, 1:]
    return values > np.median(values)


def _hamming_distance(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


class DuplicateDetector:
    """Track already-scanned files and report matches without removing files."""

    def __init__(self, perceptual_distance: int = 6) -> None:
        self.perceptual_distance = perceptual_distance
        self._exact: dict[str, str] = {}
        self._seen: list[_SeenImage] = []

    def check(self, path: Path, data: bytes, image: np.ndarray) -> DuplicateMatch:
        relative_name = path.as_posix()
        exact_hash = hashlib.sha256(data).hexdigest()
        previous_exact = self._exact.get(exact_hash)
        image_hash = _perceptual_hash(image)
        self._exact.setdefault(exact_hash, relative_name)
        if previous_exact is not None:
            self._seen.append(_SeenImage(relative_name, image_hash))
            return DuplicateMatch("exact_duplicate", f"exact_duplicate_of:{previous_exact}")

        for previous in self._seen:
            distance = _hamming_distance(image_hash, previous.perceptual_hash)
            if distance <= self.perceptual_distance:
                self._seen.append(_SeenImage(relative_name, image_hash))
                return DuplicateMatch(
                    "similar_duplicate",
                    f"similar_duplicate_of:{previous.relative_name};perceptual_distance:{distance}",
                )

        self._seen.append(_SeenImage(relative_name, image_hash))
        return DuplicateMatch("unique")
