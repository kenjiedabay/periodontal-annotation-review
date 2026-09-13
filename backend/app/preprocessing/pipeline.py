"""Run configured preprocessing over a cleaned image directory."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from .config import PreprocessingConfig
from .metadata import ImageMetadata, write_metadata
from .transforms import apply_transforms


@dataclass(frozen=True)
class PreprocessingSummary:
    files_discovered: int
    processed_images: int
    skipped_images: int
    comparison_images: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _image_id(relative_path: Path) -> str:
    return relative_path.with_suffix("").as_posix().replace("/", "__")


def _size_text(image: np.ndarray) -> str:
    height, width = image.shape[:2]
    return f"{width}x{height}"


def _comparison(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    def to_color(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    left = to_color(original)
    right = to_color(processed)
    target_height = max(left.shape[0], right.shape[0])

    def fit_height(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == target_height:
            return image
        width = max(1, round(image.shape[1] * target_height / image.shape[0]))
        return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)

    left = fit_height(left)
    right = fit_height(right)
    cv2.putText(left, "Original Image", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(right, "Processed Image", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2, cv2.LINE_AA)
    divider = np.full((target_height, 4, 3), 220, dtype=np.uint8)
    return cv2.hconcat([left, divider, right])


def process_dataset(input_dir: Path, output_dir: Path, config: PreprocessingConfig) -> tuple[list[ImageMetadata], PreprocessingSummary]:
    """Process all readable images without modifying input_dir."""
    config.validate()
    if not input_dir.exists():
        raise FileNotFoundError(f"Cleaned dataset directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    comparison_dir = output_dir / "comparisons"
    records: list[ImageMetadata] = []
    skipped_images = 0
    comparison_images = 0
    paths = sorted(path for path in input_dir.rglob("*") if path.is_file())

    for source_path in paths:
        original = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if original is None:
            skipped_images += 1
            continue
        relative_path = source_path.relative_to(input_dir)
        processed, operations, normalization = apply_transforms(original, config)
        operations.insert(0, "format_standardized_to:PNG")
        output_relative = relative_path.with_suffix(".png")
        output_path = output_dir / output_relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), processed):
            skipped_images += 1
            continue

        image_id = _image_id(relative_path)
        metadata = ImageMetadata(
            image_id=image_id,
            original_size=_size_text(original),
            processed_size=_size_text(processed),
            operations=operations,
            normalization=normalization,
            preprocessing_version=config.preprocessing_version,
        )
        write_metadata(metadata, metadata_dir / output_relative.with_suffix(".json"))
        records.append(metadata)

        if config.create_comparisons:
            comparison_path = comparison_dir / output_relative
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(comparison_path), _comparison(original, processed)):
                comparison_images += 1

    (output_dir / "preprocessing_config.json").write_text(
        json.dumps(config.as_dict(), indent=2), encoding="utf-8"
    )
    (metadata_dir / "preprocessing_metadata.json").write_text(
        json.dumps([record.as_dict() for record in records], indent=2), encoding="utf-8"
    )
    summary = PreprocessingSummary(
        files_discovered=len(paths),
        processed_images=len(records),
        skipped_images=skipped_images,
        comparison_images=comparison_images,
    )
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary.as_dict(), indent=2), encoding="utf-8"
    )
    return records, summary
