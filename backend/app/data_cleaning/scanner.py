"""Non-destructive dataset scanner and cleaner."""

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .duplicates import DuplicateDetector
from .quality import QualityThresholds, inspect_image
from .report import write_reports

SUPPORTED_FORMATS = {"JPEG", "PNG", "TIFF", "BMP", "WEBP"}


@dataclass(frozen=True)
class ScanConfig:
    raw_dir: Path
    cleaned_dir: Path
    reports_dir: Path
    thresholds: QualityThresholds = QualityThresholds()
    perceptual_distance: int = 6


@dataclass(frozen=True)
class ScanSummary:
    files_discovered: int
    valid_images: int
    invalid_images: int
    copied_images: int
    exact_duplicates: int
    similar_duplicates: int
    flagged_for_review: int
    unflagged_by_automated_checks: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _image_id(relative_path: Path) -> str:
    return relative_path.with_suffix("").as_posix().replace("/", "__")


def _base_record(relative_path: Path, file_size: int) -> dict[str, Any]:
    relative_name = relative_path.as_posix()
    return {
        "image_id": _image_id(relative_path),
        "filename": relative_name,
        "width": None,
        "height": None,
        "format": None,
        "channels": None,
        "file_size": file_size,
        "blur_score": None,
        "brightness_score": None,
        "contrast_score": None,
        "duplicate_status": "not_evaluated",
        "quality_status": "flagged_for_review",
        "manual_review_required": True,
        "reason": "",
    }


def _invalid_record(relative_path: Path, file_size: int, reason: str) -> dict[str, Any]:
    record = _base_record(relative_path, file_size)
    record["reason"] = reason
    return record


def _scan_file(path: Path, relative_path: Path, detector: DuplicateDetector, thresholds: QualityThresholds) -> tuple[dict[str, Any], bool]:
    file_size = path.stat().st_size
    try:
        with Image.open(path) as candidate:
            image_format = (candidate.format or "").upper()
            if image_format not in SUPPORTED_FORMATS:
                return _invalid_record(relative_path, file_size, f"unsupported_format:{image_format or 'unknown'}"), False
            candidate.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            image_shape = np.asarray(image).shape
            channels = image_shape[-1] if len(image_shape) == 3 else 1
            image_array = np.asarray(image.convert("RGB"))
            duplicate = detector.check(relative_path, path.read_bytes(), image_array)
            quality = inspect_image(image, thresholds)
    except UnidentifiedImageError:
        return _invalid_record(relative_path, file_size, "unsupported_format_or_unreadable_image"), False
    except (OSError, ValueError) as error:
        return _invalid_record(relative_path, file_size, f"corrupted_or_unreadable_image:{type(error).__name__}"), False

    reasons = list(quality.reasons)
    if duplicate.reason:
        reasons.insert(0, duplicate.reason)
    manual_review = quality.manual_review_required or duplicate.status != "unique"
    record = {
        "image_id": _image_id(relative_path),
        "filename": relative_path.as_posix(),
        "width": width,
        "height": height,
        "format": image_format,
        "channels": channels,
        "file_size": file_size,
        "blur_score": quality.blur_score,
        "brightness_score": quality.brightness_score,
        "contrast_score": quality.contrast_score,
        "duplicate_status": duplicate.status,
        "quality_status": "flagged_for_review" if manual_review else quality.quality_status,
        "manual_review_required": manual_review,
        "reason": ";".join(reasons),
    }
    return record, True


def scan_dataset(config: ScanConfig) -> tuple[list[dict[str, Any]], ScanSummary, tuple[Path, Path]]:
    """Inspect every file below raw_dir and copy valid files without deleting raw data."""
    config.cleaned_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    if not config.raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {config.raw_dir}")

    paths = sorted(path for path in config.raw_dir.rglob("*") if path.is_file())
    detector = DuplicateDetector(config.perceptual_distance)
    records: list[dict[str, Any]] = []
    copied_images = 0
    for path in paths:
        relative_path = path.relative_to(config.raw_dir)
        record, valid = _scan_file(path, relative_path, detector, config.thresholds)
        records.append(record)
        if valid:
            destination = config.cleaned_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied_images += 1

    summary = ScanSummary(
        files_discovered=len(records),
        valid_images=sum(record["format"] is not None for record in records),
        invalid_images=sum(record["format"] is None for record in records),
        copied_images=copied_images,
        exact_duplicates=sum(record["duplicate_status"] == "exact_duplicate" for record in records),
        similar_duplicates=sum(record["duplicate_status"] == "similar_duplicate" for record in records),
        flagged_for_review=sum(record["manual_review_required"] for record in records),
        unflagged_by_automated_checks=sum(not record["manual_review_required"] for record in records),
    )
    report_paths = write_reports(records, summary.as_dict(), config.reports_dir)
    return records, summary, report_paths
