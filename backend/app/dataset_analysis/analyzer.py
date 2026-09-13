"""Dataset statistics and quality analysis without modifying labels or files."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import ValidationError

from app.annotation_storage import Annotation


@dataclass(frozen=True)
class AnalysisConfig:
    processed_dir: Path
    annotations_dir: Path
    output_dir: Path
    minimum_class_samples: int = 10
    imbalance_ratio_threshold: float = 3.0


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    filename: str
    width: int | None
    height: int | None
    annotation_status: str
    disease_status: str | None
    expert_validated: bool | None
    affected_teeth_count: int
    affected_teeth: str
    severity: str | None
    findings: str
    completeness_status: str
    invalid_reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _image_id(path: Path) -> str:
    return path.stem


def _annotation_path(annotations_dir: Path, image_id: str) -> Path:
    return annotations_dir / f"{image_id}.json"


def _read_image_size(path: Path) -> tuple[int | None, int | None, str | None]:
    try:
        with Image.open(path) as image:
            return image.width, image.height, None
    except (OSError, ValueError) as error:
        return None, None, f"image_read_error:{type(error).__name__}"


def _validate_annotation(path: Path) -> tuple[Annotation | None, str | None]:
    try:
        return Annotation.model_validate_json(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        return None, f"invalid_annotation:{type(error).__name__}"


def _completeness(annotation: Annotation | None) -> str:
    if annotation is None:
        return "missing_or_invalid"
    required = [annotation.disease_status, annotation.severity, annotation.findings]
    return "complete" if all(value is not None for value in required) else "incomplete"


def _record(image_path: Path, annotations_dir: Path) -> ImageRecord:
    image_id = _image_id(image_path)
    width, height, image_error = _read_image_size(image_path)
    annotation_path = _annotation_path(annotations_dir, image_id)
    if not annotation_path.exists():
        return ImageRecord(image_id, image_path.name, width, height, "missing", None, None, 0, "", None, "", "missing", image_error or "annotation_missing")

    annotation, annotation_error = _validate_annotation(annotation_path)
    if annotation is None:
        return ImageRecord(image_id, image_path.name, width, height, "invalid", None, None, 0, "", None, "", "invalid", annotation_error or image_error or "annotation_invalid")
    if annotation.image_id != image_id:
        return ImageRecord(image_id, image_path.name, width, height, "invalid", None, annotation.expert_validated, 0, "", None, "", "invalid", "annotation_image_id_mismatch")

    completeness = _completeness(annotation)
    return ImageRecord(
        image_id=image_id,
        filename=image_path.name,
        width=width,
        height=height,
        annotation_status="valid",
        disease_status=annotation.disease_status,
        expert_validated=annotation.expert_validated,
        affected_teeth_count=len(annotation.affected_teeth),
        affected_teeth=";".join(annotation.affected_teeth),
        severity=annotation.severity,
        findings=";".join(annotation.findings),
        completeness_status=completeness,
        invalid_reason=image_error or "",
    )


def _distribution(records: list[ImageRecord], attribute: str) -> dict[str, int]:
    values = [getattr(record, attribute) for record in records if getattr(record, attribute) not in (None, "")]
    return dict(sorted(Counter(values).items()))


def _tooth_distribution(records: list[ImageRecord]) -> dict[str, int]:
    teeth: Counter[str] = Counter()
    for record in records:
        if record.annotation_status == "valid":
            teeth.update(tooth for tooth in record.affected_teeth.split(";") if tooth)
    return dict(sorted(teeth.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else 999, item[0])))


def _class_warnings(distribution: dict[str, int], config: AnalysisConfig) -> list[str]:
    warnings: list[str] = []
    if not distribution:
        return ["No valid disease labels available; supervised disease classification is not currently feasible."]
    smallest = min(distribution.values())
    largest = max(distribution.values())
    if smallest < config.minimum_class_samples:
        warnings.append(f"At least one disease class has fewer than {config.minimum_class_samples} samples.")
    if smallest and largest / smallest >= config.imbalance_ratio_threshold:
        warnings.append(f"Disease class imbalance ratio is {largest / smallest:.2f}:1, above the {config.imbalance_ratio_threshold:.1f}:1 review threshold.")
    if len(distribution) < 2:
        warnings.append("Only one disease class is represented; binary classification cannot be assessed.")
    return warnings


def _feasibility(records: list[ImageRecord], disease_distribution: dict[str, int], warnings: list[str]) -> dict[str, Any]:
    valid = [record for record in records if record.annotation_status == "valid"]
    validated = [record for record in valid if record.expert_validated]
    return {
        "disease_classification": "review required" if warnings else "potentially feasible for a baseline only",
        "tooth_localization": "potentially feasible for exploratory work" if any(record.affected_teeth_count for record in validated) else "not currently supported by validated tooth labels",
        "severity_prediction": "insufficient evidence" if len(set(record.severity for record in validated if record.severity)) < 2 else "exploratory only; check per-class sample counts",
        "recommendation": "Use the report to decide task scope; do not infer clinical performance from these counts alone.",
        "valid_annotations": len(valid),
        "expert_validated_annotations": len(validated),
        "disease_classes": disease_distribution,
    }


def analyze_dataset(config: AnalysisConfig) -> dict[str, Any]:
    """Analyze processed images and annotations without changing either input directory."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in config.processed_dir.glob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
    records = [_record(path, config.annotations_dir) for path in image_paths]
    image_ids = {_image_id(path) for path in image_paths}
    orphan_annotation_files = 0
    orphan_invalid_annotations = 0
    if config.annotations_dir.exists():
        for annotation_path in config.annotations_dir.glob("*.json"):
            if annotation_path.stem in image_ids:
                continue
            orphan_annotation_files += 1
            annotation, _ = _validate_annotation(annotation_path)
            if annotation is None:
                orphan_invalid_annotations += 1
    disease_distribution = _distribution(records, "disease_status")
    severity_distribution = _distribution(records, "severity")
    findings_distribution: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    for record in records:
        if record.findings:
            findings_distribution.update(record.findings.split(";"))
        if record.width and record.height:
            dimensions[f"{record.width}x{record.height}"] += 1
    warnings = _class_warnings(disease_distribution, config)
    missing = sum(record.annotation_status == "missing" for record in records)
    invalid = sum(record.annotation_status == "invalid" for record in records) + orphan_invalid_annotations
    summary = {
        "total_images": len(records),
        "disease_positive_images": disease_distribution.get("present", 0),
        "disease_negative_images": disease_distribution.get("absent", 0),
        "uncertain_images": disease_distribution.get("uncertain", 0),
        "number_of_affected_teeth": sum(record.affected_teeth_count for record in records),
        "distribution_by_tooth_number": _tooth_distribution(records),
        "severity_distribution": severity_distribution,
        "radiographic_findings_distribution": dict(sorted(findings_distribution.items())),
        "image_dimensions": dict(sorted(dimensions.items())),
        "class_imbalance": {"disease_classes": disease_distribution, "warnings": warnings},
        "missing_annotations": missing,
        "invalid_annotations": invalid,
        "orphan_annotation_files": orphan_annotation_files,
        "incomplete_annotations": sum(record.completeness_status == "incomplete" for record in records),
        "valid_annotations": sum(record.annotation_status == "valid" for record in records),
        "expert_validated_annotations": sum(record.expert_validated is True for record in records),
        "potential_problems": warnings + ([f"{missing} processed images have no annotation file."] if missing else []) + ([f"{invalid} annotation files failed validation."] if invalid else []) + ([f"{orphan_annotation_files} annotation files do not match a processed image."] if orphan_annotation_files else []),
        "feasibility": _feasibility(records, disease_distribution, warnings),
    }
    return {"config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}, "summary": summary, "records": [record.as_dict() for record in records]}
