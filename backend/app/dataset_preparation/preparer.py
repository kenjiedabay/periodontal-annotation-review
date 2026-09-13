"""Prepare leakage-aware train, validation, and test datasets."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Any

from PIL import Image
from pydantic import ValidationError

from app.annotation_storage import Annotation

from .augment import augment_for_training
from .config import PreparationConfig
from .manifest import write_manifests


@dataclass(frozen=True)
class PreparationSummary:
    eligible_annotations: int
    train_images: int
    validation_images: int
    test_images: int
    augmented_training_images: int
    skipped_missing_images: int
    skipped_invalid_annotations: int
    skipped_unvalidated_annotations: int
    case_groups: int
    split_case_groups: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_group_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    values = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(values, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in values.items()):
        raise ValueError("group map must be a JSON object mapping image_id to case/group ID")
    return values


def _load_annotation(path: Path) -> Annotation | None:
    try:
        return Annotation.model_validate_json(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError):
        return None


def _assign_splits(group_ids: list[str], config: PreparationConfig) -> dict[str, str]:
    rng = random.Random(config.seed)
    shuffled = list(group_ids)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = round(total * config.train_ratio)
    validation_end = train_end + round(total * config.validation_ratio)
    if total and train_end == 0:
        train_end = 1
    if validation_end > total:
        validation_end = total
    return {group_id: "train" if index < train_end else "validation" if index < validation_end else "test" for index, group_id in enumerate(shuffled)}


def _copy_original(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _manifest_record(image_path: Path, image_id: str, group_id: str, annotation: Annotation, split: str, dataset_root: Path, annotation_path: Path) -> dict[str, Any]:
    return {
        "image_path": image_path.relative_to(dataset_root).as_posix(),
        "image_id": image_id,
        "group_id": group_id,
        "label": annotation.disease_status,
        "affected_tooth": ";".join(annotation.affected_teeth),
        "severity": annotation.severity,
        "annotation_path": (Path("annotations") / annotation_path.name).as_posix(),
        "split": split,
    }


def prepare_dataset(config: PreparationConfig) -> tuple[list[dict[str, Any]], PreparationSummary, tuple[Path, Path]]:
    """Copy eligible validated images and augment training copies only."""
    config.validate()
    group_map = _load_group_map(config.group_map_path)
    dataset_root = config.output_dir
    records: list[dict[str, Any]] = []
    eligible: list[tuple[Path, Path, Annotation, str, str]] = []
    skipped_missing = 0
    skipped_invalid = 0
    skipped_unvalidated = 0
    for split in ("train", "validation", "test"):
        (config.output_dir / split).mkdir(parents=True, exist_ok=True)

    for annotation_path in sorted(config.annotations_dir.glob("*.json")) if config.annotations_dir.exists() else []:
        annotation = _load_annotation(annotation_path)
        if annotation is None:
            skipped_invalid += 1
            continue
        image_id = annotation.image_id
        source_candidates = [config.processed_dir / f"{image_id}.png", config.processed_dir / f"{image_id}.jpg", config.processed_dir / f"{image_id}.jpeg"]
        source = next((candidate for candidate in source_candidates if candidate.exists()), None)
        if source is None:
            skipped_missing += 1
            continue
        if config.require_expert_validated and not (annotation.expert_validated and annotation.validation_status == "validated"):
            skipped_unvalidated += 1
            continue
        group_id = group_map.get(image_id, image_id)
        eligible.append((source, annotation_path, annotation, image_id, group_id))

    split_by_group = _assign_splits(sorted({item[4] for item in eligible}), config)
    rng = random.Random(config.seed)
    augmented_count = 0
    for source, annotation_path, annotation, image_id, group_id in eligible:
        split = split_by_group[group_id]
        relative_name = f"{image_id}.png"
        destination = config.output_dir / split / relative_name
        with Image.open(source) as image:
            image = image.convert("L")
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="PNG")
            records.append(_manifest_record(destination, image_id, group_id, annotation, split, dataset_root, annotation_path))
            if split == "train":
                for augmentation_index in range(config.augmentations_per_training_image):
                    augmented = augment_for_training(image, config, rng)
                    augmented_name = f"{image_id}__aug{augmentation_index + 1:02d}.png"
                    augmented_path = config.output_dir / split / augmented_name
                    augmented.image.save(augmented_path, format="PNG")
                    records.append(_manifest_record(augmented_path, image_id, group_id, annotation, split, dataset_root, annotation_path))
                    augmented_count += 1

    split_counts = Counter(record["split"] for record in records)
    summary = PreparationSummary(
        eligible_annotations=len(eligible),
        train_images=split_counts.get("train", 0),
        validation_images=split_counts.get("validation", 0),
        test_images=split_counts.get("test", 0),
        augmented_training_images=augmented_count,
        skipped_missing_images=skipped_missing,
        skipped_invalid_annotations=skipped_invalid,
        skipped_unvalidated_annotations=skipped_unvalidated,
        case_groups=len(split_by_group),
        split_case_groups={split: sum(value == split for value in split_by_group.values()) for split in ("train", "validation", "test")},
    )
    manifests_dir = config.output_dir / "manifests"
    paths = write_manifests(records, manifests_dir, config.as_dict(), summary.as_dict())
    (manifests_dir / "preparation_summary.json").write_text(json.dumps(summary.as_dict(), indent=2), encoding="utf-8")
    (manifests_dir / "preparation_config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    return records, summary, paths
