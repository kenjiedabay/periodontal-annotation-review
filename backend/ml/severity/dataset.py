"""Manifest and expert-validated severity dataset."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from app.annotation_storage import Annotation
from ml.datasets.radiograph import validate_manifest
from .config import SeverityConfig


@dataclass(frozen=True)
class SeverityRecord:
    image_path: str
    image_id: str
    severity: str
    class_index: int
    annotation_path: str
    split: str
    group_id: str


def _manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Manifest must contain a records list")
    return records


def read_severity_records(config: SeverityConfig) -> tuple[list[SeverityRecord], list[str], dict[str, int]]:
    raw_records = _manifest(config.manifest_path)
    validate_manifest(raw_records, _training_config_adapter(config))
    candidates: list[tuple[dict[str, Any], Annotation]] = []
    skipped = 0
    for raw in raw_records:
        annotation_path_text = str(raw.get("annotation_path", ""))
        annotation_path = config.dataset_root / annotation_path_text
        if not annotation_path.exists():
            skipped += 1
            continue
        try:
            annotation = Annotation.model_validate_json(annotation_path.read_text(encoding="utf-8-sig"))
        except Exception:
            skipped += 1
            continue
        if not annotation.expert_validated or annotation.validation_status != "validated":
            skipped += 1
            continue
        if not annotation.severity:
            continue
        if annotation.image_id != str(raw.get("image_id", "")):
            raise ValueError(f"Annotation image_id mismatch: {annotation_path}")
        candidates.append((raw, annotation))

    observed_counts: dict[str, int] = {}
    for _, annotation in candidates:
        observed_counts[annotation.severity] = observed_counts.get(annotation.severity, 0) + 1
    observed_labels = sorted(observed_counts)
    if config.allowed_labels:
        requested = {label.strip().lower().replace(" ", "_") for label in config.allowed_labels}
        unknown = requested - set(observed_labels)
        if unknown:
            raise ValueError(f"Requested severity labels are not established in expert-validated data: {sorted(unknown)}")
        observed_labels = [label for label in observed_labels if label in requested]
    label_to_index = {label: index for index, label in enumerate(observed_labels)}
    records = [SeverityRecord(str(raw["image_path"]), annotation.image_id, annotation.severity, label_to_index[annotation.severity], str(raw["annotation_path"]), str(raw["split"]), str(raw.get("group_id", annotation.image_id))) for raw, annotation in candidates if annotation.severity in label_to_index]
    return records, observed_labels, observed_counts


def _training_config_adapter(config: SeverityConfig) -> Any:
    class Adapter:
        pass
    adapter = Adapter()
    adapter.__dict__.update({"positive_label": "present", "negative_label": "absent"})
    return adapter


def _transform(config: SeverityConfig, training: bool) -> transforms.Compose:
    operations: list[Any] = [transforms.Grayscale(num_output_channels=3)]
    if training:
        operations.extend([transforms.RandomResizedCrop(config.image_size, scale=(0.90, 1.0), ratio=(0.97, 1.03)), transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02), fill=0), transforms.ColorJitter(brightness=0.05, contrast=0.05)])
    else:
        operations.append(transforms.Resize((config.image_size, config.image_size)))
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.485] * 3, std=[0.229] * 3)])
    return transforms.Compose(operations)


class SeverityDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[SeverityRecord], config: SeverityConfig, split: str) -> None:
        self.records = [record for record in records if record.split == split]
        self.root = config.dataset_root
        self.transform = _transform(config, split == "train")
        if not self.records:
            raise ValueError(f"No severity records available for {split} split")
        for record in self.records:
            image_path = self.root / record.image_path
            if not image_path.exists():
                raise FileNotFoundError(f"Severity image is missing: {image_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(self.root / record.image_path) as image:
            return self.transform(image.convert("L")), torch.tensor(record.class_index, dtype=torch.long)
