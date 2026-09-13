"""Manifest-backed dataset with validation and leakage checks."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from app.annotation_storage import Annotation
from ml.config import TrainingConfig


@dataclass(frozen=True)
class ManifestRecord:
    image_path: str
    image_id: str
    label: int
    label_name: str
    annotation_path: str
    split: str
    group_id: str


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Manifest must contain a records list")
    return records


def validate_manifest(records: list[dict[str, Any]], config: TrainingConfig) -> None:
    image_splits: dict[str, str] = {}
    groups: dict[str, set[str]] = {}
    for record in records:
        split = str(record.get("split", ""))
        image_id = str(record.get("image_id", ""))
        group_id = str(record.get("group_id", image_id))
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split in manifest: {split}")
        if not image_id:
            raise ValueError("Every manifest record requires image_id")
        previous_split = image_splits.get(image_id)
        if previous_split is not None and previous_split != split:
            raise ValueError(f"Image ID appears across multiple splits: {image_id}")
        image_splits[image_id] = split
        groups.setdefault(group_id, set()).add(split)
    leaked_groups = [group for group, group_splits in groups.items() if len(group_splits) > 1]
    if leaked_groups:
        raise ValueError(f"Case groups appear across multiple splits: {leaked_groups[:5]}")


def read_records(config: TrainingConfig) -> list[ManifestRecord]:
    raw_records = _load_manifest(config.manifest_path)
    validate_manifest(raw_records, config)
    records: list[ManifestRecord] = []
    for raw in raw_records:
        label_name = str(raw.get("label", "")).lower().replace(" ", "_")
        if label_name not in {config.positive_label, config.negative_label}:
            raise ValueError(f"Binary classifier received unsupported label: {label_name}")
        records.append(ManifestRecord(
            image_path=str(raw["image_path"]),
            image_id=str(raw["image_id"]),
            label=int(label_name == config.positive_label),
            label_name=label_name,
            annotation_path=str(raw.get("annotation_path", "")),
            split=str(raw["split"]),
            group_id=str(raw.get("group_id", raw["image_id"])),
        ))
    return records


def _transform(config: TrainingConfig, training: bool) -> transforms.Compose:
    operations: list[Any] = [transforms.Grayscale(num_output_channels=3)]
    if training:
        operations.extend([
            transforms.RandomResizedCrop(config.image_size, scale=(0.90, 1.0), ratio=(0.97, 1.03)),
            transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02), fill=0),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
        ])
    else:
        operations.append(transforms.Resize((config.image_size, config.image_size)))
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
    ])
    return transforms.Compose(operations)


class RadiographDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[ManifestRecord], config: TrainingConfig, split: str) -> None:
        self.records = [record for record in records if record.split == split]
        self.root = config.dataset_root
        self.transform = _transform(config, training=split == "train")
        if not self.records:
            raise ValueError(f"No records available for {split} split")
        for record in self.records:
            image_path = self.root / record.image_path
            annotation_path = self.root / record.annotation_path if record.annotation_path else None
            if not image_path.exists():
                raise FileNotFoundError(f"Manifest image is missing: {image_path}")
            if annotation_path is not None and not annotation_path.exists():
                raise FileNotFoundError(f"Manifest annotation is missing: {annotation_path}")
            if annotation_path is not None:
                annotation = Annotation.model_validate_json(annotation_path.read_text(encoding="utf-8-sig"))
                if annotation.image_id != record.image_id or annotation.validation_status != "validated" or not annotation.expert_validated:
                    raise ValueError(f"Annotation is not expert-validated for image {record.image_id}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(self.root / record.image_path) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, torch.tensor(record.label, dtype=torch.long)
